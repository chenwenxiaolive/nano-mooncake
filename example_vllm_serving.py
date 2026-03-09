"""
nano-mooncake Proxy: vLLM Disaggregated Serving Proxy
=====================================================
Simplified version of Mooncake's vllm_v1_proxy_server.py.
Reproduces the architecture from:
  https://kvcache-ai.github.io/Mooncake/performance/vllm-v1-support-benchmark.html

Architecture:
  ┌────────┐       ┌───────────────────┐       ┌────────────────────────┐
  │ Client  │──────>│  Proxy (:8000)    │──────>│ vLLM Prefill (:8010)   │
  │         │       │  round-robin      │       │ --kv-role kv_producer   │
  │         │       │  routing          │       │ --max-tokens 1          │
  │         │       │                   │       └────────┬───────────────┘
  │         │       │                   │                │ MooncakeConnector
  │         │       │                   │                │ (KVCache via RDMA/TCP)
  │         │       │                   │                ▼
  │         │<──────│                   │<───────┌───────┴───────────────┐
  │ (stream)│       │                   │ stream │ vLLM Decode (:8020)   │
  └────────┘       └───────────────────┘        │ --kv-role kv_consumer  │
                                                 └───────────────────────┘

Proxy Flow (same as real Mooncake vllm_v1_proxy_server.py):
  1. Client sends OpenAI-compatible request to Proxy
  2. Proxy -> Prefill: inject kv_transfer_params{do_remote_decode=True},
     set max_tokens=1, stream=False (prefill only, no decoding)
  3. Prefill computes KVCache, returns response with kv_transfer_params
  4. Proxy -> Decode: inject kv_transfer_params from prefill response,
     restore original max_tokens and stream=True
  5. Decode pulls KVCache via MooncakeConnector (Transfer Engine),
     generates tokens, streams back through Proxy to Client

Usage:
  # Step 1: Launch vLLM Prefill instance (kv_producer)
  vllm serve Qwen/Qwen3-8B \\
      --port 8010 \\
      --tensor-parallel-size 8 \\
      --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \\
      --disable-log-requests

  # Step 2: Launch vLLM Decode instance (kv_consumer)
  vllm serve Qwen/Qwen3-8B \\
      --port 8020 \\
      --tensor-parallel-size 8 \\
      --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' \\
      --disable-log-requests

  # Step 3: Start Proxy
  python example_vllm_serving.py \\
      --prefill-url http://PREFILL_HOST:8010 \\
      --decode-url http://DECODE_HOST:8020 \\
      --port 8000

  # Step 4: Send requests (OpenAI-compatible)
  curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}]}'
"""

import argparse
import itertools
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError


# ─────────────────────────────────────────────────────────────────
# Proxy Server — maps to mooncake-wheel/mooncake/vllm_v1_proxy_server.py
# ─────────────────────────────────────────────────────────────────

class DisaggregatedProxy:
    """
    Manages prefill/decode endpoints with round-robin load balancing.
    Maps to the FastAPI app state in vllm_v1_proxy_server.py.
    """

    def __init__(self, prefill_urls: list[str], decode_urls: list[str]):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self._prefill_iter = itertools.cycle(range(len(prefill_urls)))
        self._decode_iter = itertools.cycle(range(len(decode_urls)))

    def next_prefill(self) -> str:
        return self.prefill_urls[next(self._prefill_iter)]

    def next_decode(self) -> str:
        return self.decode_urls[next(self._decode_iter)]


# Global proxy instance (set in main)
proxy: DisaggregatedProxy = None


def send_to_prefill(prefill_url: str, req_data: dict) -> dict:
    """
    Send request to Prefill with do_remote_decode=True, max_tokens=1.
    Maps to vllm_v1_proxy_server.py::send_request_to_service().

    Key modifications (same as real proxy):
      - Force max_tokens=1: prefill only computes KV, no decoding
      - Force stream=False: get complete response with kv_transfer_params
      - Inject kv_transfer_params with do_remote_decode=True
    """
    prefill_req = req_data.copy()
    prefill_req["max_tokens"] = 1
    prefill_req["stream"] = False
    prefill_req["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }

    req = Request(
        f"{prefill_url}/v1/chat/completions",
        data=json.dumps(prefill_req).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def stream_from_decode(decode_url: str, req_data: dict):
    """
    Send request to Decode and yield streaming chunks.
    Maps to vllm_v1_proxy_server.py::stream_service_response().

    Key modifications:
      - Inject kv_transfer_params from prefill response
      - Restore original max_tokens and stream=True
      - Decode pulls KVCache from Prefill via MooncakeConnector internally
    """
    req = Request(
        f"{decode_url}/v1/chat/completions",
        data=json.dumps(req_data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=300) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            yield chunk


class ProxyHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for the disaggregated proxy.
    Routes: Client -> Prefill (compute KV) -> Decode (generate) -> Client.
    Maps to vllm_v1_proxy_server.py::_handle_completions().
    """

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self.send_error(404)
            return

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        # ── Phase 1: Route to Prefill (max_tokens=1, stream=False) ──
        # Prefill computes KVCache for the prompt, returns kv_transfer_params
        prefill_url = proxy.next_prefill()
        try:
            prefill_result = send_to_prefill(prefill_url, body)
        except URLError as e:
            self.send_error(502, f"Prefill error: {e}")
            return

        # ── Phase 2: Extract kv_transfer_params ──
        # These contain the prefill node's address for KVCache transfer
        kv_transfer_params = prefill_result.get("kv_transfer_params", {})

        # ── Phase 3: Route to Decode with kv_transfer_params injected ──
        # Decode will pull KVCache from Prefill via MooncakeConnector,
        # then generate tokens and stream back
        decode_url = proxy.next_decode()
        decode_req = body.copy()
        if kv_transfer_params:
            decode_req["kv_transfer_params"] = kv_transfer_params

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            for chunk in stream_from_decode(decode_url, decode_req):
                self.wfile.write(chunk)
                self.wfile.flush()
        except URLError as e:
            pass  # Connection may close normally

    def log_message(self, format, *args):
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {args[0]}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nano-mooncake: Disaggregated serving proxy for vLLM + Mooncake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Single prefill + single decode (same as benchmark setup)
  python example_vllm_serving.py \\
      --prefill-url http://10.0.28.193:8010 \\
      --decode-url http://10.0.28.202:8020

  # Multiple prefill/decode nodes (round-robin)
  python example_vllm_serving.py \\
      --prefill-url http://node1:8010 http://node2:8010 \\
      --decode-url http://node3:8020 http://node4:8020
        """,
    )
    parser.add_argument(
        "--prefill-url", nargs="+", default=["http://127.0.0.1:8010"],
        help="Prefill vLLM instance URL(s) (default: http://127.0.0.1:8010)",
    )
    parser.add_argument(
        "--decode-url", nargs="+", default=["http://127.0.0.1:8020"],
        help="Decode vLLM instance URL(s) (default: http://127.0.0.1:8020)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Proxy listen port (default: 8000)",
    )

    args = parser.parse_args()

    global proxy
    proxy = DisaggregatedProxy(args.prefill_url, args.decode_url)

    print("=" * 70)
    print("nano-mooncake: Disaggregated Serving Proxy")
    print("=" * 70)
    print(f"\n  Proxy:    http://0.0.0.0:{args.port}")
    print(f"  Prefill:  {', '.join(args.prefill_url)}")
    print(f"  Decode:   {', '.join(args.decode_url)}")
    print(f"\n  Flow: Client -> Proxy -> Prefill (KV) -> Decode (generate) -> Client")
    print(f"\n  Send requests to: http://localhost:{args.port}/v1/chat/completions")
    print("=" * 70)

    server = HTTPServer(("0.0.0.0", args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy...")
        server.shutdown()


if __name__ == "__main__":
    main()
