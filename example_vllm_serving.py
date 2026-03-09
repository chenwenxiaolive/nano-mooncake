"""
nano-mooncake: End-to-end vLLM Disaggregated Serving Demo
==========================================================
Launches real vLLM instances and routes requests through the disaggregated proxy.

Architecture (same as Mooncake vLLM v1 benchmark):
  ┌────────┐       ┌───────────────────┐       ┌────────────────────────┐
  │ Client  │──────>│  Proxy (:8000)    │──────>│ vLLM Prefill (:8010)   │
  │         │       │                   │       │ max_tokens=1            │
  │         │       │                   │       └────────────────────────┘
  │         │       │                   │                │
  │         │<──────│                   │<───────┌───────┴──────────────┐
  │ (stream)│       │                   │ stream │ vLLM Decode (:8020)  │
  └────────┘       └───────────────────┘        └──────────────────────┘

What this demo shows:
  - Real vLLM instances serving a real LLM (Qwen2.5-0.5B-Instruct)
  - Proxy routing: Prefill (compute KV, max_tokens=1) -> Decode (generate)
  - OpenAI-compatible API throughout the pipeline

Note: Without Mooncake's C++ Transfer Engine installed, KVCache is not
actually transferred via RDMA/TCP between instances. The decode instance
re-computes the prompt internally. The proxy routing logic and API flow
are identical to a real Mooncake deployment — install mooncake and add
--kv-transfer-config to enable actual KV transfer.

Requirements: pip install vllm (already installed via uv)
"""

import json
import os
import signal
import subprocess
import sys
import time
import itertools
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PREFILL_PORT = 8010
DECODE_PORT = 8020
PROXY_PORT = 8000
# Split GPU memory between two instances
GPU_MEMORY_UTIL = 0.45


# ─────────────────────────────────────────────────────────────────
# vLLM Instance Launcher
# ─────────────────────────────────────────────────────────────────

def launch_vllm(role: str, port: int, gpu_memory_utilization: float) -> subprocess.Popen:
    """Launch a vLLM serve process."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "2048",
        "--no-enable-log-requests",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(url: str, timeout: int = 180) -> bool:
    """Wait for a vLLM server to become ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(f"{url}/health", method="GET")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ─────────────────────────────────────────────────────────────────
# Proxy (same logic as example_vllm_serving.py)
# ─────────────────────────────────────────────────────────────────

prefill_urls = [f"http://127.0.0.1:{PREFILL_PORT}"]
decode_urls = [f"http://127.0.0.1:{DECODE_PORT}"]
prefill_iter = itertools.cycle(range(len(prefill_urls)))
decode_iter = itertools.cycle(range(len(decode_urls)))


def send_to_prefill(req_data: dict) -> dict:
    """
    Route to Prefill: max_tokens=1, stream=False.
    In real Mooncake: also injects kv_transfer_params{do_remote_decode=True}.
    """
    prefill_req = req_data.copy()
    prefill_req["max_tokens"] = 1
    prefill_req["stream"] = False
    # In real Mooncake deployment, uncomment:
    # prefill_req["kv_transfer_params"] = {
    #     "do_remote_decode": True, "do_remote_prefill": False,
    #     "remote_engine_id": None, "remote_block_ids": None,
    #     "remote_host": None, "remote_port": None,
    # }

    url = prefill_urls[next(prefill_iter)]
    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(prefill_req).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def stream_from_decode(req_data: dict):
    """Route to Decode and yield streaming chunks."""
    url = decode_urls[next(decode_iter)]
    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(req_data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            yield chunk


class ProxyHandler(BaseHTTPRequestHandler):
    """Disaggregated proxy: Client -> Prefill -> Decode -> Client."""

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self.send_error(404)
            return

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        # Phase 1: Prefill (max_tokens=1)
        try:
            prefill_result = send_to_prefill(body)
        except URLError as e:
            self.send_error(502, f"Prefill error: {e}")
            return

        # Phase 2: Extract kv_transfer_params (if Mooncake is installed)
        kv_transfer_params = prefill_result.get("kv_transfer_params", {})

        # Phase 3: Decode with full generation
        decode_req = body.copy()
        if kv_transfer_params:
            decode_req["kv_transfer_params"] = kv_transfer_params

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        try:
            for chunk in stream_from_decode(decode_req):
                self.wfile.write(chunk)
                self.wfile.flush()
        except URLError:
            pass

    def log_message(self, format, *args):
        pass


# ─────────────────────────────────────────────────────────────────
# Client: send test requests
# ─────────────────────────────────────────────────────────────────

def send_test_request(prompt: str, max_tokens: int = 50) -> dict:
    """Send a chat completion request through the proxy."""
    req_data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    req = Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions",
        data=json.dumps(req_data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("nano-mooncake: vLLM Disaggregated Serving Demo")
    print("=" * 70)

    processes = []

    try:
        # ── Step 1: Launch vLLM Prefill instance ──
        print(f"\n[1] Launching vLLM Prefill (:{PREFILL_PORT}, model={MODEL})...")
        prefill_proc = launch_vllm("prefill", PREFILL_PORT, GPU_MEMORY_UTIL)
        processes.append(prefill_proc)

        # ── Step 2: Launch vLLM Decode instance ──
        print(f"[2] Launching vLLM Decode (:{DECODE_PORT}, model={MODEL})...")
        decode_proc = launch_vllm("decode", DECODE_PORT, GPU_MEMORY_UTIL)
        processes.append(decode_proc)

        # ── Step 3: Wait for servers to be ready ──
        print("\n[3] Waiting for vLLM servers to initialize...")
        print("    (This may take 30-60 seconds for model loading)")

        if not wait_for_server(f"http://127.0.0.1:{PREFILL_PORT}"):
            print("    ERROR: Prefill server failed to start")
            # Print last output for debugging
            prefill_proc.terminate()
            out = prefill_proc.stdout.read().decode()
            print(out[-2000:])
            return
        print(f"    Prefill server ready at :{PREFILL_PORT}")

        if not wait_for_server(f"http://127.0.0.1:{DECODE_PORT}"):
            print("    ERROR: Decode server failed to start")
            decode_proc.terminate()
            out = decode_proc.stdout.read().decode()
            print(out[-2000:])
            return
        print(f"    Decode server ready at :{DECODE_PORT}")

        # ── Step 4: Start Proxy ──
        print(f"\n[4] Starting Proxy (:{PROXY_PORT})...")
        proxy_server = HTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
        proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
        proxy_thread.start()
        print(f"    Proxy ready at :{PROXY_PORT}")

        # ── Step 5: Send test requests ──
        print("\n" + "=" * 70)
        print("Sending test requests through the disaggregated pipeline:")
        print("  Client -> Proxy(:8000) -> Prefill(:8010) -> Decode(:8020)")
        print("=" * 70)

        test_prompts = [
            ("What is 2+2?", 30),
            ("Write a haiku about the moon.", 50),
            ("Explain KV cache in one sentence.", 50),
        ]

        for i, (prompt, max_tok) in enumerate(test_prompts):
            print(f"\n── Request {i+1}: \"{prompt}\" (max_tokens={max_tok})")

            t0 = time.time()
            result = send_test_request(prompt, max_tok)
            elapsed = time.time() - t0

            reply = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")

            print(f"   Response ({elapsed:.1f}s, {prompt_tokens}+{completion_tokens} tokens):")
            # Indent and wrap response
            for line in reply.strip().split("\n"):
                print(f"   > {line}")

        # ── Summary ──
        print(f"\n{'=' * 70}")
        print("Demo complete! Architecture summary:")
        print()
        print("  Current demo (without Mooncake C++ engine):")
        print("    Proxy -> Prefill (max_tokens=1) -> Decode (full generation)")
        print("    KVCache is re-computed by Decode (no transfer)")
        print()
        print("  With Mooncake installed, add to vLLM launch:")
        print(f'    --kv-transfer-config \'{{"kv_connector":"MooncakeConnector",')
        print(f'      "kv_role":"kv_producer"}}\'   # for Prefill')
        print(f'    --kv-transfer-config \'{{"kv_connector":"MooncakeConnector",')
        print(f'      "kv_role":"kv_consumer"}}\'   # for Decode')
        print()
        print("  Then KVCache transfers via RDMA at 87+ GB/s!")
        print(f"{'=' * 70}")

        proxy_server.shutdown()

    finally:
        # Clean up vLLM processes
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("\nAll servers shut down.")


if __name__ == "__main__":
    main()
