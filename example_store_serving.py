"""
nano-mooncake: vLLM Disaggregated Serving Demo (Store Mode)
============================================================
Same as example_vllm_serving.py but uses NanoMooncakeStoreConnector
(Mooncake Store layer) instead of NanoMooncakeConnector (P2P Transfer Engine).

Key difference: data movement goes through StoreClient.put()/get() with
MasterService handling centralized allocation, two-phase commit, and
lease-based eviction — matching real Mooncake's Store mode.

Architecture:
                 ┌───────────────────────┐
                 │  Metadata Server      │
                 │  (:8090)              │
                 │  - segment registry   │
                 │  - MasterService      │
                 │    (alloc/track/evict) │
                 └───┬─────────────┬─────┘
              put_start/end    query/get
              register_seg     discover
                 ┌───┴───┐    ┌───┴───┐
  ┌────────┐    │Prefill │    │Decode │
  │ Client  │   │StoreClient  │StoreClient
  │         │   │(:8010) │    │(:8020)│
  └────────┘    └───┬────┘    └───┬───┘
       ↑            │   TCP TE    │
       │            └─────────────┘
  ┌────┴───┐     data transfer via Store
  │ Proxy  │
  │(:8000) │
  └────────┘

Flow:
  1. Metadata server starts with MasterService on :8090
  2. Prefill: save_kv_layer() → StoreClient.put(key, bytes) →
     Master allocates + TE transfers + Master commits
  3. Decode: start_load_kv() → StoreClient.get(key) →
     Master queries replica → TE reads → inject into GPU
  4. Proxy routes: Client → Prefill (max_tokens=1) → Decode (full generation)

Requirements: pip install vllm
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
METADATA_PORT = 8090
METADATA_URL = f"http://127.0.0.1:{METADATA_PORT}"
GPU_MEMORY_UTIL = 0.45


# ─────────────────────────────────────────────────────────────────
# Metadata Server Launcher (with MasterService enabled)
# ─────────────────────────────────────────────────────────────────

def launch_metadata_server() -> subprocess.Popen:
    """Launch the metadata server with MasterService enabled."""
    cmd = [
        sys.executable, "metadata_server.py",
        "--port", str(METADATA_PORT),
        "--enable-master",  # Store mode requires MasterService
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_metadata_server(url: str, timeout: int = 15) -> bool:
    """Wait for metadata server to become ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(f"{url}/health", method="GET")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ─────────────────────────────────────────────────────────────────
# vLLM Instance Launcher (Store mode)
# ─────────────────────────────────────────────────────────────────

def launch_vllm(role: str, port: int, gpu_memory_utilization: float) -> subprocess.Popen:
    """Launch a vLLM serve process with NanoMooncakeStoreConnector."""
    kv_config = json.dumps({
        "kv_connector": "NanoMooncakeStoreConnector",
        "kv_connector_module_path": "nano_store_connector",
        "kv_role": "kv_both",
    })
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "2048",
        "--no-enable-log-requests",
        "--kv-transfer-config", kv_config,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["NANO_METADATA_URL"] = METADATA_URL
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + ":" + env.get("PYTHONPATH", "")
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
# Proxy
# ─────────────────────────────────────────────────────────────────

prefill_urls = [f"http://127.0.0.1:{PREFILL_PORT}"]
decode_urls = [f"http://127.0.0.1:{DECODE_PORT}"]
prefill_iter = itertools.cycle(range(len(prefill_urls)))
decode_iter = itertools.cycle(range(len(decode_urls)))


def send_to_prefill(req_data: dict) -> dict:
    """Route to Prefill: max_tokens=1, stream=False."""
    prefill_req = req_data.copy()
    prefill_req["max_tokens"] = 1
    prefill_req["stream"] = False
    url = prefill_urls[next(prefill_iter)]
    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(prefill_req).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def send_to_decode(req_data: dict) -> dict:
    """Route to Decode: full generation, non-streaming."""
    decode_req = req_data.copy()
    decode_req["stream"] = False
    url = decode_urls[next(decode_iter)]
    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(decode_req).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


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

        # Phase 2: Extract kv_transfer_params (if present)
        kv_transfer_params = prefill_result.get("kv_transfer_params", {})

        # Phase 3: Decode with full generation
        decode_req = body.copy()
        if kv_transfer_params:
            decode_req["kv_transfer_params"] = kv_transfer_params

        try:
            decode_result = send_to_decode(decode_req)
        except URLError as e:
            self.send_error(502, f"Decode error: {e}")
            return

        response_body = json.dumps(decode_result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        pass


# ─────────────────────────────────────────────────────────────────
# Client
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
    print("nano-mooncake: vLLM Disaggregated Serving Demo (Store Mode)")
    print("  with NanoMooncakeStoreConnector (KV transfer via Mooncake Store)")
    print("  MasterService for centralized allocation + lease-based eviction")
    print("=" * 70)

    processes = []

    try:
        # ── Step 1: Launch Metadata Server with MasterService ──
        print(f"\n[1] Launching Metadata Server with MasterService (:{METADATA_PORT})...")
        meta_proc = launch_metadata_server()
        processes.append(meta_proc)

        if not wait_for_metadata_server(METADATA_URL):
            print("    ERROR: Metadata server failed to start")
            meta_proc.terminate()
            out = meta_proc.stdout.read().decode()
            print(out[-2000:])
            return
        print(f"    Metadata server ready at {METADATA_URL} (MasterService enabled)")

        # ── Step 2: Launch vLLM Prefill instance ──
        print(f"\n[2] Launching vLLM Prefill (:{PREFILL_PORT}, model={MODEL})...")
        prefill_proc = launch_vllm("prefill", PREFILL_PORT, GPU_MEMORY_UTIL)
        processes.append(prefill_proc)

        # ── Step 3: Launch vLLM Decode instance ──
        print(f"[3] Launching vLLM Decode (:{DECODE_PORT}, model={MODEL})...")
        decode_proc = launch_vllm("decode", DECODE_PORT, GPU_MEMORY_UTIL)
        processes.append(decode_proc)

        # ── Step 4: Wait for servers to be ready ──
        print("\n[4] Waiting for vLLM servers to initialize...")
        print("    (This may take 30-60 seconds for model loading)")

        if not wait_for_server(f"http://127.0.0.1:{PREFILL_PORT}"):
            print("    ERROR: Prefill server failed to start")
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

        # ── Step 5: Start Proxy ──
        print(f"\n[5] Starting Proxy (:{PROXY_PORT})...")
        proxy_server = HTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
        proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
        proxy_thread.start()
        print(f"    Proxy ready at :{PROXY_PORT}")

        # ── Step 6: Send test requests ──
        print("\n" + "=" * 70)
        print("Sending test requests through the Store-backed pipeline:")
        print("  Client -> Proxy(:8000) -> Prefill(:8010) -> Decode(:8020)")
        print(f"  MasterService + metadata coordination via :{METADATA_PORT}")
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
            for line in reply.strip().split("\n"):
                print(f"   > {line}")

        # ── Summary ──
        print(f"\n{'=' * 70}")
        print("Demo complete! Architecture summary:")
        print()
        print("  NanoMooncakeStoreConnector (this demo):")
        print("    - KV connector: nano_store_connector.NanoMooncakeStoreConnector")
        print("    - Data plane: StoreClient.put()/get() over TCP Transfer Engine")
        print(f"    - Control plane: MasterService via HTTP (:{METADATA_PORT})")
        print("    - Prefill: save_kv_layer() -> StoreClient.put(key, bytes)")
        print("    - Decode: start_load_kv() -> StoreClient.get(key) -> GPU")
        print("    - MasterService: centralized alloc, two-phase commit, lease eviction")
        print()
        print("  vs NanoMooncakeConnector (P2P mode):")
        print("    - Direct TE buffer management + HTTP KV metadata")
        print("    - No centralized allocation or lease management")
        print()
        print("  Real Mooncake (production):")
        print("    - Store mode: MooncakeStoreConnector + RDMA + etcd-backed Master")
        print("    - P2P mode: MooncakeConnector + RDMA + ZMQ")
        print(f"{'=' * 70}")

        proxy_server.shutdown()

    finally:
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("\nAll servers shut down.")


if __name__ == "__main__":
    main()
