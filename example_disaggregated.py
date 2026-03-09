"""
Example: Prefill/Decode Disaggregated Inference with nano-mooncake
===================================================================
Demonstrates the core Mooncake use case: separating Prefill and Decode
into different nodes, with KVCache transferred via Transfer Engine.

Architecture:
  ┌─────────────┐    Transfer Engine    ┌─────────────┐
  │ Prefill Node │ ─── KVCache WRITE ──>│ Decode Node  │
  │ (compute KV) │                      │ (generate)   │
  └─────────────┘                       └─────────────┘
         │                                     │
         └────── Shared MetadataStore ─────────┘

This maps to Mooncake's vLLM/SGLang integration where:
- Prefill cluster handles prompt processing (compute-intensive)
- Decode cluster handles token generation (memory-bound)
- KVCache is transferred between them via Transfer Engine
"""

import time
import numpy as np
from transfer_engine import (
    MetadataStore, TransferEngine, TcpTransport,
    OpCode, TransferRequest,
)


def simulate_prefill(num_layers: int, num_heads: int, head_dim: int, seq_len: int):
    """Simulate prefill computation producing KVCache (random data)."""
    # Shape: [num_layers, 2 (K+V), num_heads, seq_len, head_dim]
    kv_cache = np.random.randn(num_layers, 2, num_heads, seq_len, head_dim).astype(np.float16)
    print(f"  Prefill computed KVCache: {kv_cache.shape}, {kv_cache.nbytes} bytes")
    return kv_cache


def main():
    print("=" * 60)
    print("nano-mooncake: Prefill/Decode Disaggregated Inference Demo")
    print("=" * 60)

    # --- Model config (tiny for demo) ---
    NUM_LAYERS, NUM_HEADS, HEAD_DIM, SEQ_LEN = 4, 8, 64, 128

    # --- Shared metadata store (in real Mooncake: etcd) ---
    metadata = MetadataStore()

    # --- Initialize Prefill Node ---
    print("\n[1] Starting Prefill Node...")
    prefill_engine = TransferEngine()
    prefill_transport = TcpTransport()
    prefill_engine.init("prefill-node", prefill_transport, metadata=metadata, listen_port=19100)
    prefill_buf = prefill_engine.register_local_memory(4 * 1024 * 1024, "127.0.0.1")
    print(f"  Prefill segment registered: 4MB @ port 19100")

    # --- Initialize Decode Node ---
    print("\n[2] Starting Decode Node...")
    decode_engine = TransferEngine()
    decode_transport = TcpTransport()
    decode_engine.init("decode-node", decode_transport, metadata=metadata, listen_port=19101)
    decode_buf = decode_engine.register_local_memory(4 * 1024 * 1024, "127.0.0.1")
    print(f"  Decode segment registered: 4MB @ port 19101")

    time.sleep(0.3)  # Wait for servers to start

    # --- Prefill Phase ---
    print("\n[3] Prefill Phase: computing KVCache...")
    kv_cache = simulate_prefill(NUM_LAYERS, NUM_HEADS, HEAD_DIM, SEQ_LEN)
    kv_bytes = kv_cache.tobytes()

    # Write KVCache into prefill node's local buffer
    prefill_buf[:len(kv_bytes)] = kv_bytes
    print(f"  KVCache staged in prefill buffer ({len(kv_bytes)} bytes)")

    # --- Transfer KVCache: Prefill -> Decode ---
    print("\n[4] Transferring KVCache: Prefill -> Decode via Transfer Engine...")
    t0 = time.time()

    # In real Mooncake, this would be split into per-layer or per-block transfers
    # Here we transfer in one batch for simplicity
    batch_id = prefill_engine.allocate_batch_id(1)
    req = TransferRequest(
        opcode=OpCode.WRITE,
        source_offset=0,
        target_id="decode-node",
        target_offset=0,
        length=len(kv_bytes),
    )
    prefill_engine.submit_transfer(batch_id, [req])
    status = prefill_engine.wait_for_completion(batch_id)

    elapsed = time.time() - t0
    print(f"  Transfer status: {status.value}")
    print(f"  Transfer time: {elapsed * 1000:.1f} ms")
    print(f"  Throughput: {len(kv_bytes) / elapsed / 1024 / 1024:.1f} MB/s")

    # --- Decode Phase: verify KVCache received ---
    print("\n[5] Decode Phase: verifying received KVCache...")
    received = np.frombuffer(bytes(decode_buf[:len(kv_bytes)]), dtype=np.float16)
    received = received.reshape(kv_cache.shape)

    if np.array_equal(kv_cache, received):
        print("  KVCache transfer verified! Data matches perfectly.")
    else:
        print("  ERROR: KVCache mismatch!")

    # --- Simulate decode using received KVCache ---
    print("\n[6] Decode Phase: generating tokens with received KVCache...")
    for i in range(5):
        # In real inference, decode would use KVCache for attention
        token = np.random.randint(0, 32000)
        print(f"  Generated token {i}: {token}")

    print("\n" + "=" * 60)
    print("Demo complete! This showed the core Mooncake flow:")
    print("  Prefill (compute KV) -> Transfer Engine -> Decode (generate)")
    print("=" * 60)

    prefill_engine.shutdown()
    decode_engine.shutdown()


if __name__ == "__main__":
    main()
