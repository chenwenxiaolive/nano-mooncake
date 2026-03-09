"""
Example: Distributed KVCache Store with nano-mooncake
=====================================================
Demonstrates Mooncake Store's put/get API for KVCache management.

Architecture:
  ┌──────────┐      ┌──────────────┐      ┌──────────┐
  │ Client A  │ ──> │ MasterService │ <── │ Client B  │
  │ (writer)  │     │ (metadata)    │     │ (reader)  │
  └──────────┘      └──────────────┘      └──────────┘
       │                                        │
       └──── Transfer Engine (TCP) ─────────────┘

Flow:
  1. Client A puts KVCache for "request-123:layer-0"
  2. Master allocates buffer on Client B's segment
  3. Client A transfers data to Client B via Transfer Engine
  4. Client B reads KVCache directly from local buffer
"""

import time
import numpy as np
from transfer_engine import MetadataStore
from mooncake_store import MasterService, StoreClient


def main():
    print("=" * 60)
    print("nano-mooncake: Distributed KVCache Store Demo")
    print("=" * 60)

    # Shared metadata and master
    metadata = MetadataStore()
    master = MasterService(default_lease_seconds=5.0)  # Short TTL for demo

    # --- Create two store clients (simulating two nodes) ---
    print("\n[1] Creating store clients...")
    client_a = StoreClient(
        master, metadata, "node-A",
        host="127.0.0.1", port=19200, segment_size=1 * 1024 * 1024,
    )
    print("  Client A ready (node-A @ port 19200)")

    client_b = StoreClient(
        master, metadata, "node-B",
        host="127.0.0.1", port=19201, segment_size=1 * 1024 * 1024,
    )
    print("  Client B ready (node-B @ port 19201)")

    time.sleep(0.3)

    # --- Client A: store KVCache ---
    print("\n[2] Client A: storing KVCache for 'req-123:layer-0'...")
    kv_data = np.random.randn(8, 128, 64).astype(np.float16).tobytes()
    print(f"  KVCache size: {len(kv_data)} bytes")

    t0 = time.time()
    ok = client_a.put("req-123:layer-0", kv_data)
    elapsed = time.time() - t0
    print(f"  Put result: {'success' if ok else 'failed'} ({elapsed * 1000:.1f} ms)")

    # --- Client B: retrieve KVCache ---
    print("\n[3] Client B: retrieving KVCache for 'req-123:layer-0'...")
    t0 = time.time()
    retrieved = client_b.get("req-123:layer-0")
    elapsed = time.time() - t0

    if retrieved and retrieved == kv_data:
        print(f"  Get result: success, data verified! ({elapsed * 1000:.1f} ms)")
    elif retrieved:
        print(f"  Get result: data mismatch!")
    else:
        print(f"  Get result: not found")

    # --- Store multiple layers ---
    print("\n[4] Storing multiple layers...")
    for layer in range(4):
        key = f"req-456:layer-{layer}"
        data = np.random.randn(8, 64, 64).astype(np.float16).tobytes()
        ok = client_a.put(key, data)
        print(f"  {key}: {'ok' if ok else 'failed'} ({len(data)} bytes)")

    # --- Query existence ---
    print("\n[5] Querying objects...")
    for key in ["req-123:layer-0", "req-456:layer-2", "nonexistent"]:
        obj = master.query(key)
        if obj:
            r = obj.replicas[0]
            print(f"  {key}: found on {r.segment_name} "
                  f"(offset={r.offset}, status={r.status.value})")
        else:
            print(f"  {key}: not found")

    # --- Demonstrate lease expiration ---
    print("\n[6] Waiting for lease expiration (5 seconds)...")
    time.sleep(5.5)
    master.run_eviction()

    obj = master.query("req-123:layer-0")
    print(f"  After expiry, 'req-123:layer-0': "
          f"{'found' if obj else 'evicted (as expected)'}")

    print("\n" + "=" * 60)
    print("Demo complete! This showed Mooncake Store's core flow:")
    print("  put(key, data) -> Master allocates + TE transfers -> get(key)")
    print("  Objects auto-evict after lease timeout (TTL)")
    print("=" * 60)

    client_a.shutdown()
    client_b.shutdown()


if __name__ == "__main__":
    main()
