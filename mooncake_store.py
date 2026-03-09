"""
nano-mooncake Store
===================
Minimal prototype of Mooncake Store (mooncake-store/).

Core concepts preserved:
- MasterService: centralized metadata manager (allocation, replica tracking, lease)
- StoreClient: client-side API (put/get/query with Transfer Engine)
- ObjectMetadata: per-object state (replicas, lease timeout)
- Replica: single data copy in a segment (status state machine)
- Lease-based eviction: objects auto-expire after TTL
- Two-phase put: put_start (allocate) -> transfer -> put_end (commit)

Simplified away:
- Multi-replica (fixed to 1 replica)
- Disk replicas (memory only)
- 1024-shard locking (single lock)
- CacheLib slab allocator (simple bump allocator)
- Copy/Move background tasks
- Soft-pinning
"""

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from transfer_engine import (
    MetadataStore, OpCode, SegmentDesc, TransferEngine, TransferRequest, TcpTransport,
)


# ---------------------------------------------------------------------------
# Data types (maps to mooncake-store types.h / replica.h)
# ---------------------------------------------------------------------------

class ReplicaStatus(enum.Enum):
    INITIALIZED = "initialized"  # Buffer allocated, not yet written
    PROCESSING = "processing"    # Write in progress
    COMPLETE = "complete"        # Data fully written, readable
    REMOVED = "removed"


@dataclass
class Replica:
    """
    A single copy of data in a segment (maps to mooncake-store Replica).
    Real Mooncake has 3 types: Memory, Disk, LocalDisk. We only keep Memory.
    """
    segment_name: str
    offset: int
    size: int
    status: ReplicaStatus = ReplicaStatus.INITIALIZED


@dataclass
class ObjectMetadata:
    """Per-object metadata tracked by Master (maps to mooncake-store ObjectMetadata)."""
    key: str
    size: int
    replicas: list[Replica] = field(default_factory=list)
    lease_timeout: float = 0.0  # Unix timestamp when lease expires
    created_by: str = ""        # Client segment name


# ---------------------------------------------------------------------------
# Bump Allocator (simplified from CacheLib slab allocator)
# ---------------------------------------------------------------------------

class BumpAllocator:
    """Simple linear allocator for a segment. Real Mooncake uses CacheLib."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.offset = 0
        self._lock = threading.Lock()

    def allocate(self, size: int) -> Optional[int]:
        with self._lock:
            if self.offset + size > self.capacity:
                return None
            alloc_offset = self.offset
            self.offset += size
            return alloc_offset

    def free(self, offset: int, size: int):
        pass  # Simplified: no real deallocation


# ---------------------------------------------------------------------------
# Master Service (maps to mooncake-store master_service.h)
# ---------------------------------------------------------------------------

class MasterService:
    """
    Centralized metadata manager. Handles:
    - Segment registration and buffer allocation
    - Object metadata (key -> replicas mapping)
    - Two-phase put protocol (put_start / put_end)
    - Lease-based auto-eviction

    Real Mooncake shards metadata into 1024 buckets for concurrency.
    We use a single lock for simplicity.
    """

    def __init__(self, default_lease_seconds: float = 300.0):
        self._objects: dict[str, ObjectMetadata] = {}
        self._segments: dict[str, BumpAllocator] = {}
        self._segment_descs: dict[str, SegmentDesc] = {}
        self._lock = threading.Lock()
        self._lease_ttl = default_lease_seconds

    def register_segment(self, desc: SegmentDesc):
        """Register a client's segment for allocation (maps to MountSegment)."""
        with self._lock:
            self._segments[desc.name] = BumpAllocator(desc.size)
            self._segment_descs[desc.name] = desc

    def put_start(self, client_name: str, key: str, size: int) -> Optional[Replica]:
        """
        Phase 1 of put: allocate buffer for the object.
        Returns Replica descriptor so client knows where to write.
        (Maps to MasterService::PutStart)
        """
        with self._lock:
            # Pick a segment to allocate from (round-robin or first-fit)
            for seg_name, allocator in self._segments.items():
                offset = allocator.allocate(size)
                if offset is not None:
                    replica = Replica(
                        segment_name=seg_name, offset=offset,
                        size=size, status=ReplicaStatus.PROCESSING,
                    )
                    if key not in self._objects:
                        self._objects[key] = ObjectMetadata(
                            key=key, size=size, created_by=client_name,
                        )
                    self._objects[key].replicas.append(replica)
                    return replica
            return None  # No space

    def put_end(self, key: str) -> bool:
        """
        Phase 2 of put: mark object as complete and start lease.
        (Maps to MasterService::PutEnd)
        """
        with self._lock:
            obj = self._objects.get(key)
            if obj is None:
                return False
            for replica in obj.replicas:
                if replica.status == ReplicaStatus.PROCESSING:
                    replica.status = ReplicaStatus.COMPLETE
            obj.lease_timeout = time.time() + self._lease_ttl
            return True

    def query(self, key: str) -> Optional[ObjectMetadata]:
        """
        Query object metadata (maps to MasterService::Query).
        Returns replica locations so client can fetch data.
        """
        with self._lock:
            obj = self._objects.get(key)
            if obj is None:
                return None
            # Check lease expiry
            if obj.lease_timeout > 0 and time.time() > obj.lease_timeout:
                self._evict_locked(key)
                return None
            return obj

    def remove(self, key: str) -> bool:
        """Remove an object (maps to MasterService::Remove)."""
        with self._lock:
            return self._evict_locked(key)

    def _evict_locked(self, key: str) -> bool:
        obj = self._objects.pop(key, None)
        if obj is None:
            return False
        for replica in obj.replicas:
            replica.status = ReplicaStatus.REMOVED
            # In real Mooncake, allocator.free() would reclaim space
        return True

    def run_eviction(self):
        """Background eviction of expired leases (simplified)."""
        with self._lock:
            expired = [
                k for k, obj in self._objects.items()
                if obj.lease_timeout > 0 and time.time() > obj.lease_timeout
            ]
            for k in expired:
                self._evict_locked(k)


# ---------------------------------------------------------------------------
# Store Client (maps to mooncake-store client_service.h)
# ---------------------------------------------------------------------------

class StoreClient:
    """
    Client-side KVCache store API. Coordinates:
      1. Master (metadata) for allocation and tracking
      2. Transfer Engine for actual data movement

    Usage:
      client = StoreClient(master, engine, "node-1", port=9100, segment_size=1MB)
      client.put("req-123:layer-0", kv_data)
      data = client.get("req-123:layer-0")
    """

    def __init__(
        self, master: MasterService, metadata: MetadataStore,
        local_name: str, host: str = "127.0.0.1", port: int = 9100,
        segment_size: int = 10 * 1024 * 1024,
    ):
        self.master = master
        self.local_name = local_name
        self.host = host
        self.port = port

        # Initialize Transfer Engine
        self.engine = TransferEngine()
        transport = TcpTransport()
        self.engine.init(local_name, transport, metadata=metadata, listen_port=port)
        self.buffer = self.engine.register_local_memory(segment_size, host)

        # Register segment with Master
        desc = SegmentDesc(name=local_name, host=host, port=port, size=segment_size)
        master.register_segment(desc)

    def put(self, key: str, data: bytes) -> bool:
        """
        Store data under key. Two-phase protocol:
          1. put_start: Master allocates buffer on some segment
          2. Transfer Engine writes data to that segment
          3. put_end: Master marks object as complete
        (Maps to ClientService::Put)
        """
        # Phase 1: Allocate
        replica = self.master.put_start(self.local_name, key, len(data))
        if replica is None:
            return False

        # Write data into local buffer first, then transfer to target segment
        # Find a free spot in our local buffer for staging
        src_offset = 0  # Use beginning of local buffer as staging area
        self.buffer[src_offset:src_offset + len(data)] = data

        if replica.segment_name == self.local_name:
            # Local write — just copy within buffer
            self.buffer[replica.offset:replica.offset + len(data)] = data
        else:
            # Remote write via Transfer Engine
            batch_id = self.engine.allocate_batch_id(1)
            req = TransferRequest(
                opcode=OpCode.WRITE,
                source_offset=src_offset,
                target_id=replica.segment_name,
                target_offset=replica.offset,
                length=len(data),
            )
            self.engine.submit_transfer(batch_id, [req])
            status = self.engine.wait_for_completion(batch_id)
            if status.value != "completed":
                return False

        # Phase 2: Commit
        return self.master.put_end(key)

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve data by key.
          1. Query Master for replica location
          2. Transfer Engine reads data from that segment
        (Maps to ClientService::Get)
        """
        obj = self.master.query(key)
        if obj is None:
            return None

        # Find first complete replica
        replica = None
        for r in obj.replicas:
            if r.status == ReplicaStatus.COMPLETE:
                replica = r
                break
        if replica is None:
            return None

        if replica.segment_name == self.local_name:
            # Local read
            return bytes(self.buffer[replica.offset:replica.offset + replica.size])

        # Remote read via Transfer Engine
        dst_offset = 0  # Read into beginning of local buffer
        batch_id = self.engine.allocate_batch_id(1)
        req = TransferRequest(
            opcode=OpCode.READ,
            source_offset=dst_offset,
            target_id=replica.segment_name,
            target_offset=replica.offset,
            length=replica.size,
        )
        self.engine.submit_transfer(batch_id, [req])
        status = self.engine.wait_for_completion(batch_id)
        if status.value != "completed":
            return None

        return bytes(self.buffer[dst_offset:dst_offset + replica.size])

    def shutdown(self):
        self.engine.shutdown()
