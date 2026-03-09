"""
nano-mooncake Transfer Engine
=============================
Minimal prototype of Mooncake's Transfer Engine (mooncake-transfer-engine/).

Core concepts preserved:
- TransferRequest: describes a single data transfer (READ/WRITE)
- Segment: a registered remote memory region
- BatchDesc: tracks a group of transfers with atomic completion counting
- Transport (abstract) -> TcpTransport: pluggable transport protocol
- TransferEngine: unified API for registering memory, submitting batch transfers

Simplified away:
- RDMA / NVLink / CXL / Ascend transports (TCP only)
- Topology-aware routing, multi-NIC aggregation
- Connection pooling, retry logic
- GPU memory (cudaMemcpy)

TCP Wire Protocol (per transfer):
  [8B size | 8B remote_offset | 1B opcode] + [payload]
"""

import asyncio
import enum
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Core data types (maps to transport.h)
# ---------------------------------------------------------------------------

class OpCode(enum.IntEnum):
    READ = 0   # Pull data from remote into local buffer
    WRITE = 1  # Push data from local to remote buffer

class TransferStatus(enum.Enum):
    WAITING = "waiting"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

HEADER_FMT = "<QQB"  # little-endian: uint64 size, uint64 offset, uint8 opcode
HEADER_SIZE = struct.calcsize(HEADER_FMT)

@dataclass
class SegmentDesc:
    """Describes a remote memory segment (maps to SegmentDesc in transfer_metadata.h)."""
    name: str
    host: str
    port: int
    size: int

@dataclass
class TransferRequest:
    """A single transfer operation (maps to Transport::TransferRequest)."""
    opcode: OpCode
    source_offset: int   # offset within local segment buffer
    target_id: str       # remote segment name
    target_offset: int   # offset within remote segment buffer
    length: int

@dataclass
class TransferTask:
    """Tracks one transfer's status within a batch."""
    request: TransferRequest
    status: TransferStatus = TransferStatus.WAITING

@dataclass
class BatchDesc:
    """Tracks a batch of transfers (maps to Transport::BatchDesc)."""
    batch_id: int
    tasks: list[TransferTask] = field(default_factory=list)
    completed_count: int = 0
    has_failure: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_task_done(self, task_idx: int, success: bool):
        with self._lock:
            self.tasks[task_idx].status = (
                TransferStatus.COMPLETED if success else TransferStatus.FAILED
            )
            if not success:
                self.has_failure = True
            self.completed_count += 1

    @property
    def is_finished(self) -> bool:
        return self.completed_count >= len(self.tasks)

    @property
    def overall_status(self) -> TransferStatus:
        if not self.is_finished:
            return TransferStatus.PENDING
        return TransferStatus.FAILED if self.has_failure else TransferStatus.COMPLETED


# ---------------------------------------------------------------------------
# Transport interface (maps to transport.h abstract class)
# ---------------------------------------------------------------------------

class Transport(ABC):
    """Abstract transport protocol. Real Mooncake has RDMA, NVLink, CXL, etc."""

    @abstractmethod
    def start_server(self, local_buffer: bytearray, port: int):
        """Start listening for incoming transfers."""

    @abstractmethod
    def submit_transfer(
        self, batch: BatchDesc, task_idx: int,
        local_buffer: bytearray, remote: SegmentDesc,
    ):
        """Initiate one transfer asynchronously."""

    @abstractmethod
    def shutdown(self):
        """Stop the transport."""


# ---------------------------------------------------------------------------
# TCP Transport (maps to tcp_transport.cc — the simplest real transport)
# ---------------------------------------------------------------------------

class TcpTransport(Transport):
    """
    TCP-based transport. Each transfer:
      1. Client connects to remote server
      2. Sends header: [size(8B) | offset(8B) | opcode(1B)]
      3. WRITE: client sends payload -> server writes to buffer
         READ:  server sends payload -> client writes to buffer
    """

    def __init__(self):
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start_server(self, local_buffer: bytearray, port: int):
        """Start TCP server in a background thread (maps to TcpContext::acceptor)."""
        self._loop = asyncio.new_event_loop()
        self._local_buffer = local_buffer

        async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                while True:
                    header = await reader.readexactly(HEADER_SIZE)
                    size, offset, opcode = struct.unpack(HEADER_FMT, header)
                    if opcode == OpCode.WRITE:
                        data = await reader.readexactly(size)
                        self._local_buffer[offset:offset + size] = data
                        writer.write(b"\x01")  # ACK
                    else:  # READ
                        data = bytes(self._local_buffer[offset:offset + size])
                        writer.write(data)
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                writer.close()

        async def run_server():
            self._server = await asyncio.start_server(
                handle_connection, "0.0.0.0", port
            )
            await self._server.serve_forever()

        def thread_target():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(run_server())

        self._thread = threading.Thread(target=thread_target, daemon=True)
        self._thread.start()

    def submit_transfer(
        self, batch: BatchDesc, task_idx: int,
        local_buffer: bytearray, remote: SegmentDesc,
    ):
        """Submit one transfer in a background thread."""
        task = batch.tasks[task_idx]
        task.status = TransferStatus.PENDING

        def do_transfer():
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(
                    self._do_transfer_async(task.request, local_buffer, remote)
                )
                loop.close()
                batch.mark_task_done(task_idx, success=True)
            except Exception:
                batch.mark_task_done(task_idx, success=False)

        threading.Thread(target=do_transfer, daemon=True).start()

    async def _do_transfer_async(
        self, req: TransferRequest, local_buffer: bytearray, remote: SegmentDesc
    ):
        reader, writer = await asyncio.open_connection(remote.host, remote.port)
        header = struct.pack(HEADER_FMT, req.length, req.target_offset, req.opcode)
        writer.write(header)
        if req.opcode == OpCode.WRITE:
            data = bytes(local_buffer[req.source_offset:req.source_offset + req.length])
            writer.write(data)
            await writer.drain()
            await reader.readexactly(1)  # wait ACK
        else:  # READ
            await writer.drain()
            data = await reader.readexactly(req.length)
            local_buffer[req.source_offset:req.source_offset + req.length] = data
        writer.close()

    def shutdown(self):
        if self._server and self._loop:
            self._loop.call_soon_threadsafe(self._server.close)


# ---------------------------------------------------------------------------
# Metadata Store (maps to TransferMetadata — backed by etcd/HTTP/Redis)
# ---------------------------------------------------------------------------

class MetadataStore:
    """
    Simplified metadata store. Real Mooncake uses etcd or HTTP metadata server.
    Here we use a shared dict (works in-process) or connect to metadata_server.py.
    """

    def __init__(self):
        self._segments: dict[str, SegmentDesc] = {}
        self._lock = threading.Lock()

    def register_segment(self, desc: SegmentDesc):
        with self._lock:
            self._segments[desc.name] = desc

    def get_segment(self, name: str) -> Optional[SegmentDesc]:
        with self._lock:
            return self._segments.get(name)

    def all_segments(self) -> dict[str, SegmentDesc]:
        with self._lock:
            return dict(self._segments)


# ---------------------------------------------------------------------------
# Transfer Engine (maps to transfer_engine.h — the unified public API)
# ---------------------------------------------------------------------------

class TransferEngine:
    """
    Unified data transfer engine. Core API:
      1. init() — connect to metadata, install transport
      2. register_local_memory() — make local buffer remotely accessible
      3. allocate_batch_id() — create a batch for tracking transfers
      4. submit_transfer() — submit transfer requests
      5. get_transfer_status() — poll completion
    """

    def __init__(self):
        self._transport: Optional[Transport] = None
        self._metadata = MetadataStore()
        self._local_buffer: Optional[bytearray] = None
        self._local_segment_name: Optional[str] = None
        self._batches: dict[int, BatchDesc] = {}
        self._next_batch_id = 0
        self._lock = threading.Lock()

    def init(
        self, local_name: str, transport: Transport,
        metadata: Optional[MetadataStore] = None, listen_port: int = 0,
    ):
        """Initialize engine with a transport and metadata store."""
        self._transport = transport
        if metadata:
            self._metadata = metadata
        self._local_segment_name = local_name
        self._listen_port = listen_port

    def register_local_memory(self, size: int, host: str = "127.0.0.1") -> bytearray:
        """
        Register local memory as a remotely-accessible segment.
        Returns the buffer (maps to TransferEngine::registerLocalMemory).
        """
        self._local_buffer = bytearray(size)
        desc = SegmentDesc(
            name=self._local_segment_name,
            host=host, port=self._listen_port, size=size
        )
        self._metadata.register_segment(desc)
        self._transport.start_server(self._local_buffer, self._listen_port)
        return self._local_buffer

    def open_segment(self, name: str) -> Optional[SegmentDesc]:
        """Look up a remote segment by name (maps to TransferEngine::openSegment)."""
        return self._metadata.get_segment(name)

    def allocate_batch_id(self, batch_size: int) -> int:
        """Allocate a batch ID for tracking transfers (maps to allocateBatchID)."""
        with self._lock:
            bid = self._next_batch_id
            self._next_batch_id += 1
            self._batches[bid] = BatchDesc(batch_id=bid)
        return bid

    def submit_transfer(self, batch_id: int, requests: list[TransferRequest]):
        """
        Submit transfer requests to the batch (maps to submitTransfer).
        Each request is executed asynchronously via the installed transport.
        """
        batch = self._batches[batch_id]
        for req in requests:
            task = TransferTask(request=req)
            batch.tasks.append(task)
            task_idx = len(batch.tasks) - 1
            remote = self._metadata.get_segment(req.target_id)
            if remote is None:
                batch.mark_task_done(task_idx, success=False)
                continue
            self._transport.submit_transfer(batch, task_idx, self._local_buffer, remote)

    def get_transfer_status(self, batch_id: int) -> TransferStatus:
        """Poll batch completion status (maps to getTransferStatus)."""
        return self._batches[batch_id].overall_status

    def wait_for_completion(self, batch_id: int, timeout: float = 10.0) -> TransferStatus:
        """Block until batch completes (convenience helper)."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_transfer_status(batch_id)
            if status in (TransferStatus.COMPLETED, TransferStatus.FAILED):
                return status
            time.sleep(0.01)
        return TransferStatus.FAILED

    def shutdown(self):
        if self._transport:
            self._transport.shutdown()
