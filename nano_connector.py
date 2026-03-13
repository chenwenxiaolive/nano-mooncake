"""
nano-mooncake KV Connector for vLLM
=====================================
A vLLM KV Connector that uses nano-mooncake's Transfer Engine for
KVCache transfer between Prefill and Decode instances.

Based on vLLM's ExampleConnector pattern, but replaces disk I/O with
nano-mooncake's TCP Transfer Engine for network-based KV transfer.

Architecture:
  Prefill (kv_producer)              Decode (kv_consumer)
  ┌──────────────────┐              ┌──────────────────┐
  │ vLLM forward pass │              │ vLLM forward pass │
  │   save_kv_layer() │              │  start_load_kv()  │
  │        │          │              │        │          │
  │   GPU -> CPU      │              │   CPU -> GPU      │
  │        │          │              │        ↑          │
  │   Write to TE buf │              │   Read from TE    │
  └────────┬──────────┘              └────────┬──────────┘
           │      Transfer Engine TCP          │
           └──────────────────────────────────→┘
           │  HTTP Metadata Server (:8090)      │
           └──────────────────────────────────→┘

Usage:
  # Start metadata server first
  python metadata_server.py --port 8090

  # Prefill
  NANO_METADATA_URL=http://127.0.0.1:8090 vllm serve <model> --port 8010 \\
    --kv-transfer-config '{
      "kv_connector": "NanoMooncakeConnector",
      "kv_connector_module_path": "nano_connector",
      "kv_role": "kv_both"
    }'

  # Decode
  NANO_METADATA_URL=http://127.0.0.1:8090 vllm serve <model> --port 8020 \\
    --kv-transfer-config '{
      "kv_connector": "NanoMooncakeConnector",
      "kv_connector_module_path": "nano_connector",
      "kv_role": "kv_both"
    }'
"""

import os
import socket
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Import nano-mooncake Transfer Engine
from transfer_engine import (
    OpCode,
    SegmentDesc,
    TransferEngine,
    TransferRequest,
    TcpTransport,
)

# Import HTTP metadata client (cross-process coordination)
from metadata_server import MetadataHttpClient

logger = init_logger(__name__)

# Metadata server URL (centralized cross-process coordination)
NANO_METADATA_URL = os.environ.get("NANO_METADATA_URL", "http://127.0.0.1:8090")
BUFFER_SIZE = 512 * 1024 * 1024  # 512MB Transfer Engine buffer


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _token_hash(token_ids: torch.Tensor, mm_hashes: list[str]) -> str:
    token_bytes = token_ids.numpy().tobytes()
    if mm_hashes:
        token_bytes += "-".join(mm_hashes).encode()
    return safe_hash(token_bytes, usedforsecurity=False).hexdigest()[:16]


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens - 1) // block_size * block_size


# ─────────────────────────────────────────────────────────────────
# Metadata (scheduler -> worker communication)
# ─────────────────────────────────────────────────────────────────

@dataclass
class ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(
        token_ids: list[int], block_ids: list[int],
        block_size: int, is_store: bool, mm_hashes: list[str],
    ) -> "ReqMeta":
        valid_num_tokens = _align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids_tensor.reshape(-1, 1) * block_size
        ).flatten()[:valid_num_tokens]
        return ReqMeta(token_ids_tensor, slot_mapping, is_store, mm_hashes)


@dataclass
class NanoMooncakeMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self, token_ids: list[int], block_ids: list[int],
        block_size: int, is_store: bool, mm_hashes: list[str],
    ):
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


# ─────────────────────────────────────────────────────────────────
# NanoMooncakeConnector — the vLLM KV Connector
# ─────────────────────────────────────────────────────────────────

class NanoMooncakeConnector(KVConnectorBase_V1):
    """
    vLLM KV Connector using nano-mooncake Transfer Engine.

    Data plane: nano-mooncake TCP Transport (real Mooncake uses RDMA)
    Control plane: HTTP metadata server (real Mooncake uses ZMQ + etcd)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, "Request"] = {}

        # HTTP metadata client for cross-process coordination
        self._meta_client = MetadataHttpClient(NANO_METADATA_URL)

        # Initialize Transfer Engine
        self._te_port = _get_free_port()
        hostname = socket.gethostname()
        self._segment_name = f"vllm-{hostname}-{os.getpid()}"

        self._engine = TransferEngine()
        self._transport = TcpTransport()
        # Pass the HTTP client as metadata store — TransferEngine uses it
        # for segment discovery during transfers
        self._engine.init(
            self._segment_name, self._transport,
            metadata=self._meta_client, listen_port=self._te_port,
        )
        self._buffer = self._engine.register_local_memory(BUFFER_SIZE, "127.0.0.1")
        self._buffer_offset = 0  # Next free offset in buffer

        # Register our segment in the centralized metadata server
        self._meta_client.register_segment(SegmentDesc(
            name=self._segment_name,
            host="127.0.0.1",
            port=self._te_port,
            size=BUFFER_SIZE,
        ))

        logger.info(
            "NanoMooncakeConnector initialized: segment=%s port=%d buffer=%dMB metadata=%s",
            self._segment_name, self._te_port, BUFFER_SIZE // 1024 // 1024,
            NANO_METADATA_URL,
        )

    def _ensure_remote_segment(self, segment_name: str):
        """
        Ensure a remote segment is discoverable via the metadata server.
        Retries with polling since the remote process may not have registered yet.
        """
        max_retries = 10
        for attempt in range(max_retries):
            desc = self._meta_client.get_segment(segment_name)
            if desc is not None:
                return
            if attempt < max_retries - 1:
                time.sleep(0.5)
                logger.debug(
                    "Waiting for remote segment %s (attempt %d/%d)",
                    segment_name, attempt + 1, max_retries,
                )
        logger.warning("Remote segment %s not found after %d retries", segment_name, max_retries)

    def _allocate_buffer(self, size: int) -> int:
        """Allocate space in local Transfer Engine buffer."""
        offset = self._buffer_offset
        if offset + size > BUFFER_SIZE:
            raise RuntimeError("Transfer Engine buffer full")
        self._buffer_offset += size
        return offset

    # ─────────────────────────────────────────────────────────────
    # Worker-side: save KV (Prefill)
    # ─────────────────────────────────────────────────────────────

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Extract KV from GPU and write to Transfer Engine buffer.
        Maps to Mooncake's send_kv_to_decode() via Transfer Engine.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, NanoMooncakeMetadata)

        for request in metadata.requests:
            if not request.is_store:
                continue

            # Extract KV from paged GPU memory
            kv_data = self._extract_kv(kv_layer, request.slot_mapping, attn_metadata)
            kv_bytes = kv_data.detach().cpu().numpy().tobytes()

            # Write to Transfer Engine buffer
            offset = self._allocate_buffer(len(kv_bytes))
            self._buffer[offset:offset + len(kv_bytes)] = kv_bytes

            # Record metadata in centralized server for decode to find
            token_hash = _token_hash(request.token_ids, request.mm_hashes)
            self._meta_client.save_kv_meta(
                token_hash, layer_name, self._segment_name, offset, len(kv_bytes),
            )

            logger.info(
                "Saved KV layer %s (%d bytes) at offset %d, hash=%s",
                layer_name, len(kv_bytes), offset, token_hash,
            )

    def _extract_kv(
        self, kv_layer: torch.Tensor, slot_mapping: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        """Extract KV cache from paged GPU memory."""
        num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
        flat = kv_layer.reshape(2, num_pages * page_size, -1)
        return flat[:, slot_mapping, ...]

    def wait_for_save(self):
        """All saves are synchronous in this implementation."""
        return

    # ─────────────────────────────────────────────────────────────
    # Worker-side: load KV (Decode)
    # ─────────────────────────────────────────────────────────────

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Load KV from remote Prefill via Transfer Engine.
        Maps to Mooncake's receive_kv() -> engine.batch_transfer_sync_write().
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, NanoMooncakeMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for request in metadata.requests:
            if request.is_store:
                continue

            token_hash = _token_hash(request.token_ids, request.mm_hashes)

            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]

                # Look up where the KV data is stored (via HTTP metadata server)
                kv_meta = self._meta_client.load_kv_meta(token_hash, layer_name)
                if kv_meta is None:
                    logger.warning("KV meta not found: hash=%s layer=%s", token_hash, layer_name)
                    continue

                remote_segment = kv_meta["segment"]
                remote_offset = kv_meta["offset"]
                kv_size = kv_meta["size"]

                # Ensure remote segment is registered (with retry/polling)
                self._ensure_remote_segment(remote_segment)

                # Read from remote via Transfer Engine
                local_offset = self._allocate_buffer(kv_size)
                batch_id = self._engine.allocate_batch_id(1)
                req = TransferRequest(
                    opcode=OpCode.READ,
                    source_offset=local_offset,
                    target_id=remote_segment,
                    target_offset=remote_offset,
                    length=kv_size,
                )
                self._engine.submit_transfer(batch_id, [req])
                status = self._engine.wait_for_completion(batch_id, timeout=30.0)

                if status.value != "completed":
                    logger.error("Transfer failed for hash=%s layer=%s", token_hash, layer_name)
                    continue

                # Reconstruct tensor and inject into GPU
                kv_bytes = bytes(self._buffer[local_offset:local_offset + kv_size])
                num_pages, page_size = kv_cache_layer.shape[1], kv_cache_layer.shape[2]
                rest_dims = kv_cache_layer.shape[3:]
                num_tokens = len(request.slot_mapping)
                kv_tensor = torch.frombuffer(bytearray(kv_bytes), dtype=kv_cache_layer.dtype)
                kv_tensor = kv_tensor.reshape(2, num_tokens, *rest_dims).cuda()

                # Inject into paged KV cache
                flat = kv_cache_layer.reshape(2, num_pages * page_size, -1)
                flat[:, request.slot_mapping, ...] = kv_tensor

                logger.info(
                    "Loaded KV layer %s (%d bytes) via Transfer Engine, hash=%s",
                    layer_name, kv_size, token_hash,
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """All loads are synchronous."""
        return

    # ─────────────────────────────────────────────────────────────
    # Scheduler-side: match & allocate
    # ─────────────────────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check if KV cache exists for this request in the Transfer Engine."""
        token_ids = list(request.prompt_token_ids or [])
        mm_hashes = [f.identifier for f in request.mm_features]
        num_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)
        token_hash = _token_hash(torch.tensor(token_ids)[:num_to_check], mm_hashes)

        if not self._meta_client.has_kv(token_hash):
            return 0, False

        logger.info("nano-mooncake cache hit! hash=%s", token_hash)
        return num_to_check - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int,
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata for worker to process saves and loads."""
        meta = NanoMooncakeMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]

            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids, block_ids=new_req.block_ids[0],
                    block_size=self._block_size, is_store=False, mm_hashes=mm_hashes,
                )
            else:
                num_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)
                token_hash = _token_hash(torch.tensor(token_ids)[:num_to_check], mm_hashes)
                if not self._meta_client.has_kv(token_hash):
                    meta.add_request(
                        token_ids=token_ids, block_ids=new_req.block_ids[0],
                        block_size=self._block_size, is_store=True, mm_hashes=mm_hashes,
                    )

        self._requests_need_load.clear()
        return meta

    def shutdown(self):
        self._engine.shutdown()
        logger.info("NanoMooncakeConnector shut down")
