"""
nano-mooncake Store KV Connector for vLLM
==========================================
A vLLM KV Connector that uses nano-mooncake's Store layer (MasterService +
StoreClient) for KVCache transfer between Prefill and Decode instances.

Unlike NanoMooncakeConnector (P2P direct TE buffer management), this connector
delegates all buffer allocation and data movement to StoreClient.put()/get(),
matching how real Mooncake's Store mode works.

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
                 │Prefill │    │Decode │
                 │StoreClient  │StoreClient
                 │(:8010) │    │(:8020)│
                 └───┬────┘    └───┬───┘
                     │   TCP TE    │
                     └─────────────┘
                  data transfer via Store

Usage:
  # Start metadata server with MasterService enabled
  python metadata_server.py --port 8090 --enable-master

  # Prefill
  NANO_METADATA_URL=http://127.0.0.1:8090 vllm serve <model> --port 8010 \\
    --kv-transfer-config '{
      "kv_connector": "NanoMooncakeStoreConnector",
      "kv_connector_module_path": "nano_store_connector",
      "kv_role": "kv_both"
    }'

  # Decode
  NANO_METADATA_URL=http://127.0.0.1:8090 vllm serve <model> --port 8020 \\
    --kv-transfer-config '{
      "kv_connector": "NanoMooncakeStoreConnector",
      "kv_connector_module_path": "nano_store_connector",
      "kv_role": "kv_both"
    }'
"""

import os
import socket
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from mooncake_store import StoreClient
from metadata_server import MetadataHttpClient, HttpMasterClient

# Reuse shared helpers from the P2P connector
from nano_connector import (
    ReqMeta,
    NanoMooncakeMetadata,
    _token_hash,
    _align_to_block_size,
)

logger = init_logger(__name__)

NANO_METADATA_URL = os.environ.get("NANO_METADATA_URL", "http://127.0.0.1:8090")
STORE_SEGMENT_SIZE = 512 * 1024 * 1024  # 512MB segment for StoreClient


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─────────────────────────────────────────────────────────────────
# NanoMooncakeStoreConnector — the vLLM KV Connector (Store mode)
# ─────────────────────────────────────────────────────────────────

class NanoMooncakeStoreConnector(KVConnectorBase_V1):
    """
    vLLM KV Connector using nano-mooncake Store layer.

    Key difference from NanoMooncakeConnector:
    - No direct TE buffer management or _allocate_buffer()
    - All data movement via StoreClient.put()/get()
    - MasterService (via HTTP) handles allocation and tracking
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

        # HTTP clients for cross-process coordination
        self._meta_client = MetadataHttpClient(NANO_METADATA_URL)
        self._master_client = HttpMasterClient(NANO_METADATA_URL)

        # Create StoreClient — it manages its own TE internally
        te_port = _get_free_port()
        hostname = socket.gethostname()
        self._segment_name = f"vllm-store-{hostname}-{os.getpid()}"

        self._store = StoreClient(
            master=self._master_client,
            metadata=self._meta_client,
            local_name=self._segment_name,
            host="127.0.0.1",
            port=te_port,
            segment_size=STORE_SEGMENT_SIZE,
        )

        logger.info(
            "NanoMooncakeStoreConnector initialized: segment=%s port=%d "
            "segment_size=%dMB metadata=%s",
            self._segment_name, te_port,
            STORE_SEGMENT_SIZE // 1024 // 1024, NANO_METADATA_URL,
        )

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
        """Extract KV from GPU and store via StoreClient.put()."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, NanoMooncakeMetadata)

        for request in metadata.requests:
            if not request.is_store:
                continue

            # Extract KV from paged GPU memory
            kv_data = self._extract_kv(kv_layer, request.slot_mapping, attn_metadata)
            kv_cpu = kv_data.detach().cpu()
            if kv_cpu.dtype == torch.bfloat16:
                kv_cpu = kv_cpu.to(torch.float16)
            kv_bytes = kv_cpu.numpy().tobytes()

            # Store via Store layer (Master allocates + TE transfers + Master commits)
            token_hash = _token_hash(request.token_ids, request.mm_hashes)
            key = f"{token_hash}:{layer_name}"
            ok = self._store.put(key, kv_bytes)

            if ok:
                logger.info(
                    "Saved KV layer %s (%d bytes) via Store, hash=%s",
                    layer_name, len(kv_bytes), token_hash,
                )
            else:
                logger.error(
                    "Failed to store KV layer %s, hash=%s", layer_name, token_hash,
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
        """Load KV from Store via StoreClient.get()."""
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

                # Retrieve via Store layer
                key = f"{token_hash}:{layer_name}"
                kv_bytes = self._store.get(key)
                if kv_bytes is None:
                    logger.warning(
                        "KV not found in Store: key=%s", key,
                    )
                    continue

                # Reconstruct tensor and inject into GPU
                num_pages, page_size = kv_cache_layer.shape[1], kv_cache_layer.shape[2]
                num_tokens = len(request.slot_mapping)
                wire_dtype = (
                    torch.float16
                    if kv_cache_layer.dtype == torch.bfloat16
                    else kv_cache_layer.dtype
                )
                kv_tensor = torch.frombuffer(bytearray(kv_bytes), dtype=wire_dtype)
                kv_tensor = kv_tensor.reshape(2, num_tokens, -1)
                kv_tensor = kv_tensor.to(dtype=kv_cache_layer.dtype).cuda()

                # Inject into paged KV cache
                flat = kv_cache_layer.reshape(2, num_pages * page_size, -1)
                flat[:, request.slot_mapping, ...] = kv_tensor

                logger.info(
                    "Loaded KV layer %s (%d bytes) via Store, hash=%s",
                    layer_name, len(kv_bytes), token_hash,
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
        """Check if KV cache exists for this request in the Store."""
        token_ids = list(request.prompt_token_ids or [])
        mm_hashes = [f.identifier for f in request.mm_features]
        num_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)
        token_hash = _token_hash(torch.tensor(token_ids)[:num_to_check], mm_hashes)

        # Check via Master query — if any layer exists, the KV was stored
        key = f"{token_hash}:model.layers.0.self_attn"
        obj = self._master_client.query(key)
        if obj is None:
            return 0, False

        logger.info("nano-mooncake Store cache hit! hash=%s", token_hash)
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
                token_hash = _token_hash(
                    torch.tensor(token_ids)[:num_to_check], mm_hashes,
                )
                key = f"{token_hash}:model.layers.0.self_attn"
                obj = self._master_client.query(key)
                if obj is None:
                    meta.add_request(
                        token_ids=token_ids, block_ids=new_req.block_ids[0],
                        block_size=self._block_size, is_store=True, mm_hashes=mm_hashes,
                    )

        self._requests_need_load.clear()
        return meta

    def shutdown(self):
        self._store.shutdown()
        logger.info("NanoMooncakeStoreConnector shut down")
