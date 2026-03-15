"""
nano-mooncake Metadata Server
==============================
Simplified HTTP metadata server for segment discovery and KV location metadata.
Maps to TransferMetadata in Mooncake (which uses etcd/HTTP/Redis).

Run standalone:  python metadata_server.py --port 8090
Or use MetadataHttpClient to connect from TransferEngine / NanoMooncakeConnector.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from transfer_engine import MetadataStore, SegmentDesc
from mooncake_store import MasterService, Replica, ReplicaStatus, ObjectMetadata


class MetadataHttpHandler(BaseHTTPRequestHandler):
    """HTTP handler for segment registration, discovery, and KV metadata."""

    store: MetadataStore  # Set by server setup
    master: Optional[MasterService]  # Set by server setup (None if not enabled)
    kv_store: dict  # {token_hash: {layer_name: {segment, offset, size}}}
    kv_lock: threading.Lock

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path == "/segments":
            desc = SegmentDesc(
                name=body["name"], host=body["host"],
                port=body["port"], size=body["size"],
            )
            self.store.register_segment(desc)
            self._respond(200, {"status": "ok"})
        elif self.path == "/kv":
            token_hash = body["token_hash"]
            layer_name = body["layer_name"]
            with self.kv_lock:
                if token_hash not in self.kv_store:
                    self.kv_store[token_hash] = {}
                self.kv_store[token_hash][layer_name] = {
                    "segment": body["segment"],
                    "offset": body["offset"],
                    "size": body["size"],
                }
            self._respond(200, {"status": "ok"})
        elif self.path == "/master/register_segment" and self.master is not None:
            desc = SegmentDesc(
                name=body["name"], host=body["host"],
                port=body["port"], size=body["size"],
            )
            self.master.register_segment(desc)
            self._respond(200, {"status": "ok"})
        elif self.path == "/master/put_start" and self.master is not None:
            replica = self.master.put_start(body["client_name"], body["key"], body["size"])
            if replica is None:
                self._respond(200, {"replica": None})
            else:
                self._respond(200, {"replica": {
                    "segment_name": replica.segment_name,
                    "offset": replica.offset,
                    "size": replica.size,
                    "status": replica.status.value,
                }})
        elif self.path == "/master/put_end" and self.master is not None:
            ok = self.master.put_end(body["key"])
            self._respond(200, {"ok": ok})
        else:
            self._respond(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path.startswith("/segments/"):
            name = self.path[len("/segments/"):]
            seg = self.store.get_segment(name)
            if seg:
                self._respond(200, {
                    "name": seg.name, "host": seg.host,
                    "port": seg.port, "size": seg.size,
                })
            else:
                self._respond(404, {"error": "segment not found"})
        elif self.path == "/segments":
            segs = self.store.all_segments()
            self._respond(200, {
                name: {"host": s.host, "port": s.port, "size": s.size}
                for name, s in segs.items()
            })
        elif self.path.startswith("/kv/"):
            parts = self.path[len("/kv/"):].split("/", 1)
            token_hash = parts[0]
            with self.kv_lock:
                if token_hash not in self.kv_store:
                    self._respond(404, {"error": "kv not found"})
                    return
                if len(parts) == 2:
                    # GET /kv/<hash>/<layer> — lookup specific layer
                    layer_name = parts[1]
                    layer_meta = self.kv_store[token_hash].get(layer_name)
                    if layer_meta:
                        self._respond(200, layer_meta)
                    else:
                        self._respond(404, {"error": "layer not found"})
                else:
                    # GET /kv/<hash> — check if KV exists
                    self._respond(200, {"exists": True})
        elif self.path.startswith("/master/query/") and self.master is not None:
            key = self.path[len("/master/query/"):]
            obj = self.master.query(key)
            if obj is None:
                self._respond(404, {"error": "not found"})
            else:
                self._respond(200, {
                    "key": obj.key,
                    "size": obj.size,
                    "created_by": obj.created_by,
                    "lease_timeout": obj.lease_timeout,
                    "replicas": [{
                        "segment_name": r.segment_name,
                        "offset": r.offset,
                        "size": r.size,
                        "status": r.status.value,
                    } for r in obj.replicas],
                })
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def start_metadata_server(
    port: int = 8090, store: Optional[MetadataStore] = None,
    master: Optional[MasterService] = None,
) -> tuple[HTTPServer, MetadataStore]:
    """Start metadata HTTP server in a background thread. Returns (server, store)."""
    if store is None:
        store = MetadataStore()
    MetadataHttpHandler.store = store
    MetadataHttpHandler.master = master
    MetadataHttpHandler.kv_store = {}
    MetadataHttpHandler.kv_lock = threading.Lock()
    server = HTTPServer(("0.0.0.0", port), MetadataHttpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, store


# ─────────────────────────────────────────────────────────────────
# HTTP Client (drop-in replacement for MetadataStore + KVMetadataStore)
# ─────────────────────────────────────────────────────────────────

class MetadataHttpClient:
    """
    HTTP client for the metadata server.
    Drop-in replacement for MetadataStore + file-based KVMetadataStore.
    Enables cross-process segment discovery and KV location tracking.
    """

    def __init__(self, server_url: str):
        self._url = server_url.rstrip("/")

    # ── Segment API (MetadataStore interface) ──

    def register_segment(self, desc: SegmentDesc):
        self._post("/segments", {
            "name": desc.name, "host": desc.host,
            "port": desc.port, "size": desc.size,
        })

    def get_segment(self, name: str) -> Optional[SegmentDesc]:
        data = self._get(f"/segments/{name}")
        if data is None:
            return None
        return SegmentDesc(
            name=data["name"], host=data["host"],
            port=data["port"], size=data["size"],
        )

    def all_segments(self) -> dict[str, SegmentDesc]:
        data = self._get("/segments")
        if data is None:
            return {}
        return {
            name: SegmentDesc(name=name, host=s["host"], port=s["port"], size=s["size"])
            for name, s in data.items()
        }

    # ── KV Metadata API (replaces file-based KVMetadataStore) ──

    def save_kv_meta(
        self, token_hash: str, layer_name: str,
        segment_name: str, offset: int, size: int,
    ):
        self._post("/kv", {
            "token_hash": token_hash,
            "layer_name": layer_name,
            "segment": segment_name,
            "offset": offset,
            "size": size,
        })

    def load_kv_meta(self, token_hash: str, layer_name: str) -> Optional[dict]:
        return self._get(f"/kv/{token_hash}/{layer_name}")

    def has_kv(self, token_hash: str) -> bool:
        data = self._get(f"/kv/{token_hash}")
        return data is not None and data.get("exists", False)

    # ── HTTP helpers ──

    def _post(self, path: str, data: dict):
        body = json.dumps(data).encode()
        req = Request(
            f"{self._url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> Optional[dict]:
        req = Request(f"{self._url}{path}", method="GET")
        try:
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except URLError as e:
            if hasattr(e, "code") and e.code == 404:
                return None
            raise


# ─────────────────────────────────────────────────────────────────
# HTTP Master Client (drop-in replacement for MasterService)
# ─────────────────────────────────────────────────────────────────

class HttpMasterClient:
    """
    HTTP client for MasterService endpoints on the metadata server.
    Duck-typed to match MasterService interface so StoreClient works
    with either a local MasterService or this remote client.
    """

    def __init__(self, server_url: str):
        self._url = server_url.rstrip("/")

    def register_segment(self, desc: SegmentDesc):
        self._post("/master/register_segment", {
            "name": desc.name, "host": desc.host,
            "port": desc.port, "size": desc.size,
        })

    def put_start(self, client_name: str, key: str, size: int) -> Optional[Replica]:
        data = self._post("/master/put_start", {
            "client_name": client_name, "key": key, "size": size,
        })
        r = data.get("replica")
        if r is None:
            return None
        return Replica(
            segment_name=r["segment_name"], offset=r["offset"],
            size=r["size"], status=ReplicaStatus(r["status"]),
        )

    def put_end(self, key: str) -> bool:
        data = self._post("/master/put_end", {"key": key})
        return data.get("ok", False)

    def query(self, key: str) -> Optional[ObjectMetadata]:
        data = self._get(f"/master/query/{key}")
        if data is None:
            return None
        replicas = [
            Replica(
                segment_name=r["segment_name"], offset=r["offset"],
                size=r["size"], status=ReplicaStatus(r["status"]),
            )
            for r in data.get("replicas", [])
        ]
        return ObjectMetadata(
            key=data["key"], size=data["size"],
            replicas=replicas, lease_timeout=data.get("lease_timeout", 0.0),
            created_by=data.get("created_by", ""),
        )

    def _post(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req = Request(
            f"{self._url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> Optional[dict]:
        req = Request(f"{self._url}{path}", method="GET")
        try:
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except URLError as e:
            if hasattr(e, "code") and e.code == 404:
                return None
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="nano-mooncake metadata server")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--enable-master", action="store_true",
                        help="Enable MasterService for Store-mode operation")
    args = parser.parse_args()

    store = MetadataStore()
    master = MasterService() if args.enable_master else None
    MetadataHttpHandler.store = store
    MetadataHttpHandler.master = master
    MetadataHttpHandler.kv_store = {}
    MetadataHttpHandler.kv_lock = threading.Lock()
    server = HTTPServer(("0.0.0.0", args.port), MetadataHttpHandler)
    mode = "with MasterService" if master else "metadata only"
    print(f"Metadata server listening on port {args.port} ({mode})")
    server.serve_forever()
