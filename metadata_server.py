"""
nano-mooncake Metadata Server
==============================
Simplified HTTP metadata server for segment discovery.
Maps to TransferMetadata in Mooncake (which uses etcd/HTTP/Redis).

Run standalone:  python metadata_server.py --port 8080
Or use MetadataHttpClient to connect from TransferEngine.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from transfer_engine import MetadataStore, SegmentDesc


class MetadataHttpHandler(BaseHTTPRequestHandler):
    """HTTP handler for segment registration and discovery."""

    store: MetadataStore  # Set by server setup

    def do_POST(self):
        """POST /segments — register a new segment."""
        if self.path == "/segments":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            desc = SegmentDesc(
                name=body["name"], host=body["host"],
                port=body["port"], size=body["size"],
            )
            self.store.register_segment(desc)
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_GET(self):
        """GET /segments/<name> or GET /segments (list all)."""
        if self.path.startswith("/segments/"):
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
    port: int = 8080, store: Optional[MetadataStore] = None
) -> tuple[HTTPServer, MetadataStore]:
    """Start metadata HTTP server in a background thread. Returns (server, store)."""
    if store is None:
        store = MetadataStore()
    MetadataHttpHandler.store = store
    server = HTTPServer(("0.0.0.0", port), MetadataHttpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, store


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="nano-mooncake metadata server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    store = MetadataStore()
    MetadataHttpHandler.store = store
    server = HTTPServer(("0.0.0.0", args.port), MetadataHttpHandler)
    print(f"Metadata server listening on port {args.port}")
    server.serve_forever()
