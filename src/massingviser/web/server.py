"""An HTTP face for the engine bridge.

The browser layer needs four things and nothing else: the manifest, the payload buffers, an answer
to "what did I click", and an answer to "what can I see". This serves exactly those, over
``http.server`` from the standard library -- no framework, because four routes do not justify one,
and because this package has to stay dependency-free like every other non-viewer package here.

The division of labour is the point. Picking is a BVH descent, culling is a frustum test, choosing
which elements to send is a set difference over content ids -- all of it happens here, and the
client receives GlobalIds and buffers. A client that wanted to do any of it itself would need the
model, which is the thing this architecture exists to avoid shipping.

Threading: ``http.server`` hands each request to a worker thread while the kernel lives in an
asyncio loop. Every coroutine is submitted to that loop with ``run_coroutine_threadsafe`` and waited
on, so the kernel is only ever touched from its own thread. This is the same boundary the viser
bridge draws, for the same reason.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..geometry import SpatialIndexToken
from ..kernel import Kernel
from ..plugins.engine import SceneExportToken, to_manifest


def _web_root() -> Path:
    """Where the static client lives.

    Two layouts, because both are real. An installed package carries the client at
    ``massingviser/web/client``; a git checkout has it at ``web/`` in the repository root, and that
    is the copy a contributor edits. Preferring the installed copy and falling back to the checkout
    means neither has to know about the other.
    """
    installed = Path(__file__).resolve().parent / "client"
    if (installed / "index.html").is_file():
        return installed
    return Path(__file__).resolve().parents[3] / "web"


#: Resolved once at import. A missing directory is a 404 rather than a crash, so a deployment that
#: only wants the JSON API is not obliged to ship a browser client at all.
WEB_ROOT = _web_root()

#: How long a request will wait on the kernel loop before giving up. A request that hangs forever
#: holds a worker thread, and enough of them stop the server answering at all.
REQUEST_TIMEOUT = 30.0


class SceneBridge:
    """Synchronous access to an asyncio kernel, for use from request threads."""

    __slots__ = ("_kernel", "_loop")

    def __init__(self, kernel: Kernel[Any], loop: asyncio.AbstractEventLoop) -> None:
        self._kernel = kernel
        self._loop = loop

    def _run(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(REQUEST_TIMEOUT)

    def manifest(self) -> dict[str, Any]:
        service = self._kernel.capabilities.get(SceneExportToken)
        if service is None:
            raise LookupError("No scene export service is installed.")
        built = self._run(service.build())
        if not built.ok:
            raise LookupError(built.error.message)
        return to_manifest(built.value)

    def payload(self, payload_id: str) -> bytes:
        service = self._kernel.capabilities.get(SceneExportToken)
        if service is None:
            raise LookupError("No scene export service is installed.")
        found = service.read_payload(payload_id)
        if not found.ok:
            raise LookupError(found.error.message)
        return found.value

    def plan(self, have: Sequence[str]) -> dict[str, Any]:
        service = self._kernel.capabilities.get(SceneExportToken)
        if service is None:
            raise LookupError("No scene export service is installed.")
        planned = self._run(service.plan(tuple(have)))
        if not planned.ok:
            raise LookupError(planned.error.message)
        return {
            "fetch": [
                {"id": ref.id, "path": ref.path, "byteLength": ref.byte_length, "lod": ref.lod}
                for ref in planned.value.fetch
            ],
            "fetchBytes": planned.value.fetch_bytes,
            "reuse": list(planned.value.reuse),
            "stale": list(planned.value.stale),
        }

    def pick(self, origin: Sequence[float], direction: Sequence[float], limit: int = 8) -> Any:
        index = self._kernel.capabilities.get(SpatialIndexToken)
        if index is None:
            return []
        hits = index.pick(tuple(origin), tuple(direction), limit=limit)
        return [{"globalId": hit.global_id, "distance": hit.distance} for hit in hits]

    def cull(self, view_projection: Sequence[float]) -> list[str]:
        index = self._kernel.capabilities.get(SpatialIndexToken)
        if index is None:
            return []
        return list(index.cull(tuple(view_projection)))


def create_handler(bridge: SceneBridge, *, web_root: Path = WEB_ROOT) -> type:
    """Build a request handler bound to one bridge."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "MassingViser"
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            """Silence the default stderr access log; the kernel has its own logger."""

        # -- plumbing ------------------------------------------------------------------------

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Payloads are content-addressed, so their bytes can never change under a given id.
            # Anything else must not be cached, or an edit would not show up.
            if self.path.startswith("/api/payload/"):
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            else:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError:
                return None

        # -- routes --------------------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/manifest":
                    self._json(200, bridge.manifest())
                    return
                if path.startswith("/api/payload/"):
                    payload_id = path[len("/api/payload/") :].removesuffix(".bin")
                    self._send(200, bridge.payload(payload_id), "application/octet-stream")
                    return
                self._static(path)
            except LookupError as thrown:
                self._error(404, str(thrown))
            except Exception as thrown:  # noqa: BLE001 -- a request must not kill the server
                self._error(500, str(thrown))

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            body = self._body()
            if body is None:
                self._error(400, "Request body is not valid JSON.")
                return
            try:
                if path == "/api/plan":
                    have = body.get("have", [])
                    if isinstance(have, str):
                        # A bare string iterates as characters and would report everything stale.
                        self._error(400, '"have" is a list of payload ids, not a string.')
                        return
                    self._json(200, bridge.plan(have))
                    return
                if path == "/api/pick":
                    origin, direction = body.get("origin"), body.get("direction")
                    if not _is_vector(origin) or not _is_vector(direction):
                        self._error(
                            400, "pick needs an origin and a direction, each three numbers."
                        )
                        return
                    self._json(200, {"hits": bridge.pick(origin, direction, body.get("limit", 8))})
                    return
                if path == "/api/cull":
                    matrix = body.get("viewProjection")
                    if not isinstance(matrix, list) or len(matrix) != 16:
                        self._error(400, "cull needs a 16-element column-major matrix.")
                        return
                    self._json(200, {"visible": bridge.cull(matrix)})
                    return
                self._error(404, f"No route for {path}.")
            except LookupError as thrown:
                self._error(404, str(thrown))
            except Exception as thrown:  # noqa: BLE001
                self._error(500, str(thrown))

        def _static(self, path: str) -> None:
            relative = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (web_root / relative).resolve()
            try:
                # Refuse anything that escapes the served directory. `..` in a URL is the oldest
                # trick there is, and `resolve()` is what makes the containment check meaningful.
                target.relative_to(web_root.resolve())
            except ValueError:
                self._error(403, "Path escapes the served directory.")
                return
            if not target.is_file():
                self._error(404, f"No such file: {relative}")
                return
            kind, _ = mimetypes.guess_type(str(target))
            self._send(200, target.read_bytes(), kind or "application/octet-stream")

    return Handler


def _is_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def serve(
    kernel: Kernel[Any],
    loop: asyncio.AbstractEventLoop,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    web_root: Path = WEB_ROOT,
) -> ThreadingHTTPServer:
    """Start the server on a background thread and return it, already listening."""
    handler = create_handler(SceneBridge(kernel, loop), web_root=web_root)
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="massingviser-web", daemon=True)
    thread.start()
    return server
