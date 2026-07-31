"""The HTTP face of the engine bridge.

These drive the server the way a browser does -- real sockets, real JSON, real binary bodies --
because the interesting failures are all at that boundary: a payload served with the wrong content
type, a matrix accepted at the wrong length, a path that escapes the served directory.

The client-side counterpart lives in `web/test/mvmesh.test.mjs` and runs under Node against
fixtures this same encoder produced.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from massingviser import build_kernel
from massingviser.geometry import MESH_ENCODING, decode_mesh_batch
from massingviser.plugins.massing import MASSING_COMMANDS
from massingviser.web import SceneBridge, create_handler, serve


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _Server:
    """A kernel on its own loop thread, with the web server in front of it."""

    def __init__(self, web_root: Path) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()
        self.thread.start()
        self.ready.wait(5)

        self.kernel = build_kernel()
        self._submit(self.kernel.start())
        self._seed()

        self.port = _free_port()
        self.http = serve(self.kernel, self.loop, port=self.port, web_root=web_root)
        self.base = f"http://127.0.0.1:{self.port}"

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self.ready.set)
        self.loop.run_forever()

    def _submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(30)

    def _seed(self) -> None:
        sketched = self._submit(
            self.kernel.commands.execute(
                MASSING_COMMANDS.sketch_profile,
                {"points": [(0, 0, 0), (20, 0, 0), (20, 12, 0), (0, 12, 0)], "name": "Block"},
            )
        )
        self._submit(
            self.kernel.commands.execute(
                MASSING_COMMANDS.create_mass,
                {
                    "name": "Block",
                    "profile_id": sketched.value,
                    "story_count": 5,
                    "story_height": 3.5,
                },
            )
        )

    def get(self, path: str):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=30) as response:
            return response.status, response.read(), dict(response.headers)

    def post(self, path: str, body: dict):
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())

    def close(self) -> None:
        self.http.shutdown()
        self._submit(self.kernel.stop())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    (root / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (root / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "deep.txt").write_text("ok", encoding="utf-8")
    running = _Server(root)
    yield running
    running.close()


# ---------------------------------------------------------------------------------------------
# The manifest and the buffers
# ---------------------------------------------------------------------------------------------


def test_the_manifest_describes_the_scene(server):
    status, body, headers = server.get("/api/manifest")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    manifest = json.loads(body)
    assert len(manifest["nodes"]) == 5
    assert all(node["geometry"] for node in manifest["nodes"])


def test_a_payload_is_served_as_bytes_a_client_can_decode(server):
    manifest = json.loads(server.get("/api/manifest")[1])
    payload_id = manifest["payloads"][0]["id"]
    status, body, headers = server.get(f"/api/payload/{payload_id}.bin")
    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    # The decoder is the contract; if this reads, a browser reads it too.
    meshes = decode_mesh_batch(body)
    assert len(meshes) == manifest["payloads"][0]["meshCount"]
    assert meshes[0].normals is not None


def test_a_payload_is_cached_forever_because_its_id_is_its_content(server):
    """The id is a hash, so the bytes behind it can never change. Anything else must not cache."""
    manifest = json.loads(server.get("/api/manifest")[1])
    _, _, payload_headers = server.get(f"/api/payload/{manifest['payloads'][0]['id']}.bin")
    assert "immutable" in payload_headers["Cache-Control"]
    _, _, manifest_headers = server.get("/api/manifest")
    assert manifest_headers["Cache-Control"] == "no-store"


def test_the_extension_is_optional(server):
    manifest = json.loads(server.get("/api/manifest")[1])
    payload_id = manifest["payloads"][0]["id"]
    assert (
        server.get(f"/api/payload/{payload_id}")[1]
        == server.get(f"/api/payload/{payload_id}.bin")[1]
    )


def test_an_unknown_payload_is_a_404_that_names_it(server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.get("/api/payload/" + "0" * 32)
    assert raised.value.code == 404
    assert "0000" in json.loads(raised.value.read())["error"]


def test_the_declared_encoding_is_the_one_the_client_reads(server):
    manifest = json.loads(server.get("/api/manifest")[1])
    assert all(
        payload["encoding"] == MESH_ENCODING
        for payload in manifest["payloads"]
        if payload["role"] == "geometry"
    )


# ---------------------------------------------------------------------------------------------
# The transfer plan
# ---------------------------------------------------------------------------------------------


def test_a_cold_client_is_told_to_fetch_everything(server):
    status, plan = server.post("/api/plan", {"have": []})
    assert status == 200
    assert len(plan["fetch"]) > 0
    assert plan["fetchBytes"] > 0
    assert plan["reuse"] == [] and plan["stale"] == []


def test_a_warm_client_transfers_nothing(server):
    manifest = json.loads(server.get("/api/manifest")[1])
    held = [payload["id"] for payload in manifest["payloads"]]
    _, plan = server.post("/api/plan", {"have": held})
    assert plan["fetch"] == []
    assert plan["fetchBytes"] == 0
    assert sorted(plan["reuse"]) == sorted(held)


def test_a_client_holding_something_the_scene_dropped_is_told_it_is_stale(server):
    _, plan = server.post("/api/plan", {"have": ["c" * 32]})
    assert plan["stale"] == ["c" * 32]


def test_a_have_list_given_as_a_string_is_refused(server):
    """It would iterate as characters and report every payload as stale."""
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.post("/api/plan", {"have": "abc"})
    assert raised.value.code == 400
    assert "not a string" in json.loads(raised.value.read())["error"]


# ---------------------------------------------------------------------------------------------
# Picking and culling
# ---------------------------------------------------------------------------------------------


def test_a_ray_comes_back_as_global_ids_nearest_first(server):
    status, result = server.post(
        "/api/pick", {"origin": [10.0, 6.0, 500.0], "direction": [0.0, 0.0, -1.0]}
    )
    assert status == 200
    hits = result["hits"]
    assert hits, "a ray straight down the middle of the block should hit it"
    assert [hit["distance"] for hit in hits] == sorted(hit["distance"] for hit in hits)
    # The identity everything else keys on, not a render handle.
    assert hits[0]["globalId"].startswith("mass-")


def test_a_ray_into_empty_space_hits_nothing(server):
    _, result = server.post(
        "/api/pick", {"origin": [900.0, 900.0, 500.0], "direction": [0.0, 0.0, -1.0]}
    )
    assert result["hits"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"origin": [0, 0, 0]},
        {"origin": [0, 0], "direction": [0, 0, -1]},
        {"origin": "nope", "direction": [0, 0, -1]},
        {"origin": [0, 0, 0], "direction": [True, False, True]},
    ],
)
def test_a_malformed_ray_is_a_400_not_a_crash(server, body):
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.post("/api/pick", body)
    assert raised.value.code == 400


def test_culling_returns_the_ids_a_camera_can_see(server):
    # An orthographic box wide enough to contain the whole block.
    matrix = [
        1 / 60,
        0,
        0,
        0,
        0,
        1 / 60,
        0,
        0,
        0,
        0,
        -1 / 60,
        0,
        0,
        0,
        0,
        1,
    ]
    _, result = server.post("/api/cull", {"viewProjection": matrix})
    assert len(result["visible"]) == 5


def test_a_matrix_of_the_wrong_length_is_refused(server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.post("/api/cull", {"viewProjection": [1, 0, 0, 1]})
    assert raised.value.code == 400
    assert "16-element" in json.loads(raised.value.read())["error"]


def test_a_body_that_is_not_json_is_a_400(server):
    request = urllib.request.Request(
        f"{server.base}/api/plan",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=30)
    assert raised.value.code == 400


def test_an_unknown_route_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.post("/api/nonsense", {})
    assert raised.value.code == 404


# ---------------------------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------------------------


def test_the_root_serves_the_client(server):
    status, body, headers = server.get("/")
    assert status == 200
    assert b"<!doctype html>" in body
    assert headers["Content-Type"].startswith("text/html")


def test_a_module_is_served_with_a_type_a_browser_will_execute(server):
    """A JavaScript module served as text/plain is refused by every browser's module loader."""
    _, _, headers = server.get("/app.js")
    assert "javascript" in headers["Content-Type"]


def test_a_nested_file_is_served(server):
    assert server.get("/nested/deep.txt")[1] == b"ok"


@pytest.mark.parametrize("path", ["/../pyproject.toml", "/nested/../../pyproject.toml"])
def test_a_path_that_escapes_the_served_directory_is_refused(server, path):
    """The oldest trick there is, and `resolve()` is what makes the check mean anything."""
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.get(path)
    assert raised.value.code in (403, 404)


def test_a_missing_file_is_a_404_that_names_it(server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        server.get("/nope.js")
    assert raised.value.code == 404
    assert "nope.js" in json.loads(raised.value.read())["error"]


# ---------------------------------------------------------------------------------------------
# The bridge on its own
# ---------------------------------------------------------------------------------------------


def test_the_bridge_reports_a_missing_service_rather_than_raising_something_opaque():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        kernel = build_kernel(plugins=())
        asyncio.run_coroutine_threadsafe(kernel.start(), loop).result(10)
        bridge = SceneBridge(kernel, loop)
        with pytest.raises(LookupError, match="No scene export service"):
            bridge.manifest()
        # Picking with no spatial index is empty rather than an error: a client asking what is
        # under a ray in an empty project should get "nothing", not a failure.
        assert bridge.pick([0, 0, 0], [0, 0, -1]) == []
        assert bridge.cull([0] * 16) == []
        asyncio.run_coroutine_threadsafe(kernel.stop(), loop).result(10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


def test_a_handler_can_be_built_without_a_kernel_running():
    """The handler is a plain class over a bridge, so it is constructible in isolation."""
    loop = asyncio.new_event_loop()
    handler = create_handler(SceneBridge(build_kernel(), loop), web_root=Path("."))
    assert callable(handler)
    loop.close()
