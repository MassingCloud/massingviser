"""The viser shell.

These tests drive the shell exactly as a browser would -- through the bridge -- and assert on what
the server would send. They do not need a browser, because the interesting failures (a stale storey
left in the scene, a GUI callback that never reaches the kernel, a race between the websocket
thread and the state store) are all visible server-side.
"""

from __future__ import annotations

import socket

import pytest

viser = pytest.importorskip("viser")

from massingviser import build_kernel
from massingviser.demo import seed
from massingviser.plugins.massing import MASSING_COMMANDS, MassingToken
from massingviser.viewer import KernelBridge, MassingViserApp, hex_to_rgb, mesh_arrays
from massingviser.viewer.scene import DEFAULT_COLOR


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def app():
    kernel = build_kernel()
    bridge = KernelBridge(kernel)
    bridge.start()
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    instance = MassingViserApp(bridge=bridge, server=server, title="test").build()
    try:
        yield instance
    finally:
        bridge.close()
        server.stop()


def _scene_names(app) -> set[str]:
    return set(app.server.scene._handle_from_node_name)


def _storey_meshes(app, mass_id: str) -> list[str]:
    return [
        name
        for name in _scene_names(app)
        if name.startswith(f"/masses/{mass_id}/story_")
    ]


# ---------------------------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#4C78A8", (76, 120, 168)),
        ("4C78A8", (76, 120, 168)),
        ("#abc", (170, 187, 204)),
        (None, DEFAULT_COLOR),
        ("not-a-colour", DEFAULT_COLOR),
        ("#GGGGGG", DEFAULT_COLOR),
    ],
)
def test_colour_parsing_never_raises(value, expected):
    """A malformed colour in a record must not blank the viewport."""
    assert hex_to_rgb(value) == expected


def test_mesh_arrays_have_the_shapes_viser_wants():
    from massingviser.plugins.massing.tessellate import extrude

    mesh = extrude([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], [], 0.0, 3.0)
    vertices, faces = mesh_arrays(mesh)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert faces.max() < len(vertices)


# ---------------------------------------------------------------------------------------------
# The shell
# ---------------------------------------------------------------------------------------------


def test_the_shell_builds_the_static_scene(app):
    names = _scene_names(app)
    assert "/ground" in names


def test_a_mass_is_drawn_as_one_solid_per_storey(app):
    profile = app.bridge.execute(
        MASSING_COMMANDS.sketch_profile,
        {"points": [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)]},
    ).value
    mass = app.bridge.execute(
        MASSING_COMMANDS.create_mass,
        {"name": "T", "profile_id": profile, "story_count": 7, "story_height": 3.0},
    ).value
    app.refresh()

    assert len(_storey_meshes(app, mass.id)) == 7
    assert f"/masses/{mass.id}/label" in _scene_names(app)


def test_editing_storeys_rebuilds_the_geometry_and_leaves_nothing_stale(app):
    profile = app.bridge.execute(
        MASSING_COMMANDS.sketch_profile,
        {"points": [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)]},
    ).value
    mass = app.bridge.execute(
        MASSING_COMMANDS.create_mass,
        {"name": "T", "profile_id": profile, "story_count": 10, "story_height": 3.0},
    ).value
    app.refresh()
    assert len(_storey_meshes(app, mass.id)) == 10

    app.bridge.execute(MASSING_COMMANDS.set_story_count, {"id": mass.id, "count": 4})
    app.refresh()
    # The stale storeys 5-10 must be gone, not merely hidden behind the new ones.
    assert len(_storey_meshes(app, mass.id)) == 4

    app.bridge.undo()
    app.refresh()
    assert len(_storey_meshes(app, mass.id)) == 10


def test_deleting_a_mass_removes_it_from_the_scene(app):
    profile = app.bridge.execute(
        MASSING_COMMANDS.sketch_profile,
        {"points": [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)]},
    ).value
    mass = app.bridge.execute(
        MASSING_COMMANDS.create_mass,
        {"name": "T", "profile_id": profile, "story_count": 3},
    ).value
    app.refresh()
    assert _storey_meshes(app, mass.id)

    app.bridge.execute(MASSING_COMMANDS.remove_mass, {"id": mass.id})
    app.refresh()
    assert _storey_meshes(app, mass.id) == []
    assert f"/masses/{mass.id}/label" not in _scene_names(app)


def test_the_demo_scheme_renders_every_block_including_the_courtyard(app):
    seed(app.bridge)
    app.refresh()

    masses = app.bridge.read(lambda: app.bridge.kernel.capabilities.get(MassingToken).list())
    assert len(masses) == 3
    total = sum(len(_storey_meshes(app, mass.id)) for mass in masses)
    assert total == 6 + 9 + 28  # every storey of every block
    assert "/site/boundary" in _scene_names(app)


def test_a_failed_command_is_surfaced_not_swallowed(app):
    app._report(
        app.bridge.execute(MASSING_COMMANDS.sketch_profile, {"points": [(0, 0, 0)]}), ""
    )
    assert "⚠️" in app._status.content


def test_the_bridge_converts_a_dispatch_failure_into_a_result(app):
    app.bridge.close()
    result = app.bridge.execute(MASSING_COMMANDS.compute_metrics, {"id": "nope"})
    assert not result.ok  # a dead loop must not raise into the websocket thread


def test_panels_are_built_from_the_ui_registry_not_hard_coded(app):
    """Installing a capability plugin is what puts a panel on screen."""
    points = app.bridge.read(lambda: app.bridge.kernel.ui.by_point("panel"))
    ids = {contribution.id for contribution in points}
    # Every capability family that ships a panel is on screen, and each one is attributed to the
    # plugin that contributed it -- which is what makes the shell's layout a consequence of what
    # is installed rather than a list in this file.
    assert {"massing.panel", "markup.panel", "estimating.panel"} <= ids
    active = {
        record.manifest.id
        for record in app.bridge.read(app.bridge.kernel.plugins.list)
        if record.status == "active"
    }
    assert all(contribution.plugin_id in active for contribution in points)
    assert len(ids) == len(points)  # no duplicate panel ids
    # One rerender hook per panel that has live content, plus diagnostics.
    assert len(app._rerender) >= 4


def test_a_kernel_without_a_capability_still_builds_a_shell():
    """A shell over a kernel with only massing must not fail on the missing cost panel."""
    kernel = build_kernel(plugins=("massing",))
    bridge = KernelBridge(kernel)
    bridge.start()
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    try:
        instance = MassingViserApp(bridge=bridge, server=server).build()
        instance.refresh()
        panels = bridge.read(lambda: kernel.ui.by_point("panel"))
        assert {contribution.id for contribution in panels} == {"massing.panel"}
    finally:
        bridge.close()
        server.stop()
