from __future__ import annotations

import math

import pytest

from massingviser.plugins.massing import (
    MASSING_COMMANDS,
    MassingToken,
    MetricsToken,
    ProfileToken,
    StoryToken,
    centroid,
    compute_mass_metrics,
    is_simple_polygon,
    massing_plugin,
    net_area,
    normalise_winding,
    polygon_area,
    resolve_story_heights,
    signed_area,
    validate_profile,
)
from massingviser.plugins.massing.tessellate import extrude, extrude_stories, triangulate

SQUARE = [(0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)]
COURTYARD = [(10.0, 8.0), (20.0, 8.0), (20.0, 14.0), (10.0, 14.0)]
L_SHAPE = [(0.0, 0.0), (20.0, 0.0), (20.0, 8.0), (8.0, 8.0), (8.0, 20.0), (0.0, 20.0)]


# ---------------------------------------------------------------------------------------------
# Planar geometry
# ---------------------------------------------------------------------------------------------


def test_signed_area_carries_winding():
    assert signed_area(SQUARE) > 0
    assert signed_area(list(reversed(SQUARE))) < 0
    # Normalising a clockwise ring flips it back to counter-clockwise...
    assert signed_area(normalise_winding(list(reversed(SQUARE)))) > 0
    # ...and leaves an already-counter-clockwise ring untouched.
    assert normalise_winding(SQUARE) == SQUARE


def test_net_area_subtracts_courtyards():
    assert polygon_area(SQUARE) == 600.0
    assert net_area(SQUARE, [COURTYARD]) == 540.0


def test_centroid_is_the_area_centroid_not_the_vertex_average():
    # Extra vertices along one edge skew the vertex average but not the area centroid.
    skewed = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)]
    cx, cy = centroid(skewed)
    assert cx == pytest.approx(15.0)
    assert cy == pytest.approx(10.0)
    vertex_average_y = sum(p[1] for p in skewed) / len(skewed)
    assert vertex_average_y != pytest.approx(cy)


def test_a_bow_tie_is_rejected_rather_than_silently_measured():
    bow_tie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    assert not is_simple_polygon(bow_tie)
    codes = {issue.code for issue in validate_profile(bow_tie)}
    assert "self-intersecting" in codes


def test_a_hole_outside_the_outline_is_reported():
    outside = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0)]
    codes = {issue.code for issue in validate_profile(SQUARE, [outside])}
    assert "hole-outside-outer" in codes


def test_resolve_story_heights_pads_with_the_last_known_height():
    # A heights list that disagrees with the count happens constantly mid-edit and must not fail.
    assert resolve_story_heights(5, [4.5, 3.0]) == [4.5, 3.0, 3.0, 3.0, 3.0]
    assert resolve_story_heights(3, None, 3.2) == [3.2, 3.2, 3.2]


def test_metrics_are_computed_per_story_so_setbacks_are_honoured():
    tower = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]  # 100 m2
    setback = [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)]  # 50 m2

    result = compute_mass_metrics(
        outer=tower, story_heights=[3.0, 3.0, 3.0], story_outlines={2: setback}
    )
    assert result.gross_floor_area == pytest.approx(250.0)  # not 300
    assert result.volume == pytest.approx(750.0)


def test_excluded_stories_leave_gfa_but_stay_in_the_volume():
    result = compute_mass_metrics(
        outer=SQUARE, story_heights=[3.0, 3.0, 3.0], excluded_stories=[2]
    )
    assert result.gross_floor_area == pytest.approx(1200.0)  # plant level excluded
    assert result.volume == pytest.approx(5400.0)  # but it is still a volume


# ---------------------------------------------------------------------------------------------
# Tessellation
# ---------------------------------------------------------------------------------------------


def _triangulated_area(ring, triangles) -> float:
    total = 0.0
    for a, b, c in triangles:
        (x1, y1), (x2, y2), (x3, y3) = ring[a], ring[b], ring[c]
        total += abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
    return total


@pytest.mark.parametrize(
    ("outer", "holes"),
    [
        (SQUARE, []),
        (SQUARE, [COURTYARD]),
        (SQUARE, [COURTYARD, [(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0)]]),
        (L_SHAPE, []),
        (list(reversed(SQUARE)), [COURTYARD]),  # clockwise input must still work
    ],
)
def test_triangulation_conserves_area(outer, holes):
    """The property that matters: triangles cover the polygon exactly, holes included."""
    ring, triangles = triangulate(outer, holes)
    assert triangles, "triangulation produced nothing"
    assert _triangulated_area(ring, triangles) == pytest.approx(net_area(outer, holes), rel=1e-9)


def test_extrusion_is_a_closed_solid():
    mesh = extrude(SQUARE, [COURTYARD], base_elevation=0.0, height=4.0)
    assert not mesh.is_empty
    # Every edge is shared by exactly two triangles in a closed manifold.
    edges: dict[tuple[int, int], int] = {}
    positions = {v: i for i, v in enumerate(mesh.vertices)}
    for a, b, c in mesh.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((positions[mesh.vertices[u]], positions[mesh.vertices[v]])))
            edges[key] = edges.get(key, 0) + 1
    assert all(count % 2 == 0 for count in edges.values())


def test_degenerate_input_yields_an_empty_mesh_rather_than_raising():
    assert extrude([(0.0, 0.0), (1.0, 0.0)], [], 0.0, 3.0).is_empty
    assert extrude(SQUARE, [], 0.0, 0.0).is_empty


def test_stories_are_separate_solids_stacked_by_elevation():
    stories = extrude_stories(SQUARE, [], [4.0, 3.5, 3.5], base_elevation=2.0)
    assert [s.elevation for s in stories] == [2.0, 6.0, 9.5]
    assert all(not s.mesh.is_empty for s in stories)


# ---------------------------------------------------------------------------------------------
# The plugin
# ---------------------------------------------------------------------------------------------


async def test_the_plugin_activates_and_provides_its_capabilities(harness):
    await harness.load(massing_plugin)
    for token in (ProfileToken, MassingToken, StoryToken, MetricsToken):
        assert harness.capability(token) is not None


async def test_a_self_intersecting_sketch_is_refused_by_the_command(harness):
    await harness.load(massing_plugin)
    result = await harness.execute(
        MASSING_COMMANDS.sketch_profile,
        {"points": [(0, 0, 0), (10, 10, 0), (10, 0, 0), (0, 10, 0)]},
    )
    assert not result.ok
    assert "crosses itself" in result.error.message


async def test_story_edits_preserve_per_story_annotations(harness):
    await harness.load(massing_plugin)
    profile = (
        await harness.execute(MASSING_COMMANDS.sketch_profile, {"points": [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]})
    ).value
    mass = (
        await harness.execute(
            MASSING_COMMANDS.create_mass,
            {"name": "T", "profile_id": profile, "story_count": 4, "story_height": 3.0},
        )
    ).value

    stories = harness.capability(StoryToken)
    await stories.edit_stories(mass.id, lambda s: s.index == 1, {"programme": "Retail"})
    # Adding a floor must not wipe the annotation on the floor below it.
    await harness.execute(MASSING_COMMANDS.set_story_count, {"id": mass.id, "count": 6})
    assert stories.stories(mass.id)[1].programme == "Retail"


async def test_deleting_a_mass_and_undoing_restores_it_exactly(harness):
    await harness.load(massing_plugin)
    profile = (
        await harness.execute(MASSING_COMMANDS.sketch_profile, {"points": [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]})
    ).value
    mass = (
        await harness.execute(
            MASSING_COMMANDS.create_mass,
            {"name": "T", "profile_id": profile, "story_count": 3, "story_height": 3.0},
        )
    ).value
    stories = harness.capability(StoryToken)
    await stories.edit_stories(mass.id, lambda s: s.index == 0, {"programme": "Lobby"})

    await harness.execute(MASSING_COMMANDS.remove_mass, {"id": mass.id})
    assert harness.capability(MassingToken).get(mass.id) is None

    await harness.kernel.commands.undo()
    restored = harness.capability(MassingToken).get(mass.id)
    assert restored is not None and restored.name == "T"
    # The per-story extras come back too -- a fresh create would not reproduce them.
    assert stories.stories(mass.id)[0].programme == "Lobby"


async def test_a_locked_mass_refuses_edits(harness):
    await harness.load(massing_plugin)
    profile = (
        await harness.execute(MASSING_COMMANDS.sketch_profile, {"points": [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]})
    ).value
    masses = harness.capability(MassingToken)
    mass = (
        await harness.execute(
            MASSING_COMMANDS.create_mass,
            {"name": "T", "profile_id": profile, "story_count": 2},
        )
    ).value
    masses._stores.masses.update(mass.id, {"editable": False})
    result = await masses.update(mass.id, {"name": "renamed"})
    assert not result.ok and "locked" in result.error.message


async def test_plot_ratio_appears_once_a_site_is_set(harness):
    await harness.load(massing_plugin)
    profile = (
        await harness.execute(MASSING_COMMANDS.sketch_profile, {"points": [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)]})
    ).value
    mass = (
        await harness.execute(
            MASSING_COMMANDS.create_mass,
            {"name": "T", "profile_id": profile, "story_count": 5, "story_height": 3.0},
        )
    ).value
    metrics = harness.capability(MetricsToken)
    assert (await metrics.compute(mass.id)).value.floor_area_ratio is None

    await harness.execute(
        MASSING_COMMANDS.set_site_boundary,
        {"points": [(0, 0, 0), (40, 0, 0), (40, 40, 0), (0, 40, 0)], "max_floor_area_ratio": 2.0},
    )
    # 5 storeys x 400 m2 = 2000 m2 over a 1600 m2 site.
    assert (await metrics.compute(mass.id)).value.floor_area_ratio == pytest.approx(1.25)


async def test_promotion_without_a_handler_says_so_instead_of_pretending(harness):
    await harness.load(massing_plugin)
    profile = (
        await harness.execute(MASSING_COMMANDS.sketch_profile, {"points": [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]})
    ).value
    mass = (
        await harness.execute(
            MASSING_COMMANDS.create_mass, {"name": "T", "profile_id": profile, "story_count": 2}
        )
    ).value
    result = await harness.execute(
        MASSING_COMMANDS.promote, {"id": mass.id, "target": "building-systems"}
    )
    assert not result.ok
    assert result.error.code == "CAPABILITY_NOT_FOUND"
