"""Server-side spatial compute.

Everything here is work the browser would otherwise do. The tests are about correctness of the
answers, because the point of moving them server-side is that the client stops being able to check.
"""

from __future__ import annotations

import math

import pytest

from massingviser.geometry import Aabb, Bvh, SceneIndex, frustum_from_matrix


def _grid(side: int = 10, pitch: float = 2.0) -> tuple[list[str], list[Aabb]]:
    labels, boxes = [], []
    for x in range(side):
        for y in range(side):
            labels.append(f"E{x}-{y}")
            boxes.append(Aabb((x * pitch, y * pitch, 0.0), (x * pitch + 1.0, y * pitch + 1.0, 3.0)))
    return labels, boxes


# ---------------------------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------------------------


def test_overlap_and_penetration():
    a = Aabb((0, 0, 0), (10, 10, 3))
    b = Aabb((8, 8, 1), (18, 18, 4))
    assert a.overlaps(b)
    # Shallowest axis: 2 in x, 2 in y, 2 in z -- the distance you would move them apart.
    assert a.penetration(b) == pytest.approx(2.0)


def test_separated_boxes_report_negative_penetration():
    a = Aabb((0, 0, 0), (1, 1, 1))
    b = Aabb((5, 0, 0), (6, 1, 1))
    assert not a.overlaps(b)
    assert a.penetration(b) < 0
    # A clearance test is exactly "overlap within a tolerance".
    assert a.overlaps(b, tolerance=4.5)


def test_bounds_of_points():
    box = Aabb.of_points([(0, 0, 0), (5, -2, 9), (1, 1, 1)])
    assert box.min == (0.0, -2.0, 0.0)
    assert box.max == (5.0, 1.0, 9.0)


# ---------------------------------------------------------------------------------------------
# BVH
# ---------------------------------------------------------------------------------------------


def test_an_empty_index_answers_everything_with_nothing():
    empty = Bvh([], [])
    assert len(empty) == 0
    assert empty.bounds() is None
    assert empty.query_aabb(Aabb((0, 0, 0), (1, 1, 1))) == ()
    assert empty.raycast((0, 0, 0), (0, 0, 1)) == ()


def test_mismatched_labels_and_boxes_are_refused():
    with pytest.raises(ValueError):
        Bvh(["a"], [])


def test_an_aabb_query_finds_exactly_the_overlapping_boxes():
    labels, boxes = _grid()
    index = Bvh(labels, boxes)
    found = set(index.query_aabb(Aabb((0, 0, 0), (5, 5, 3))))
    expected = {f"E{x}-{y}" for x in range(3) for y in range(3)}
    assert found == expected


def test_a_ray_returns_hits_nearest_first():
    labels, boxes = _grid()
    index = Bvh(labels, boxes)
    hits = index.raycast((-10.0, 0.5, 1.5), (1, 0, 0))
    assert len(hits) == 10  # one per column along the row
    assert hits[0][0] == "E0-0"
    distances = [distance for _, distance in hits]
    assert distances == sorted(distances)


def test_a_ray_pointing_away_hits_nothing():
    labels, boxes = _grid()
    index = Bvh(labels, boxes)
    assert index.raycast((-10.0, 0.5, 1.5), (-1, 0, 0)) == ()


def test_a_ray_parallel_to_an_axis_does_not_divide_by_zero():
    """Orthographic cameras make this the common case, not the edge case."""
    index = Bvh(["a"], [Aabb((0, 0, 0), (1, 1, 1))])
    assert index.raycast((0.5, 0.5, -5), (0, 0, 1))[0][0] == "a"


def test_max_distance_truncates_a_ray():
    labels, boxes = _grid()
    index = Bvh(labels, boxes)
    near = index.raycast((-10.0, 0.5, 1.5), (1, 0, 0), max_distance=13.0)
    assert len(near) < 10


def test_a_zero_direction_ray_is_refused_rather_than_dividing():
    index = Bvh(["a"], [Aabb((0, 0, 0), (1, 1, 1))])
    assert index.raycast((0, 0, 0), (0, 0, 0)) == ()


def test_a_frustum_keeps_what_is_in_front_of_every_plane():
    labels, boxes = _grid()
    index = Bvh(labels, boxes)
    # Inward normal along -x with d=6 keeps x <= 6; the rest are wide open.
    planes = [(-1, 0, 0, 6), (1, 0, 0, 100), (0, -1, 0, 100), (0, 1, 0, 100), (0, 0, -1, 100), (0, 0, 1, 100)]
    inside = index.query_frustum(planes)
    assert len(inside) == 40  # columns x = 0..3, ten rows each
    assert all(int(label[1:].split("-")[0]) * 2 <= 6 for label in inside)


def test_pairwise_overlap_finds_only_real_pairs():
    labels, boxes = _grid()
    left = Bvh(labels, boxes)
    right = Bvh(
        ["P1", "P2"],
        [Aabb((0.5, 0.5, 1.0), (1.5, 1.5, 2.0)), Aabb((500, 500, 0), (501, 501, 1))],
    )
    pairs = left.overlapping_pairs(right)
    assert [(a, b) for a, b, _ in pairs] == [("E0-0", "P1")]
    assert pairs[0][2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------
# Frustum extraction
# ---------------------------------------------------------------------------------------------


def _orthographic(half: float) -> list[float]:
    """Column-major orthographic matrix, translation at 12-14, looking down -z."""
    return [
        1 / half, 0, 0, 0,
        0, 1 / half, 0, 0,
        0, 0, -1 / half, 0,
        0, 0, 0, 1,
    ]


def test_planes_come_out_normalised():
    """An unnormalised plane classifies correctly but its distances are meaningless."""
    for plane in frustum_from_matrix(_orthographic(10.0)):
        assert math.sqrt(sum(component**2 for component in plane[:3])) == pytest.approx(1.0)


def test_a_matrix_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError):
        frustum_from_matrix([1, 0, 0])


def test_culling_with_an_orthographic_camera_keeps_what_is_inside_it():
    boxes = {
        "inside": Aabb((-1, -1, -1), (1, 1, 1)),
        "outside": Aabb((50, 50, 0), (51, 51, 1)),
    }
    index = SceneIndex(boxes)
    visible = index.cull(_orthographic(10.0))
    assert visible == ("inside",)


# ---------------------------------------------------------------------------------------------
# Scene index
# ---------------------------------------------------------------------------------------------


def _scene() -> SceneIndex:
    # Two towers whose footprints overlap between x,y 10..20.
    a = [(0, 0), (20, 0), (20, 20), (0, 20)]
    b = [(10, 10), (30, 10), (30, 30), (10, 30)]
    elements = []
    for index in range(3):
        elements.append((f"A:{index:03d}", "A", a, index * 3.0, 3.0))
        elements.append((f"B:{index:03d}", "B", b, index * 3.0, 3.0))
    return SceneIndex.from_extrusions(elements)


def test_a_scene_index_is_built_from_extrusions():
    scene = _scene()
    assert len(scene) == 6
    assert scene.bounds.max[2] == pytest.approx(9.0)
    assert scene.box_of("A:000").min == (0.0, 0.0, 0.0)


def test_picking_returns_global_ids_nearest_first():
    """An answer from the index is already in the identity everything else keys on."""
    scene = _scene()
    picks = scene.pick((15, 15, 100), (0, 0, -1))
    assert picks[0].global_id in ("A:002", "B:002")  # the topmost storeys
    assert picks[0].distance < picks[-1].distance
    assert {pick.global_id for pick in picks} == {f"{m}:{i:03d}" for m in "AB" for i in range(3)}


def test_picking_outside_the_model_finds_nothing():
    assert _scene().pick((500, 500, 100), (0, 0, -1)) == ()


def test_a_pick_limit_is_honoured():
    assert len(_scene().pick((15, 15, 100), (0, 0, -1), limit=2)) == 2


def test_clash_between_two_groups_pairs_only_across_them():
    scene = _scene()
    candidates = scene.clash("A", "B")
    assert candidates
    # Never an element against itself, and never two from the same group.
    for candidate in candidates:
        assert candidate.a != candidate.b
        assert candidate.a.startswith("A") != candidate.b.startswith("A")


def test_clash_penetration_is_the_storey_overlap():
    scene = _scene()
    matching = [c for c in scene.clash("A", "B") if c.a == "A:000" and c.b == "B:000"]
    assert matching and matching[0].penetration == pytest.approx(3.0)


def test_clash_against_an_unknown_group_is_empty_not_an_error():
    assert _scene().clash("A", "does-not-exist") == ()


def test_a_tolerance_turns_clash_into_a_clearance_test():
    boxes = {"a": Aabb((0, 0, 0), (1, 1, 1)), "b": Aabb((3, 0, 0), (4, 1, 1))}
    scene = SceneIndex(boxes, groups={"a": "left", "b": "right"})
    assert scene.clash("left", "right") == ()
    assert scene.clash("left", "right", tolerance=2.5)
