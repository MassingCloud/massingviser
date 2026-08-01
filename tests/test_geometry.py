"""Server-side spatial compute.

Everything here is work the browser would otherwise do. The tests are about correctness of the
answers, because the point of moving them server-side is that the client stops being able to check.
"""

from __future__ import annotations

import math
import struct
import warnings

import numpy as np
import pytest

from massingviser.geometry import Aabb, Bvh, SceneIndex, frustum_from_matrix
from massingviser.geometry.normals import compute_shading, face_normals
from massingviser.geometry.payload import (
    FLAG_NORMALS,
    FORMAT_VERSION,
    MAGIC,
    MeshInput,
    build_geometry_payloads,
    chunk_meshes,
    decode_mesh_batch,
    encode_mesh_batch,
)


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A mesh dense enough to actually need decimating."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    vertices = [tuple(c / math.sqrt(sum(v * v for v in p)) for c in p) for p in vertices]
    for _ in range(subdivisions):
        new_faces = []
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int, *, cache: dict[tuple[int, int], int] = cache) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                point = tuple((vertices[a][i] + vertices[b][i]) / 2 for i in range(3))
                length = math.sqrt(sum(c * c for c in point))
                vertices.append(tuple(c / length for c in point))
                cache[key] = len(vertices) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.asarray(vertices), np.asarray(faces)


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
    planes = [
        (-1, 0, 0, 6),
        (1, 0, 0, 100),
        (0, -1, 0, 100),
        (0, 1, 0, 100),
        (0, 0, -1, 100),
        (0, 0, 1, 100),
    ]
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
        1 / half,
        0,
        0,
        0,
        0,
        1 / half,
        0,
        0,
        0,
        0,
        -1 / half,
        0,
        0,
        0,
        0,
        1,
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


# ---------------------------------------------------------------------------------------------
# Geometry payloads
#
# The wire format is the part a C++, C# or Rust importer will be held to, so these tests assert
# byte offsets and exact sizes rather than "it round-trips" -- a reader written from the docstring
# has to agree with this, and only the numbers make that checkable.
# ---------------------------------------------------------------------------------------------


def _cube(
    offset: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    return [(x + offset, y + offset, z + offset) for x, y, z in vertices], faces


def test_a_payload_is_exactly_the_size_the_format_says():
    vertices, faces = _cube()
    payload = encode_mesh_batch([MeshInput("A", np.asarray(vertices), np.asarray(faces))])
    # 32 header + 1 directory entry + 8 vertices x 3 floats + 36 indices.
    assert payload.byte_length == 32 + 16 + 8 * 3 * 4 + 36 * 4


def test_a_payload_starts_with_its_magic_and_version():
    """An importer's first check, and the reason a wrong file fails loudly."""
    payload = encode_mesh_batch([MeshInput("A", *(np.asarray(x) for x in _cube()))])
    assert payload.data[:4] == MAGIC
    assert struct.unpack("<I", payload.data[4:8])[0] == FORMAT_VERSION


def test_the_buffer_round_trips_through_the_decoder():
    vertices, faces = _cube()
    meshes = [
        MeshInput("A", np.asarray(vertices), np.asarray(faces)),
        MeshInput("B", np.asarray(vertices) + 10.0, np.asarray(faces)),
    ]
    decoded = decode_mesh_batch(encode_mesh_batch(meshes).data)
    assert len(decoded) == 2
    assert np.allclose(decoded[0].vertices, vertices)
    assert np.allclose(decoded[1].vertices, np.asarray(vertices) + 10.0)
    assert np.array_equal(decoded[1].faces, faces)


def test_indices_are_local_to_their_mesh():
    """So a consumer can upload one element without rebasing every index in the chunk."""
    vertices, faces = _cube()
    meshes = [MeshInput(name, np.asarray(vertices), np.asarray(faces)) for name in ("A", "B")]
    decoded = decode_mesh_batch(encode_mesh_batch(meshes).data)
    assert decoded[1].faces.max() < len(vertices)


def test_the_id_is_the_content_and_nothing_else():
    """Two models with identical geometry share the payload; the GlobalIds are not in the buffer."""
    vertices, faces = _cube()
    first = encode_mesh_batch([MeshInput("wall-a", np.asarray(vertices), np.asarray(faces))])
    second = encode_mesh_batch([MeshInput("column-z", np.asarray(vertices), np.asarray(faces))])
    assert first.id == second.id
    assert len(first.id) == 32


def test_moving_a_vertex_changes_the_id():
    vertices, faces = _cube()
    moved = np.asarray(vertices)
    moved[0][0] += 0.001
    a = encode_mesh_batch([MeshInput("A", np.asarray(vertices), np.asarray(faces))])
    b = encode_mesh_batch([MeshInput("A", moved, np.asarray(faces))])
    assert a.id != b.id


def test_the_same_geometry_at_a_different_lod_is_a_different_payload():
    """The level is in the header, so a client cannot mistake one for the other."""
    vertices, faces = _cube()
    mesh = [MeshInput("A", np.asarray(vertices), np.asarray(faces))]
    assert encode_mesh_batch(mesh, lod=0).id != encode_mesh_batch(mesh, lod=2).id


def test_a_face_indexing_past_its_own_vertices_is_refused():
    """It would silently read the next mesh in the chunk and draw garbage."""
    vertices, _ = _cube()
    with pytest.raises(ValueError, match="indexing vertex"):
        encode_mesh_batch([MeshInput("A", np.asarray(vertices), np.asarray([(0, 1, 99)]))])


def test_a_truncated_payload_is_refused_not_misread():
    payload = encode_mesh_batch([MeshInput("A", *(np.asarray(x) for x in _cube()))])
    with pytest.raises(ValueError, match="carries"):
        decode_mesh_batch(payload.data[:-4])


def test_something_that_is_not_a_payload_is_refused():
    with pytest.raises(ValueError, match="Not a mesh payload"):
        decode_mesh_batch(b"GLTF" + bytes(64))


def test_chunking_never_splits_a_mesh():
    """A mesh larger than the budget gets its own chunk rather than being cut in half."""
    vertices, faces = _cube()
    meshes = [MeshInput(f"E{i}", np.asarray(vertices), np.asarray(faces)) for i in range(5)]
    chunks = chunk_meshes(meshes, chunk_vertices=4)  # smaller than one cube
    assert len(chunks) == 5
    assert all(len(chunk) == 1 for chunk in chunks)


def test_chunking_packs_up_to_the_budget():
    vertices, faces = _cube()
    meshes = [MeshInput(f"E{i}", np.asarray(vertices), np.asarray(faces)) for i in range(6)]
    chunks = chunk_meshes(meshes, chunk_vertices=16)  # two cubes per chunk
    assert [len(chunk) for chunk in chunks] == [2, 2, 2]


def test_editing_one_element_rewrites_one_chunk():
    """The claim content addressing exists to make true."""
    vertices, faces = _cube()
    meshes = {f"E{i:02d}": (np.asarray(vertices) + i * 3.0, np.asarray(faces)) for i in range(6)}
    # A shaded cube is 24 vertices, not 8 -- every corner creases -- so two fit in a 48 budget.
    before = build_geometry_payloads(meshes, lod_budgets=(), chunk_vertices=48)

    moved = dict(meshes)
    moved["E03"] = (meshes["E03"][0] + 0.5, meshes["E03"][1])
    after = build_geometry_payloads(moved, lod_budgets=(), chunk_vertices=48)

    kept = {p.id for p in before.payloads} & {p.id for p in after.payloads}
    assert len(before.payloads) == 3
    assert len(kept) == 2  # only the chunk holding E03 was rewritten


def test_chunk_boundaries_do_not_move_with_dictionary_order():
    """Ids have to survive a differently-ordered dict, or every rebuild invalidates the cache."""
    vertices, faces = _cube()
    forward = {f"E{i:02d}": (np.asarray(vertices) + i * 3.0, np.asarray(faces)) for i in range(6)}
    backward = dict(reversed(list(forward.items())))
    assert [p.id for p in build_geometry_payloads(forward, lod_budgets=()).payloads] == [
        p.id for p in build_geometry_payloads(backward, lod_budgets=()).payloads
    ]


def test_a_lod_level_that_saves_nothing_is_not_shipped():
    """A cube is already under every budget; duplicating it three times helps nobody."""
    vertices, faces = _cube()
    built = build_geometry_payloads({"A": (np.asarray(vertices), np.asarray(faces))})
    assert [placement.lod for placement in built.placements["A"]] == [0]


def test_the_lod_ladder_gets_monotonically_coarser():
    sphere_vertices, sphere_faces = _icosphere(4)
    built = build_geometry_payloads({"A": (sphere_vertices, sphere_faces)})
    counts = [placement.face_count for placement in built.placements["A"]]
    assert len(counts) > 1
    assert counts == sorted(counts, reverse=True)
    # Each level earns its bytes: no level is within a third of the one above it.
    assert all(
        later <= earlier * 0.7 for earlier, later in zip(counts[:-1], counts[1:], strict=True)
    )


def test_levels_are_numbered_from_the_finest():
    sphere_vertices, sphere_faces = _icosphere(4)
    built = build_geometry_payloads({"A": (sphere_vertices, sphere_faces)})
    assert [placement.lod for placement in built.placements["A"]] == list(
        range(len(built.placements["A"]))
    )


def test_an_element_with_no_faces_gets_no_placement():
    built = build_geometry_payloads({"empty": ([], []), "A": _cube()})
    assert "empty" not in built.placements
    assert "A" in built.placements


def test_no_geometry_at_all_produces_no_payloads():
    assert build_geometry_payloads({}).payloads == ()


# ---------------------------------------------------------------------------------------------
# Normals
#
# The whole design rests on one claim: a single crease angle shades a box flat and a sphere smooth.
# These test that claim from both ends, plus the analytic case where the right answer is known --
# on a unit sphere the correct smooth normal at a vertex is its own position.
# ---------------------------------------------------------------------------------------------


def _cube_solid() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _cube()
    return np.asarray(vertices), np.asarray(faces)


def test_a_box_shades_flat():
    """Three faces at 90 degrees put the vertex average 54.7 degrees off each -- every corner splits."""
    vertices, faces = _cube_solid()
    shaded = compute_shading(vertices, faces)
    assert shaded.vertex_count == 24  # 8 corners x 3 faces
    assert len(shaded.faces) == len(faces)
    # Exactly six distinct normals, one per face of the box, each along an axis.
    distinct = np.unique(np.round(shaded.normals, 5), axis=0)
    assert len(distinct) == 6
    assert np.allclose(np.abs(distinct).sum(axis=1), 1.0)


def test_a_sphere_shades_smooth():
    vertices, faces = _icosphere(3)
    shaded = compute_shading(vertices, faces)
    assert shaded.vertex_count == len(vertices)  # nothing split


def test_sphere_normals_are_analytically_right():
    """On a unit sphere the smooth normal at a point is the point. Nothing to trust here."""
    vertices, faces = _icosphere(3)
    shaded = compute_shading(vertices, faces)
    assert np.allclose(shaded.normals, shaded.vertices, atol=0.02)


def test_normals_come_out_unit_length():
    for vertices, faces in (_cube_solid(), _icosphere(2)):
        shaded = compute_shading(vertices, faces)
        assert np.allclose(np.linalg.norm(shaded.normals, axis=1), 1.0)


def test_a_crease_angle_of_180_never_splits():
    """The setting is a real dial, and both ends of it behave."""
    vertices, faces = _cube_solid()
    shaded = compute_shading(vertices, faces, crease_degrees=180.0)
    assert shaded.vertex_count == 8


def test_a_crease_angle_of_zero_splits_everything():
    vertices, faces = _icosphere(2)
    shaded = compute_shading(vertices, faces, crease_degrees=0.0)
    assert shaded.vertex_count == len(faces) * 3


def test_an_impossible_crease_angle_is_refused():
    vertices, faces = _cube_solid()
    with pytest.raises(ValueError, match="between 0 and 180"):
        compute_shading(vertices, faces, crease_degrees=270.0)


def test_a_degenerate_triangle_yields_no_nan():
    """A NaN normal poisons every vertex it touches and the model renders black."""
    vertices = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0)], dtype=float)  # collinear
    shaded = compute_shading(vertices, np.array([(0, 1, 2)]))
    assert not np.isnan(shaded.normals).any()


def test_face_normals_are_area_weighted():
    """Length is twice the area, which is what makes a sliver count for less than a large face."""
    vertices = np.array([(0, 0, 0), (2, 0, 0), (0, 2, 0)], dtype=float)
    normal = face_normals(vertices, np.array([(0, 1, 2)]))[0]
    assert np.linalg.norm(normal) == pytest.approx(4.0)  # 2 x area, area = 2


def test_an_empty_mesh_shades_to_nothing():
    shaded = compute_shading(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert shaded.vertex_count == 0


# --- and through the wire format -------------------------------------------------------------


def test_a_shaded_payload_carries_a_normal_per_vertex():
    vertices, faces = _cube_solid()
    built = build_geometry_payloads({"A": (vertices, faces)})
    decoded = decode_mesh_batch(built.payloads[0].data)[0]
    assert decoded.normals is not None
    assert len(decoded.normals) == len(decoded.vertices)
    assert np.allclose(np.linalg.norm(decoded.normals, axis=1), 1.0, atol=1e-6)


def test_shading_can_be_turned_off():
    vertices, faces = _cube_solid()
    built = build_geometry_payloads({"A": (vertices, faces)}, shade=False)
    decoded = decode_mesh_batch(built.payloads[0].data)[0]
    assert decoded.normals is None
    assert len(decoded.vertices) == 8  # unsplit


def test_a_shaded_payload_is_a_different_object_from_an_unshaded_one():
    """Content addressing has to see the normals, or a client caches the wrong buffer."""
    vertices, faces = _cube_solid()
    shaded = build_geometry_payloads({"A": (vertices, faces)}).payloads[0]
    plain = build_geometry_payloads({"A": (vertices, faces)}, shade=False).payloads[0]
    assert shaded.id != plain.id


def test_a_shaded_payload_declares_the_normals_flag():
    vertices, faces = _cube_solid()
    built = build_geometry_payloads({"A": (vertices, faces)})
    flags = struct.unpack("<I", built.payloads[0].data[12:16])[0]
    assert flags & FLAG_NORMALS
    plain = build_geometry_payloads({"A": (vertices, faces)}, shade=False)
    assert not struct.unpack("<I", plain.payloads[0].data[12:16])[0] & FLAG_NORMALS


def test_a_payload_is_exactly_the_size_the_shaded_format_says():
    vertices, faces = _cube_solid()
    built = build_geometry_payloads({"A": (vertices, faces)})
    # 32 header + 1 directory entry + 24 vertices x (3 positions + 3 normals) + 36 indices.
    assert built.payloads[0].byte_length == 32 + 16 + 24 * 3 * 4 * 2 + 36 * 4


def test_a_chunk_cannot_be_half_shaded():
    """The flag is in the header, so a chunk that mixed them could not describe itself."""
    vertices, faces = _cube_solid()
    shaded = compute_shading(vertices, faces)
    with pytest.raises(ValueError, match="every mesh or for none"):
        encode_mesh_batch(
            [
                MeshInput("A", shaded.vertices, shaded.faces, shaded.normals),
                MeshInput("B", vertices, faces),
            ]
        )


def test_a_normal_count_that_disagrees_with_the_vertices_is_refused():
    vertices, faces = _cube_solid()
    with pytest.raises(ValueError, match="normals for"):
        encode_mesh_batch([MeshInput("A", vertices, faces, np.zeros((3, 3)))])


def test_a_version_1_payload_still_reads():
    """Older buffers stay readable: the layout is identical minus the normals block."""
    vertices, faces = _cube_solid()
    payload = encode_mesh_batch([MeshInput("A", vertices, faces)])
    v1 = payload.data[:4] + struct.pack("<I", 1) + payload.data[8:]
    decoded = decode_mesh_batch(v1)
    assert decoded[0].normals is None
    assert np.allclose(decoded[0].vertices, vertices)


def test_a_future_version_is_refused_rather_than_misread():
    vertices, faces = _cube_solid()
    payload = encode_mesh_batch([MeshInput("A", vertices, faces)])
    future = payload.data[:4] + struct.pack("<I", 99) + payload.data[8:]
    with pytest.raises(ValueError, match="this build reads"):
        decode_mesh_batch(future)


def test_each_lod_level_is_shaded_from_its_own_geometry():
    """Reusing level 0's normals would light a simplified surface as though it kept its detail."""
    vertices, faces = _icosphere(4)
    built = build_geometry_payloads({"A": (vertices, faces)})
    assert len(built.placements["A"]) > 1
    for placement in built.placements["A"]:
        payload = built.by_id(placement.payload_id)
        mesh = decode_mesh_batch(payload.data)[placement.geometry_index]
        assert mesh.normals is not None
        assert len(mesh.normals) == len(mesh.vertices)
        # Still a sphere at every level, so the analytic check still holds.
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Rays that run along a box face
#
# The slab test divides by each direction component, so an axis the ray does not travel along
# divides by zero. Where the origin also sits exactly on one of that axis's faces the bound is
# `0 * inf` -- NaN -- and how those are handled decides whether an ordinary plan-view pick works.
# ---------------------------------------------------------------------------------------------

SLAB = Aabb((0.0, 0.0, 5.0), (10.0, 10.0, 8.0))


@pytest.mark.parametrize(
    ("z", "why"),
    [
        (5.0, "exactly on the base face"),
        (8.0, "exactly on the top face"),
        (5.0001, "a tenth of a millimetre inside"),
        (6.5, "mid-slab"),
    ],
)
def test_a_horizontal_pick_at_a_slab_face_still_finds_the_slab(z, why):
    """Levels are exact numbers, so a plan-view ray at a storey elevation is the normal case.

    Substituting the opposite face's bound for the NaN made `t_near` positive infinity, so the
    box was rejected and the pick returned nothing -- while `query_aabb` over the same geometry
    happily reported the element as present.
    """
    index = Bvh(["slab"], [SLAB])
    assert index.raycast((-1.0, 5.0, z), (1.0, 0.0, 0.0)) == (("slab", pytest.approx(1.0)),), why


@pytest.mark.parametrize("z", [4.999, 8.001, -100.0])
def test_a_horizontal_ray_that_misses_the_slab_still_misses_it(z):
    """The fix must not turn every parallel ray into a hit."""
    assert Bvh(["slab"], [SLAB]).raycast((-1.0, 5.0, z), (1.0, 0.0, 0.0)) == ()


def test_a_ray_from_the_model_origin_hits_a_box_cornered_there():
    index = Bvh(["b"], [Aabb((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))])
    assert index.raycast((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == (("b", pytest.approx(0.0)),)
    # ...and one cornered elsewhere is still missed.
    assert index.raycast((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0)) == ()


def test_a_zero_thickness_element_reports_a_real_distance_not_nan():
    """A NaN distance is worse than a wrong one: `sorted` cannot order it, so "nearest first"
    silently becomes insertion order."""
    index = Bvh(
        ["plate", "column"],
        [Aabb((0.0, 0.0, 5.0), (10.0, 10.0, 5.0)), Aabb((6.0, 2.0, 0.0), (8.0, 4.0, 8.0))],
    )
    hits = index.raycast((-1.0, 3.0, 5.0), (1.0, 0.0, 0.0))
    assert {label for label, _ in hits} == {"plate", "column"}
    assert all(not math.isnan(distance) for _, distance in hits)
    # Nearest first, which is only meaningful once the distances are numbers.
    assert [distance for _, distance in hits] == sorted(distance for _, distance in hits)


@pytest.mark.parametrize(
    "direction",
    [(float("nan"), 0.0, 1.0), (float("inf"), 0.0, 0.0), (0.0, 0.0, 0.0)],
    ids=["nan", "infinite", "zero"],
)
def test_a_direction_that_is_not_a_direction_returns_nothing(direction):
    """Rejected up front, rather than divided by and rejected later.

    The slab test refuses these anyway, so the empty result alone would pass either way. What the
    up-front check buys is that nothing ever divides by a norm of NaN or infinity -- so the caller
    is not handed a numpy RuntimeWarning from three frames down for an input the API could have
    turned away.
    """
    index = Bvh(["b"], [Aabb((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert index.raycast((5.0, 5.0, -1.0), direction) == ()


def test_a_union_over_a_generator_is_the_union_of_all_of_it():
    """The parameter is an `Iterable`; reading it twice consumed it on the first pass."""
    boxes = (Aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), Aabb((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)))
    united = Aabb.union(box for box in boxes)
    assert tuple(float(v) for v in united.min) == (0.0, 0.0, 0.0)
    assert tuple(float(v) for v in united.max) == (3.0, 3.0, 3.0)
