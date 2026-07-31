"""Optional integrations, and the LOD they feed.

Every test here skips cleanly when its extra is absent, because that is the contract: a deployment
without ifcopenshell runs the other fifteen families unchanged.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from massingviser.adapters import REQUIREMENTS, available, load, missing
from massingviser.geometry import Aabb, cluster_decimate, decimate_to_budget, lod_chain

ifc_only = pytest.mark.skipif("ifc" not in available(), reason="ifcopenshell not installed")
solids_only = pytest.mark.skipif("solids" not in available(), reason="trimesh/manifold3d absent")
crs_only = pytest.mark.skipif("crs" not in available(), reason="pyproj not installed")


# ---------------------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------------------


def test_importing_the_adapters_package_never_fails():
    """A machine with no extras must still import the platform."""
    assert set(available()) <= set(REQUIREMENTS)
    assert set(missing()) <= set(REQUIREMENTS)
    assert set(available()) & set(missing()) == set()


def test_an_unknown_adapter_is_named_not_guessed():
    with pytest.raises(KeyError, match="No adapter"):
        load("nonsense")


def test_a_missing_extra_would_name_what_to_install():
    for name, requirements in REQUIREMENTS.items():
        assert requirements, f"{name} declares no requirements"


# ---------------------------------------------------------------------------------------------
# LOD (no extras needed)
# ---------------------------------------------------------------------------------------------


def _sphere_mesh(subdivisions: int = 3):
    """A unit icosphere, built without trimesh so the LOD tests need no extras."""
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
    vertices = [tuple(v / math.sqrt(sum(c * c for c in p)) for v in p) for p in vertices]
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


def test_decimation_honours_a_face_budget():
    vertices, faces = _sphere_mesh(3)
    for budget in (1000, 200, 50):
        result = decimate_to_budget(vertices, faces, max_faces=budget)
        assert result.face_count <= budget


def test_a_mesh_already_within_budget_is_returned_untouched():
    """A budget is a ceiling, not a target."""
    vertices, faces = _sphere_mesh(1)
    result = decimate_to_budget(vertices, faces, max_faces=10_000)
    assert result.face_count == len(faces)
    assert result.cell_size == 0.0


def test_decimation_is_deterministic():
    """The output is content-addressed and cached by hash, so it has to be reproducible."""
    vertices, faces = _sphere_mesh(3)
    first = cluster_decimate(vertices, faces, cell_size=0.3)
    second = cluster_decimate(vertices, faces, cell_size=0.3)
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)


def test_degenerate_faces_are_dropped_not_kept_flat():
    """Zero-area triangles break normals and confuse a renderer."""
    vertices, faces = _sphere_mesh(2)
    result = cluster_decimate(vertices, faces, cell_size=0.5)
    a, b, c = result.faces[:, 0], result.faces[:, 1], result.faces[:, 2]
    assert np.all((a != b) & (b != c) & (a != c))


def test_a_decimated_mesh_indexes_only_vertices_it_has():
    vertices, faces = _sphere_mesh(3)
    result = cluster_decimate(vertices, faces, cell_size=0.4)
    assert result.faces.max() < result.vertex_count


def test_lod_levels_get_coarser_and_are_each_built_from_the_original():
    vertices, faces = _sphere_mesh(3)
    chain = lod_chain(vertices, faces, budgets=(2000, 500, 100))
    counts = [level.face_count for level in chain]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] <= 100


def test_a_non_positive_cell_is_refused():
    vertices, faces = _sphere_mesh(1)
    with pytest.raises(ValueError):
        cluster_decimate(vertices, faces, cell_size=0.0)


def test_an_empty_mesh_decimates_to_nothing_without_raising():
    result = decimate_to_budget([], [], max_faces=10)
    assert result.face_count == 0


# ---------------------------------------------------------------------------------------------
# IFC
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ifc_path(tmp_path_factory):
    if "ifc" not in available():
        pytest.skip("ifcopenshell not installed")
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.geometry
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit

    file = ifcopenshell.file(schema="IFC4")
    project = ifcopenshell.api.root.create_entity(file, ifc_class="IfcProject", name="T")
    ifcopenshell.api.unit.assign_unit(file)
    model = ifcopenshell.api.context.add_context(file, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        file,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model,
    )
    site = ifcopenshell.api.root.create_entity(file, ifc_class="IfcSite", name="S")
    building = ifcopenshell.api.root.create_entity(file, ifc_class="IfcBuilding", name="B")
    storey = ifcopenshell.api.root.create_entity(file, ifc_class="IfcBuildingStorey", name="L00")
    ifcopenshell.api.aggregate.assign_object(file, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(file, products=[building], relating_object=site)
    ifcopenshell.api.aggregate.assign_object(file, products=[storey], relating_object=building)

    for index in range(3):
        wall = ifcopenshell.api.root.create_entity(file, ifc_class="IfcWall", name=f"Wall {index}")
        representation = ifcopenshell.api.geometry.add_wall_representation(
            file, context=body, length=5.0, height=3.0, thickness=0.2
        )
        ifcopenshell.api.geometry.assign_representation(
            file, product=wall, representation=representation
        )
        ifcopenshell.api.geometry.edit_object_placement(
            file,
            product=wall,
            matrix=np.array(
                [[1, 0, 0, index * 6.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
        )
        ifcopenshell.api.spatial.assign_container(file, products=[wall], relating_structure=storey)

    path = tmp_path_factory.mktemp("ifc") / "walls.ifc"
    file.write(str(path))
    return str(path)


@ifc_only
def test_an_ifc_file_is_parsed_and_tessellated_server_side(ifc_path):
    ifc = load("ifc")
    model = ifc.open_ifc(ifc_path, model_id="m1")
    assert len(model) == 3
    assert all(element.ifc_class == "IfcWall" for element in model.elements)
    assert all(element.vertices is not None for element in model.elements)


@ifc_only
def test_geometry_arrives_in_world_coordinates_and_metres(ifc_path):
    """A file authored in millimetres must not leak millimetres."""
    ifc = load("ifc")
    model = ifc.open_ifc(ifc_path, model_id="m1")
    assert model.source_units == "mm"  # provenance
    boxes = sorted((element.box for element in model.elements), key=lambda box: box.min[0])
    # 5 m long, 0.2 m thick, 3 m tall, spaced 6 m apart -- in metres, placed in the world.
    assert boxes[0].min == pytest.approx((0.0, 0.0, 0.0))
    assert boxes[0].max == pytest.approx((5.0, 0.2, 3.0))
    assert boxes[1].min[0] == pytest.approx(6.0)


@ifc_only
def test_spatial_containment_is_read_not_inferred_from_height(ifc_path):
    ifc = load("ifc")
    model = ifc.open_ifc(ifc_path, model_id="m1")
    storeys = {element.storey_global_id for element in model.elements}
    assert len(storeys) == 1 and None not in storeys


@ifc_only
def test_spatial_structure_is_excluded_from_geometry(ifc_path):
    """Tessellating an IfcSpace produces a solid that clashes with everything inside it."""
    ifc = load("ifc")
    model = ifc.open_ifc(ifc_path, model_id="m1")
    assert not any(element.ifc_class in ifc.SPATIAL_CLASSES for element in model.elements)


@ifc_only
def test_one_parsed_model_satisfies_every_capability_that_wants_elements(ifc_path):
    """Because all of them key on GlobalId, which comes from the file."""
    ifc = load("ifc")
    source = ifc.IfcModelSource(ifc.open_ifc(ifc_path, model_id="m1"))

    assert len(source.elements("m1")) == 3  # estimating
    assert len(source.snapshot("m1", "1")) == 3  # coordination
    assert len(source.nodes()) == 3  # engine bridge
    assert len(source.global_ids("m1")) == 3  # markup
    assert len(source.boxes()) == 3  # geometry

    first = source.global_ids("m1")[0]
    assert source.exists("m1", first)
    assert not source.exists("m1", "not-a-global-id")
    assert source.elements("other-model") == ()


@ifc_only
def test_scene_nodes_keep_property_sets_nested(ifc_path):
    """An importer that receives a flat map has lost what each value means."""
    ifc = load("ifc")
    source = ifc.IfcModelSource(ifc.open_ifc(ifc_path, model_id="m1"))
    for node in source.nodes():
        assert all(isinstance(values, dict) for values in node.property_sets.values())


@ifc_only
async def test_an_unreadable_payload_is_reported_not_raised(harness):
    ifc = load("ifc")
    adapter = ifc.IfcImportAdapter()
    result = await adapter.read(b"this is not an IFC file")
    assert not result.ok and "could not be read" in result.error.message


@ifc_only
async def test_importing_ifc_through_the_command_bus_rewires_the_platform(ifc_path):
    """One import, six capabilities, no plugin changed."""
    from massingviser import build_kernel
    from massingviser.geometry import SpatialIndexToken
    from massingviser.plugins.estimating import ModelElementSourceToken
    from massingviser.plugins.interop import INTEROP_COMMANDS

    kernel = build_kernel()
    await kernel.start()
    payload = Path(ifc_path).read_bytes()

    summary = (
        await kernel.commands.execute(
            INTEROP_COMMANDS.import_payload, {"payload": payload, "filename": "walls.ifc"}
        )
    ).value
    assert summary.format == "ifc" and summary.records == 3

    # The IFC model now outranks the massing bridge, because it registered at a higher priority.
    elements = kernel.capabilities.get(ModelElementSourceToken).elements("ifc-1")
    assert len(elements) == 3

    index = kernel.capabilities.get(SpatialIndexToken).build()
    assert len(index) == 3
    picks = index.pick((2.5, 0.1, 50), (0, 0, -1))
    assert picks and picks[0].global_id in {e.global_id for e in elements}
    await kernel.stop()


@ifc_only
async def test_an_imported_ifc_model_reaches_the_engine_as_drawable_geometry(ifc_path):
    """The last mile: a file on disk becomes buffers a renderer can upload, with nothing manual."""
    from massingviser import build_kernel
    from massingviser.geometry import MESH_ENCODING, decode_mesh_batch
    from massingviser.plugins.engine import ENGINE_COMMANDS, SceneExportToken
    from massingviser.plugins.interop import INTEROP_COMMANDS

    kernel = build_kernel()
    await kernel.start()
    await kernel.commands.execute(
        INTEROP_COMMANDS.import_payload,
        {"payload": Path(ifc_path).read_bytes(), "filename": "walls.ifc"},
    )

    service = kernel.capabilities.get(SceneExportToken)
    package = (await service.build()).value
    geometry_payloads = [p for p in package.payloads if p.role == "geometry"]
    assert geometry_payloads, "the IFC adapter published no geometry"
    assert all(p.encoding == MESH_ENCODING for p in geometry_payloads)

    drawable = [node for node in package.nodes if node.geometry]
    assert len(drawable) == 3
    # The scene validates as renderable, not as the semantic half.
    report = service.validate(package)
    assert report.ok
    assert not any("semantic half only" in warning for warning in report.warnings)

    # And the bytes come back through the command bus, decodable, with the wall in them.
    wall = drawable[0]
    data = (
        await kernel.commands.execute(
            ENGINE_COMMANDS.payload, {"payloadId": wall.geometry[0].payload_id}
        )
    ).value
    meshes = decode_mesh_batch(data)
    mesh = meshes[wall.geometry[0].geometry_index]
    assert len(mesh.faces) == wall.geometry[0].face_count
    # A 5 x 0.2 x 3 wall, in metres, in world coordinates -- the units the format promises.
    span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    assert sorted(round(float(v), 3) for v in span) == [0.2, 3.0, 5.0]
    await kernel.stop()


# ---------------------------------------------------------------------------------------------
# Narrow-phase clash
# ---------------------------------------------------------------------------------------------


class _Meshes:
    """Two crossing walls and one far away."""

    def __init__(self) -> None:
        import trimesh

        a = trimesh.creation.box(extents=(4, 0.3, 3))
        b = trimesh.creation.box(extents=(0.3, 4, 3))
        far = trimesh.creation.box(extents=(1, 1, 1))
        far.apply_translation((50, 50, 0))
        self._meshes = {
            "A": (a.vertices, a.faces),
            "B": (b.vertices, b.faces),
            "C": (far.vertices, far.faces),
        }

    def meshes(self):
        return self._meshes

    def boxes(self):
        return {
            name: Aabb(tuple(np.min(v, axis=0)), tuple(np.max(v, axis=0)))
            for name, (v, _) in self._meshes.items()
        }


@solids_only
def test_narrow_phase_reports_intersection_volume_not_box_overlap():
    """A triage decision is made on volume; a box cannot tell a litre from a cubic metre."""
    solids = load("solids")
    engine = solids.SolidClashEngine(_Meshes(), model_id="m1")
    clashes = engine.pairs()
    assert len(clashes) == 1
    assert {clashes[0].a, clashes[0].b} == {"A", "B"}
    # The crossing region is 0.3 x 0.3 x 3.
    assert clashes[0].volume == pytest.approx(0.27, rel=1e-6)


@solids_only
def test_an_element_far_away_is_never_a_candidate():
    clashes = load("solids").SolidClashEngine(_Meshes(), model_id="m1").pairs()
    assert all("C" not in (clash.a, clash.b) for clash in clashes)


@solids_only
def test_grouping_stops_a_discipline_clashing_with_itself():
    solids = load("solids")
    groups = {"A": "left", "B": "right", "C": "left"}
    engine = solids.SolidClashEngine(_Meshes(), model_id="m1", groups=groups)
    for clash in engine.pairs():
        assert groups[clash.a] != groups[clash.b]


@solids_only
def test_the_engine_satisfies_coordinations_contract():
    solids = load("solids")
    engine = solids.SolidClashEngine(_Meshes(), model_id="m1")
    raw = engine.intersect([], [], "hard", 0.0)
    assert len(raw) == 1
    assert raw[0].a.model_id == "m1"
    assert raw[0].distance == pytest.approx(0.27, rel=1e-6)


# ---------------------------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------------------------


@crs_only
def test_a_real_crs_is_described():
    crs = load("crs")
    info = crs.describe("EPSG:27700").value
    assert info.is_projected and not info.is_geographic
    assert info.unit == "metre"


@crs_only
def test_an_invented_crs_is_refused():
    crs = load("crs")
    assert not crs.describe("EPSG:99999999").ok


@crs_only
def test_the_origin_offset_round_trips():
    """Recording the offset is what makes the local frame reversible rather than a lossy fudge."""
    crs = load("crs")
    transformer = crs.CoordinateTransformer(
        "EPSG:27700", "EPSG:4326", origin_offset=(530000, 180000, 0)
    )
    world = transformer.to_world([(0.0, 0.0, 0.0)])
    # A British National Grid easting of 530000 is central London.
    assert world[0][0] == pytest.approx(-0.128, abs=0.01)
    assert world[0][1] == pytest.approx(51.504, abs=0.01)
    # Back again, to millimetres.
    assert transformer.to_local(world)[0] == pytest.approx((0.0, 0.0, 0.0), abs=0.01)


@crs_only
def test_degrees_declared_as_metres_is_an_error():
    """111 km per unit out, and nothing else would catch it."""
    from massingviser.schema import GeoReference

    crs = load("crs")
    report = crs.validate_georeference(GeoReference(source_crs="EPSG:4326", units="m"))
    assert not report.ok
    assert "geographic" in report.errors[0]


@crs_only
def test_a_projected_crs_without_an_origin_offset_warns_about_jitter():
    from massingviser.schema import GeoReference

    crs = load("crs")
    report = crs.validate_georeference(
        GeoReference(source_crs="EPSG:27700", units="m", method="survey", vertical_datum="ODN")
    )
    assert report.ok
    assert any("jitter" in warning for warning in report.warnings)


@crs_only
def test_an_unverified_georeference_says_so_every_time():
    """`survey` and `assumed` are different facts, and the difference is never dropped."""
    from massingviser.schema import GeoReference

    crs = load("crs")
    assumed = crs.validate_georeference(
        GeoReference(
            source_crs="EPSG:27700",
            units="m",
            method="assumed",
            vertical_datum="ODN",
            origin_offset=(530000, 180000, 0),
        )
    )
    assert assumed.ok and any("provisional" in w for w in assumed.warnings)

    surveyed = crs.validate_georeference(
        GeoReference(
            source_crs="EPSG:27700",
            units="m",
            method="survey",
            vertical_datum="ODN",
            origin_offset=(530000, 180000, 0),
        )
    )
    assert surveyed.warnings == ()


# ---------------------------------------------------------------------------------------------
# Writing IFC
#
# The round trip is the test that matters: write the model, read it back with this platform's own
# reader, and check the building that comes out is the one that went in. Anything that only checks
# "a file was produced" passes for a file full of millimetre-scale rubble.
# ---------------------------------------------------------------------------------------------


@ifc_only
def test_an_ifc_guid_is_recognised_and_anything_else_is_not():
    """Identity is preserved only where it legitimately can be, and never faked where it cannot."""
    write = load("ifc_write")
    assert write.is_ifc_guid("3vB2YO$MX4xv5uCqZZG05x")
    assert not write.is_ifc_guid("mass-1:003")  # colons and hyphens are not in IFC's alphabet
    assert not write.is_ifc_guid("tooshort")


@ifc_only
def test_geometry_written_out_reads_back_at_the_same_size():
    """The whole point. Units, indices and winding all have to be right for this to hold."""
    write = load("ifc_write")
    ifc = load("ifc")
    vertices = [
        (0, 0, 0),
        (4, 0, 0),
        (4, 2, 0),
        (0, 2, 0),
        (0, 0, 3),
        (4, 0, 3),
        (4, 2, 3),
        (0, 2, 3),
    ]
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    payload, summary = write.write_ifc(
        [write.ExportElement("mass-1:000", "Block", level="L00", vertices=vertices, faces=faces)]
    )
    assert summary.elements == 1 and summary.without_geometry == ()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "out.ifc"
        path.write_bytes(payload)
        model = ifc.open_ifc(str(path), model_id="rt")

    assert len(model) == 1
    # Metres, not millimetres. `assign_unit()` with no arguments writes mm, and a reader that
    # honours the file would scale this box to 4 cm.
    assert model.source_units == "m"
    box = model.elements[0].box
    assert tuple(round(float(v), 6) for v in box.min) == (0.0, 0.0, 0.0)
    assert tuple(round(float(v), 6) for v in box.max) == (4.0, 2.0, 3.0)


@ifc_only
def test_a_platform_id_survives_as_a_property_when_it_cannot_be_a_guid():
    write = load("ifc_write")
    ifc = load("ifc")
    payload, _ = write.write_ifc(
        [
            write.ExportElement(
                "mass-1:003",
                "Storey 3",
                level="L03",
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                faces=[(0, 1, 2)],
            )
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "out.ifc"
        path.write_bytes(payload)
        model = ifc.open_ifc(str(path), model_id="rt")

    element = model.elements[0]
    assert element.global_id != "mass-1:003"  # a real GlobalId was minted
    assert any(
        value == "mass-1:003"
        for key, value in element.properties.items()
        if key.endswith(write.SOURCE_ID_PROPERTY)
    )


@ifc_only
def test_an_ifc_guid_that_arrives_is_kept():
    """A read-write round trip must not renumber the model, or every decision keyed on it is lost."""
    write = load("ifc_write")
    ifc = load("ifc")
    guid = "3vB2YO$MX4xv5uCqZZG05x"
    payload, _ = write.write_ifc(
        [
            write.ExportElement(
                guid, "Wall", vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)]
            )
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "out.ifc"
        path.write_bytes(payload)
        model = ifc.open_ifc(str(path), model_id="rt")
    assert model.elements[0].global_id == guid


@ifc_only
def test_the_spatial_tree_matches_the_one_that_went_in():
    write = load("ifc_write")
    ifc = load("ifc")
    triangle = ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    elements = [
        write.ExportElement(
            f"{mass}:{index:03d}",
            "S",
            level=f"{mass}:{index:03d}",
            building=mass,
            vertices=triangle[0],
            faces=triangle[1],
        )
        for mass in ("mass-1", "mass-2")
        for index in range(3)
    ]
    payload, summary = write.write_ifc(elements)
    assert summary.buildings == 2
    assert summary.storeys == 6

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "out.ifc"
        path.write_bytes(payload)
        model = ifc.open_ifc(str(path), model_id="rt")
    # Every element is contained, and by six distinct storeys rather than one bucket.
    storeys = {element.storey_global_id for element in model.elements}
    assert len(storeys) == 6 and None not in storeys


@ifc_only
def test_an_element_with_no_triangles_is_reported_not_dropped():
    """Arriving in the recipient's viewer as nothing is worse than being absent from the file."""
    write = load("ifc_write")
    _, summary = write.write_ifc([write.ExportElement("E1", "Empty", level="L")])
    assert summary.elements == 1
    assert summary.without_geometry == (("E1", "no triangles to write"),)


@ifc_only
async def test_the_whole_model_exports_through_the_capability():
    """End to end: massing becomes triangles, becomes payloads, becomes an IFC file, and reads back."""
    from massingviser import build_kernel
    from massingviser.plugins.interop import ExportAdapterToken
    from massingviser.plugins.massing import MASSING_COMMANDS

    kernel = build_kernel()
    await kernel.start()
    sketched = await kernel.commands.execute(
        MASSING_COMMANDS.sketch_profile,
        {"points": [(0, 0, 0), (20, 0, 0), (20, 10, 0), (0, 10, 0)], "name": "Block"},
    )
    await kernel.commands.execute(
        MASSING_COMMANDS.create_mass,
        {"name": "Block", "profile_id": sketched.value, "story_count": 4, "story_height": 3.5},
    )

    adapter = kernel.capabilities.get(ExportAdapterToken)
    assert adapter is not None and adapter.format == "ifc"
    written = await adapter.write()
    assert written.ok, getattr(written, "error", None)
    assert adapter.last_summary.elements == 4
    assert adapter.last_summary.without_geometry == ()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "massing.ifc"
        path.write_bytes(written.value)
        model = load("ifc").open_ifc(str(path), model_id="rt")

    assert len(model) == 4
    assert model.source_units == "m"
    boxes = sorted((e.box for e in model.elements if e.box), key=lambda b: b.min[2])
    # The footprint that was drawn, at the elevations that were extruded.
    assert tuple(round(float(v), 3) for v in boxes[0].max)[:2] == (20.0, 10.0)
    assert [round(float(box.min[2]), 2) for box in boxes] == [0.0, 3.5, 7.0, 10.5]
    await kernel.stop()


@ifc_only
async def test_exporting_with_nothing_to_write_says_so():
    from massingviser import build_kernel
    from massingviser.plugins.interop import ExportAdapterToken

    kernel = build_kernel()
    await kernel.start()
    result = await kernel.capabilities.get(ExportAdapterToken).write()
    assert not result.ok and "nothing to export" in result.error.message
    await kernel.stop()


# ---------------------------------------------------------------------------------------------
# Instancing
#
# The difference between sending a facade and sending a window. Tessellating in local coordinates
# and keeping the placement separate is what makes it possible; baking world coordinates in, which
# is the simpler option, throws the shared shape away.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def instanced_ifc(tmp_path_factory):
    """Twelve walls, all placements of one representation."""
    if "ifc" not in available():
        pytest.skip("ifcopenshell not installed")
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.geometry
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit

    file = ifcopenshell.file(schema="IFC4")
    project = ifcopenshell.api.root.create_entity(file, ifc_class="IfcProject", name="T")
    ifcopenshell.api.unit.assign_unit(file)
    model = ifcopenshell.api.context.add_context(file, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        file,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model,
    )
    site = ifcopenshell.api.root.create_entity(file, ifc_class="IfcSite", name="S")
    building = ifcopenshell.api.root.create_entity(file, ifc_class="IfcBuilding", name="B")
    storey = ifcopenshell.api.root.create_entity(file, ifc_class="IfcBuildingStorey", name="L0")
    for child, parent in ((site, project), (building, site), (storey, building)):
        ifcopenshell.api.aggregate.assign_object(file, products=[child], relating_object=parent)

    shared = ifcopenshell.api.geometry.add_wall_representation(
        file, context=body, length=5.0, height=3.0, thickness=0.2
    )
    for index in range(12):
        wall = ifcopenshell.api.root.create_entity(file, ifc_class="IfcWall", name=f"W{index}")
        ifcopenshell.api.geometry.assign_representation(file, product=wall, representation=shared)
        ifcopenshell.api.geometry.edit_object_placement(
            file,
            product=wall,
            matrix=np.array(
                [[1, 0, 0, index * 7.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
        )
        ifcopenshell.api.spatial.assign_container(file, products=[wall], relating_structure=storey)

    path = tmp_path_factory.mktemp("instanced") / "walls.ifc"
    file.write(str(path))
    return str(path)


@ifc_only
def test_a_placement_matrix_is_read_column_major(instanced_ifc):
    """Translation at 12-14, as everywhere else here. Transposed loads fine and is in the wrong place."""
    model = load("ifc").open_ifc(instanced_ifc, model_id="m1")
    offsets = sorted(round(element.transform[12], 3) for element in model.elements)
    assert offsets == [round(index * 7.0, 3) for index in range(12)]
    assert all(element.transform[15] == 1.0 for element in model.elements)


@ifc_only
def test_world_vertices_still_come_out_placed(instanced_ifc):
    """Nothing that reasons about space may notice that tessellation moved to local coordinates."""
    model = load("ifc").open_ifc(instanced_ifc, model_id="m1")
    boxes = sorted((e.box for e in model.elements if e.box), key=lambda box: box.min[0])
    assert [round(float(box.min[0]), 1) for box in boxes] == [index * 7.0 for index in range(12)]
    # 5 m long, so the last one ends at 77 + 5.
    assert round(float(boxes[-1].max[0]), 1) == 82.0


@ifc_only
def test_local_vertices_are_shared_and_start_at_the_origin(instanced_ifc):
    model = load("ifc").open_ifc(instanced_ifc, model_id="m1")
    assert len({element.representation_id for element in model.elements}) == 1
    for element in model.elements:
        assert float(element.local_vertices[:, 0].min()) == pytest.approx(0.0, abs=1e-9)


@ifc_only
def test_twelve_identical_walls_collapse_to_one_shape(instanced_ifc):
    ifc = load("ifc")
    source = ifc.IfcModelSource(ifc.open_ifc(instanced_ifc, model_id="m1"))
    shapes, belongs = source.instances()
    assert len(shapes) == 1
    assert len(belongs) == 12
    assert len(set(belongs.values())) == 1


@ifc_only
async def test_the_scene_sends_one_mesh_and_twelve_placements(instanced_ifc):
    """The whole point, measured: one buffer, twelve nodes, twelve distinct transforms."""
    from pathlib import Path as _Path

    from massingviser import build_kernel
    from massingviser.plugins.engine import SceneExportToken
    from massingviser.plugins.interop import INTEROP_COMMANDS

    kernel = build_kernel()
    await kernel.start()
    await kernel.commands.execute(
        INTEROP_COMMANDS.import_payload,
        {"payload": _Path(instanced_ifc).read_bytes(), "filename": "walls.ifc"},
    )
    package = (await kernel.capabilities.get(SceneExportToken).build()).value
    geometry_payloads = [p for p in package.payloads if p.role == "geometry"]
    drawable = [node for node in package.nodes if node.geometry]

    assert len(drawable) == 12
    assert sum(payload.mesh_count for payload in geometry_payloads) == 1
    # Every node points at the same buffer and the same mesh inside it.
    assert len({(n.geometry[0].payload_id, n.geometry[0].geometry_index) for n in drawable}) == 1
    # What makes them twelve walls is the placement.
    assert sorted(round(node.transform[12], 1) for node in drawable) == [
        index * 7.0 for index in range(12)
    ]
    await kernel.stop()


@ifc_only
async def test_an_instanced_model_exports_without_stacking_at_the_origin(instanced_ifc):
    """The payload holds the shared shape, so the writer has to place it."""
    from pathlib import Path as _Path

    from massingviser import build_kernel
    from massingviser.plugins.interop import INTEROP_COMMANDS, ExportAdapterToken

    kernel = build_kernel()
    await kernel.start()
    await kernel.commands.execute(
        INTEROP_COMMANDS.import_payload,
        {"payload": _Path(instanced_ifc).read_bytes(), "filename": "walls.ifc"},
    )
    written = await kernel.capabilities.get(ExportAdapterToken).write()
    assert written.ok, getattr(written, "error", None)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "out.ifc"
        path.write_bytes(written.value)
        back = load("ifc").open_ifc(str(path), model_id="rt")

    positions = sorted(round(float(e.box.min[0]), 1) for e in back.elements if e.box)
    assert positions == [index * 7.0 for index in range(12)]
    await kernel.stop()
