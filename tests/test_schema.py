"""Record families that carry behaviour, not just shape.

Most schema modules are contracts and need no tests. These three encode rules that several
capability families have to agree about, and a second copy of any of them anywhere is a place where
they can silently disagree.
"""

from __future__ import annotations

import pytest

from massingviser.schema import (
    ALL_SCHEMAS,
    CURRENT_VERSION,
    DEFAULT_TASK_IFC_RELATIONSHIP,
    METRES_PER_UNIT,
    SCHEMA,
    Extent,
    GeoReference,
    Provenance,
    RealityDerivatives,
    TaskModelLinkRecord,
    TwinObjectRecord,
    convert_length,
    extent_is_valid,
    extent_span,
    is_measurable,
    measurability_reason,
    parse_crs_code,
)

# ---------------------------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "authority", "identifier"),
    [
        ("EPSG:27700", "EPSG", "27700"),
        ("epsg:4326", "EPSG", "4326"),
        ("  EPSG:3857  ", "EPSG", "3857"),
        ("OGC:CRS84", "OGC", "CRS84"),
    ],
)
def test_authority_qualified_crs_codes_parse(code, authority, identifier):
    parsed = parse_crs_code(code)
    assert parsed is not None
    assert (parsed.authority, parsed.code) == (authority, identifier)


@pytest.mark.parametrize("code", ["27700", "", "EPSG", ":27700", "not a code"])
def test_an_unqualified_crs_code_is_not_accepted(code):
    """A bare "27700" does not say which registry it came from."""
    assert parse_crs_code(code) is None


def test_survey_feet_are_not_international_feet():
    """They differ by ~2 ppm -- metres across a large site, and a real source of disagreement."""
    assert METRES_PER_UNIT["ft"] != METRES_PER_UNIT["us-ft"]
    one_mile_ft = convert_length(5280, "ft", "m")
    one_mile_us = convert_length(5280, "us-ft", "m")
    assert abs(one_mile_us - one_mile_ft) == pytest.approx(0.0032, abs=1e-4)


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (1000, "mm", "m", 1.0),
        (1, "m", "mm", 1000.0),
        (2.5, "m", "cm", 250.0),
        (3.0, "ft", "m", 0.9144),
        (7.0, "m", "m", 7.0),
    ],
)
def test_length_conversion(value, source, target, expected):
    assert convert_length(value, source, target) == pytest.approx(expected)


def test_an_inverted_extent_is_invalid():
    assert extent_is_valid(Extent(xmin=0, ymin=0, xmax=10, ymax=10))
    assert not extent_is_valid(Extent(xmin=10, ymin=0, xmax=0, ymax=10))
    assert not extent_is_valid(Extent(xmin=0, ymin=0, xmax=10, ymax=10, zmin=5, zmax=1))


def test_extent_span_reads_the_longest_horizontal_side():
    # 40,000 mm is a building; the same number in metres would be a 40 km capture. Span is read in
    # the extent's own units so the two are not confused.
    assert extent_span(Extent(xmin=0, ymin=0, xmax=40_000, ymax=30_000)) == 40_000


def test_a_georeference_records_whether_it_was_verified():
    """`survey` and `assumed` are different facts; treating them alike is how bad data gets trusted."""
    assert GeoReference(source_crs="EPSG:27700", method="survey").verified
    assert GeoReference(source_crs="EPSG:27700", method="control-points").verified
    assert not GeoReference(source_crs="EPSG:27700", method="assumed").verified
    assert not GeoReference(source_crs="EPSG:27700", method="declared").verified
    assert not GeoReference(source_crs="EPSG:27700").verified  # unstated is not verified


def test_the_origin_offset_exists_for_float_precision():
    reference = GeoReference(
        source_crs="EPSG:27700", origin_offset=(530_000.0, 180_000.0, 0.0), method="survey"
    )
    # A 32-bit float carries ~7 significant digits; a BNG easting uses 6 before the decimal point.
    assert reference.origin_offset is not None
    assert reference.origin_offset[0] > 100_000


# ---------------------------------------------------------------------------------------------
# Measurability
# ---------------------------------------------------------------------------------------------


def _splat(**overrides) -> TwinObjectRecord:
    base = {
        "id": "t1",
        "name": "West elevation",
        "kind": "gaussian-splat",
        "created_at": "2026-01-01T00:00:00Z",
        "provenance": Provenance(source="drone"),
    }
    return TwinObjectRecord(**{**base, **overrides})


def test_a_bare_splat_is_not_measurable():
    """A radiance field renders convincingly and has no surface.

    A dimension picked off one is a plausible-looking number with no defined relationship to the
    building, so the platform refuses rather than guessing.
    """
    assert measurability_reason(_splat()) == "no-surface"
    assert not is_measurable(_splat())


def test_deriving_a_mesh_makes_a_splat_measurable():
    """A mesh *is* a surface, so the restriction lifts."""
    withMesh = _splat(derivatives=RealityDerivatives(mesh_uri="blob:mesh"))
    assert measurability_reason(withMesh) is None
    assert is_measurable(withMesh)


def test_visualization_purpose_overrides_everything():
    """Declared for looking at, so not for taking numbers off -- even with a mesh."""
    record = _splat(
        kind="mesh-scan",
        purpose="visualization",
        derivatives=RealityDerivatives(mesh_uri="blob:mesh"),
    )
    assert measurability_reason(record) == "visualization-only"


def test_an_ordinary_scan_is_measurable():
    assert is_measurable(_splat(kind="mesh-scan", purpose="analysis"))
    assert is_measurable(_splat(kind="point-cloud"))


# ---------------------------------------------------------------------------------------------
# 4D intent
# ---------------------------------------------------------------------------------------------


def test_a_task_link_defaults_to_naming_what_the_task_produces():
    """`IfcRelAssignsToProduct` and `IfcRelAssignsToProcess` are easy to transpose.

    The result still validates either way, so the meaning is stated rather than inferred.
    """
    link = TaskModelLinkRecord(id="l1", task_id="t1", behaviour="construct")
    assert link.ifc_relationship == DEFAULT_TASK_IFC_RELATIONSHIP == "IfcRelAssignsToProduct"

    consumes = TaskModelLinkRecord(
        id="l2", task_id="t1", behaviour="construct", ifc_relationship="IfcRelAssignsToProcess"
    )
    assert consumes.ifc_relationship == "IfcRelAssignsToProcess"


# ---------------------------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------------------------


def test_every_shipped_schema_is_declared_at_a_version():
    """The forward-incompatibility guard only works for schemas the migrator knows about."""
    assert len(ALL_SCHEMAS) == len(set(ALL_SCHEMAS))  # no duplicate ids
    assert set(CURRENT_VERSION) == set(ALL_SCHEMAS)
    assert all(version >= 1 for version in CURRENT_VERSION.values())


def test_schema_ids_keep_the_wire_format_prefix():
    """They name a format, not a package -- documents written by the TypeScript build open here."""
    assert SCHEMA.massing_object == "massingifc.massing.object"
    assert all(schema.startswith("massingifc.") for schema in ALL_SCHEMAS)


def test_the_default_migration_registry_declares_all_of_them():
    from massingviser.schema import create_default_migration_registry

    registry = create_default_migration_registry()
    for schema in ALL_SCHEMAS:
        assert registry.latest_version(schema) == 1
        assert registry.is_compatible(schema, 1)
        assert not registry.is_compatible(schema, 2)  # a newer build's document is refused


# ---------------------------------------------------------------------------------------------
# Record codec
# ---------------------------------------------------------------------------------------------


def test_records_round_trip_through_json():
    """Frozen dataclasses are not JSON-serialisable, and the memory adapter hides that."""
    import json

    from massingviser.schema import ProfileRecord
    from massingviser.schema.codec import record_default, record_object_hook

    original = ProfileRecord(
        id="p1",
        points=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)),
        name="Plot",
        base_elevation=2.5,
    )
    revived = json.loads(
        json.dumps(original, default=record_default), object_hook=record_object_hook
    )
    assert revived == original
    # Tuple fields must come back as tuples: JSON has one sequence type, and a silently-listified
    # field fails nothing until something checks.
    assert isinstance(revived.points, tuple)
    assert isinstance(revived.points[0], tuple)


def test_nested_records_round_trip():
    import json

    from massingviser.schema import ElementRef, QuantityRecord, QuantitySource, UnitizedValue
    from massingviser.schema.codec import record_default, record_object_hook

    original = QuantityRecord(
        id="q1",
        model_id="m1",
        metric="NetVolume",
        quantity=UnitizedValue(41472.0, "m3"),
        source=QuantitySource(kind="model-takeoff", rule_id="r1", model_version="rev-A"),
        taken_at="2026-01-01T00:00:00Z",
        elements=(ElementRef(model_id="m1", global_id="E1"),),
    )
    revived = json.loads(
        json.dumps(original, default=record_default), object_hook=record_object_hook
    )
    assert revived == original
    assert isinstance(revived.source, QuantitySource)
    assert isinstance(revived.elements[0], ElementRef)


def test_an_unknown_record_type_decodes_to_a_dict_rather_than_importing_it():
    """A project file naming its own type must never become a code-execution vector."""
    from massingviser.schema.codec import RECORD_TAG, record_object_hook

    decoded = record_object_hook({RECORD_TAG: "os.system", "arg": "rm -rf /"})
    assert decoded == {RECORD_TAG: "os.system", "arg": "rm -rf /"}


def test_a_colliding_record_type_name_is_refused():
    """Load order must not decide how a project file decodes."""
    from dataclasses import dataclass

    from massingviser.schema.codec import register_record_type

    @dataclass(frozen=True)
    class Impostor:
        id: str

    Impostor.__name__ = "ProfileRecord"
    with pytest.raises(ValueError, match="already registered"):
        register_record_type(Impostor)
