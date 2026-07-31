"""Wave 3: family libraries, authoring, digital twin, federation.

The common thread is *identity surviving change* -- a library re-sync, a model re-issue, a
concurrent edit, a capture being registered. Each test names the thing that would otherwise be
silently lost.
"""

from __future__ import annotations

import math

import pytest

from massingviser.kernel import KERNEL_API_VERSION, err, ok
from massingviser.plugins.authoring import (
    AuthoringSessionToken,
    ConstraintRecord,
    EditCommandToken,
    EditHistoryToken,
    EditOperation,
    GeometryBackendToken,
    Level,
    PublishToken,
    SketchPlane,
    authoring_plugin,
    resolve_sketch_plane,
)
from massingviser.plugins.families import (
    FamilyLibraryRegistryToken,
    FamilyPlacementToken,
    FamilyRepositoryAdapterToken,
    FamilyResolverToken,
    FamilyVersionToken,
    PackageQuery,
    PlacementOptions,
    families_plugin,
)
from massingviser.plugins.federation import (
    FederationToken,
    ModelLoaderPortToken,
    SessionStateToken,
    federation_plugin,
)
from massingviser.plugins.twin import (
    PointPair,
    TwinAlignmentToken,
    TwinObservationToken,
    TwinPromotionToken,
    TwinRegistryToken,
    TwinTimelineToken,
    fit_planar,
    twin_plugin,
)
from massingviser.schema import (
    ElementRef,
    FamilyPackageRecord,
    FamilyParameterDefinition,
    FamilyRepositoryRecord,
    GeoReference,
    ModelRecord,
    ProjectRecord,
    Provenance,
    RealityDerivatives,
    TwinObjectRecord,
)

# ---------------------------------------------------------------------------------------------
# Family libraries
# ---------------------------------------------------------------------------------------------


def _package(slug: str, version: str, **overrides) -> FamilyPackageRecord:
    base = dict(
        id=f"{slug}@{version}",
        repository_id="repo-1",
        name=slug.split("/")[-1],
        slug=slug,
        version=version,
        api_version=f"^{KERNEL_API_VERSION}",
        license="MIT",
        parameters=(
            FamilyParameterDefinition(
                "width", "length", unit="m", default_value=1.0, min=0.1, max=5.0
            ),
            FamilyParameterDefinition(
                "finish", "enum", options=("oak", "steel"), default_value="oak"
            ),
        ),
        assets=(),
    )
    base.update(overrides)
    return FamilyPackageRecord(**base)


class _Adapter:
    kind = "local"

    def __init__(self, packages=()) -> None:
        self.packages = list(packages)
        self.connected = False

    async def connect(self, record):
        self.connected = True
        return ok(None)

    async def discover(self, query=None):
        return ok(tuple(self.packages))

    async def versions(self, slug):
        return ok(tuple(p.version for p in self.packages if p.slug == slug))

    async def fetch(self, slug, version):
        found = next((p for p in self.packages if p.slug == slug and p.version == version), None)
        return ok(found) if found else err(KeyError(slug))


REPO = FamilyRepositoryRecord(id="repo-1", name="Studio library", kind="local", uri="file:///lib")


async def test_a_repository_with_no_adapter_is_refused(harness):
    await harness.load(families_plugin)
    result = await harness.capability(FamilyLibraryRegistryToken).add_repository(REPO)
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"


async def _catalogue(harness, versions):
    await harness.load(families_plugin)
    adapter = _Adapter([_package("studio/door", version) for version in versions])
    harness.kernel.capabilities.provide(FamilyRepositoryAdapterToken, adapter)
    registry = harness.capability(FamilyLibraryRegistryToken)
    await registry.add_repository(REPO)
    await registry.sync()
    return harness.capability(FamilyResolverToken)


async def test_resolution_orders_versions_numerically_not_lexically(harness):
    """ "0.10.0" sorts below "0.9.0" as a string, which resolves a range to the wrong build."""
    resolver = await _catalogue(harness, ("0.9.0", "0.10.0"))
    assert (await resolver.resolve("studio/door", ">=0.9.0")).value.version == "0.10.0"


async def test_caret_ranges_treat_the_minor_as_breaking_below_1_0_0(harness):
    """`^0.9.0` must not match 0.10.0 -- below 1.0.0 the minor is the breaking-change axis."""
    resolver = await _catalogue(harness, ("0.9.0", "0.9.4", "0.10.0", "1.2.0", "2.0.0"))
    assert (await resolver.resolve("studio/door", "^0.9.0")).value.version == "0.9.4"
    assert (await resolver.resolve("studio/door", "^1.0.0")).value.version == "1.2.0"
    # No range at all takes the newest thing published.
    assert (await resolver.resolve("studio/door")).value.version == "2.0.0"


async def test_a_range_nothing_satisfies_says_what_is_available(harness):
    await harness.load(families_plugin)
    adapter = _Adapter([_package("studio/door", "1.0.0")])
    harness.kernel.capabilities.provide(FamilyRepositoryAdapterToken, adapter)
    registry = harness.capability(FamilyLibraryRegistryToken)
    await registry.add_repository(REPO)
    await registry.sync()

    result = await harness.capability(FamilyResolverToken).resolve("studio/door", "^9.0.0")
    assert not result.ok and result.error.code == "CAPABILITY_VERSION_MISMATCH"
    assert result.error.details["available"] == ["1.0.0"]


async def test_a_package_built_against_another_platform_is_refused(harness):
    """Content is untrusted input, so a package must be able to fail rather than be hoped into."""
    await harness.load(families_plugin)
    hostile = _package("studio/door", "1.0.0", api_version="^99.0.0")
    report = harness.capability(FamilyResolverToken).validate(hostile)
    assert not report.compatible
    assert any("built against platform API" in e for e in report.errors)


def _validation_of(harness, package):
    return harness.capability(FamilyResolverToken).validate(package)


async def test_package_validation_catches_malformed_definitions(harness):
    await harness.load(families_plugin)
    broken = _package(
        "studio/x",
        "1.0.0",
        parameters=(
            FamilyParameterDefinition("a", "enum", options=()),
            FamilyParameterDefinition("a", "number"),
            FamilyParameterDefinition("b", "number", min=10.0, max=1.0),
        ),
    )
    report = _validation_of(harness, broken)
    joined = " | ".join(report.errors)
    assert "enum" in joined and "duplicate" in joined and "min above max" in joined


async def _library(harness):
    await harness.load(families_plugin)
    adapter = _Adapter([_package("studio/door", "1.0.0"), _package("studio/door", "2.0.0")])
    harness.kernel.capabilities.provide(FamilyRepositoryAdapterToken, adapter)
    registry = harness.capability(FamilyLibraryRegistryToken)
    await registry.add_repository(REPO)
    await registry.sync()
    return registry, adapter


async def test_placement_applies_defaults_and_captures_the_slug(harness):
    registry, _ = await _library(harness)
    placement = harness.capability(FamilyPlacementToken)
    instance = (await placement.place("studio/door@1.0.0", "1.0.0", PlacementOptions())).value
    assert instance.parameters == {"width": 1.0, "finish": "oak"}
    # Package ids are catalogue-local and do not survive a re-sync; the slug is what does.
    assert instance.package_slug == "studio/door"


@pytest.mark.parametrize(
    ("parameters", "fragment"),
    [
        ({"width": 99.0}, "above its maximum"),
        ({"width": "wide"}, "must be a number"),
        ({"finish": "brass"}, "must be one of"),
        ({"widht": 1.0}, "unknown parameter"),
    ],
)
async def test_bad_parameters_are_refused_rather_than_ignored(harness, parameters, fragment):
    """A misspelled parameter that is silently dropped looks exactly like one that was applied."""
    await _library(harness)
    result = await harness.capability(FamilyPlacementToken).place(
        "studio/door@1.0.0", "1.0.0", PlacementOptions(parameters=parameters)
    )
    assert not result.ok and fragment in result.error.message


async def test_upgrading_leaves_incompatible_instances_alone_and_names_them(harness):
    await harness.load(families_plugin)
    adapter = _Adapter(
        [
            _package("studio/door", "1.0.0"),
            # v2 tightens the range, so a wide door no longer fits.
            _package(
                "studio/door",
                "2.0.0",
                parameters=(
                    FamilyParameterDefinition(
                        "width", "length", default_value=1.0, min=0.1, max=1.5
                    ),
                    FamilyParameterDefinition(
                        "finish", "enum", options=("oak", "steel"), default_value="oak"
                    ),
                ),
            ),
        ]
    )
    harness.kernel.capabilities.provide(FamilyRepositoryAdapterToken, adapter)
    registry = harness.capability(FamilyLibraryRegistryToken)
    await registry.add_repository(REPO)
    await registry.sync()

    placement = harness.capability(FamilyPlacementToken)
    narrow = (
        await placement.place(
            "studio/door@1.0.0", "1.0.0", PlacementOptions(parameters={"width": 1.0})
        )
    ).value
    wide = (
        await placement.place(
            "studio/door@1.0.0", "1.0.0", PlacementOptions(parameters={"width": 4.0})
        )
    ).value

    summary = (
        await harness.capability(FamilyVersionToken).upgrade([narrow.id, wide.id], "2.0.0")
    ).value
    assert summary.upgraded == 1
    assert [instance_id for instance_id, _ in summary.incompatible] == [wide.id]
    # The one that did not fit is still on the old version, not silently migrated.
    assert placement.instances()[1].package_version == "1.0.0"


async def test_removing_a_repository_keeps_the_instances_placed_from_it(harness):
    registry, _ = await _library(harness)
    placement = harness.capability(FamilyPlacementToken)
    await placement.place("studio/door@1.0.0", "1.0.0", PlacementOptions())

    await registry.remove_repository("repo-1")
    assert registry.search(PackageQuery()) == ()
    assert len(placement.instances()) == 1  # removing a library does not delete work done with it


# ---------------------------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------------------------


def test_a_planar_fit_recovers_a_known_rotation_exactly():
    fit = fit_planar(
        [(0, 0, 0), (10, 0, 0), (10, 10, 0)], [(100, 50, 0), (100, 60, 0), (90, 60, 0)]
    )
    assert fit is not None
    assert math.degrees(fit.rotation) == pytest.approx(90.0)
    assert fit.scale == pytest.approx(1.0)
    assert fit.rms_error == pytest.approx(0.0, abs=1e-9)
    assert fit.as_matrix()[12:15] == pytest.approx((100.0, 50.0, 0.0))


def test_a_planar_fit_recovers_scale_only_when_asked():
    without = fit_planar([(0, 0, 0), (10, 0, 0)], [(0, 0, 0), (20, 0, 0)])
    with_scale = fit_planar([(0, 0, 0), (10, 0, 0)], [(0, 0, 0), (20, 0, 0)], allow_scale=True)
    assert without.scale == 1.0 and without.rms_error > 0
    assert with_scale.scale == pytest.approx(2.0) and with_scale.rms_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("sources", "targets"),
    [
        ([(1, 1, 0)], [(2, 2, 0)]),  # one pair determines a translation and nothing else
        ([(1, 1, 0), (1, 1, 0)], [(2, 2, 0), (3, 3, 0)]),  # coincident control
        ([(0, 0, 0), (1, 0, 0)], [(0, 0, 0)]),  # mismatched lengths
    ],
)
def test_an_underdetermined_fit_returns_nothing_rather_than_an_identity(sources, targets):
    """An unregistered scan that reports itself as aligned is the failure to avoid."""
    assert fit_planar(sources, targets) is None


def _scan(**overrides) -> TwinObjectRecord:
    base = dict(
        id="scan-1",
        name="West elevation",
        kind="mesh-scan",
        created_at="2026-01-01T00:00:00Z",
        provenance=Provenance(source="drone"),
        geo_reference=GeoReference(source_crs="EPSG:27700", method="survey"),
    )
    base.update(overrides)
    return TwinObjectRecord(**base)


async def test_captured_reality_without_a_georeference_is_refused(harness):
    await harness.load(twin_plugin)
    result = await harness.capability(TwinRegistryToken).register(_scan(geo_reference=None))
    assert not result.ok and "georeference" in result.error.message


async def test_aligning_by_points_stores_its_residual(harness):
    await harness.load(twin_plugin)
    registry = harness.capability(TwinRegistryToken)
    await registry.register(_scan())
    alignment = harness.capability(TwinAlignmentToken)

    record = (
        await alignment.align_by_points(
            "scan-1",
            [
                PointPair((0, 0, 0), (100, 50, 0)),
                PointPair((10, 0, 0), (100, 60, 0)),
                PointPair((10, 10, 0), (90, 60, 0)),
            ],
        )
    ).value
    assert record.rms_error == pytest.approx(0.0, abs=1e-6)
    assert registry.get("scan-1").aligned is True
    assert registry.get("scan-1").alignment_confidence == pytest.approx(1.0)


async def test_a_hand_placed_transform_records_no_confidence(harness):
    """So a later reader can tell "somebody dragged it" from "it was registered against control"."""
    await harness.load(twin_plugin)
    await harness.capability(TwinRegistryToken).register(_scan())
    record = (await harness.capability(TwinAlignmentToken).set_transform("scan-1", (1,) * 16)).value
    assert record.rms_error is None
    assert harness.capability(TwinRegistryToken).get("scan-1").alignment_confidence is None


async def test_one_control_point_is_refused(harness):
    await harness.load(twin_plugin)
    await harness.capability(TwinRegistryToken).register(_scan())
    result = await harness.capability(TwinAlignmentToken).align_by_points(
        "scan-1", [PointPair((0, 0, 0), (1, 1, 0))]
    )
    assert not result.ok and "at least two" in result.error.message


async def test_a_bare_splat_cannot_be_promoted_to_geometry_but_can_be_catalogued(harness):
    await harness.load(twin_plugin)
    registry = harness.capability(TwinRegistryToken)
    await registry.register(_scan(id="splat", kind="gaussian-splat"))
    await harness.capability(TwinAlignmentToken).set_transform("splat", (1,) * 16)
    promotion = harness.capability(TwinPromotionToken)

    refused = await promotion.promote("splat", "authoring")
    assert not refused.ok and "no surface" in refused.error.message
    # Cataloguing claims nothing about measurement, so it stays allowed.
    assert (await promotion.promote("splat", "asset")).ok


async def test_deriving_a_mesh_unlocks_promotion(harness):
    await harness.load(twin_plugin)
    registry = harness.capability(TwinRegistryToken)
    await registry.register(
        _scan(
            id="splat",
            kind="gaussian-splat",
            derivatives=RealityDerivatives(mesh_uri="blob:mesh"),
        )
    )
    await harness.capability(TwinAlignmentToken).set_transform("splat", (1,) * 16)
    assert (await harness.capability(TwinPromotionToken).promote("splat", "authoring")).ok


async def test_an_unaligned_capture_cannot_be_promoted(harness):
    await harness.load(twin_plugin)
    await harness.capability(TwinRegistryToken).register(_scan())
    result = await harness.capability(TwinPromotionToken).promote("scan-1", "authoring")
    assert not result.ok and "not been aligned" in result.error.message


async def test_promotion_records_what_evidence_produced_the_geometry(harness):
    await harness.load(twin_plugin)
    await harness.capability(TwinRegistryToken).register(_scan())
    await harness.capability(TwinAlignmentToken).set_transform("scan-1", (1,) * 16)
    promotion = harness.capability(TwinPromotionToken)
    record = (await promotion.promote("scan-1", "authoring", target_id="wall-9")).value
    # Exactly the question people ask six months later.
    assert promotion.origin_of("wall-9").twin_object_id == "scan-1"
    assert record.target_id == "wall-9"


async def test_a_timeline_reports_the_reading_in_force_not_the_nearest(harness):
    await harness.load(twin_plugin)
    await harness.capability(TwinRegistryToken).register(_scan())
    observations = harness.capability(TwinObservationToken)
    for day, value in (("01", 1.0), ("05", 2.0), ("09", 3.0)):
        await observations.record(
            twin_object_id="scan-1",
            metric="settlement",
            value=value,
            observed_at=f"2026-03-{day}T00:00:00Z",
        )

    timelines = harness.capability(TwinTimelineToken)
    timeline = (
        await timelines.build(
            "scan-1", "settlement", "2026-03-01T00:00:00Z", "2026-03-31T00:00:00Z"
        )
    ).value
    # A reading taken after the moment asked about has not happened yet.
    assert timelines.value_at(timeline.id, "2026-03-07T00:00:00Z").value == 2.0
    assert timelines.value_at(timeline.id, "2026-02-01T00:00:00Z") is None


# ---------------------------------------------------------------------------------------------
# Federation
# ---------------------------------------------------------------------------------------------


class _Loader:
    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.fail: set[str] = set()

    async def load(self, record):
        from massingviser.kernel import KernelError

        if record.id in self.fail:
            return err(KernelError("STORAGE_FAILED", "file is corrupt"))
        self.loaded.add(record.id)
        return ok(None)

    async def unload(self, model_id):
        self.loaded.discard(model_id)
        return ok(None)

    async def set_transform(self, model_id, transform):
        return ok(None)


PROJECT = ProjectRecord(id="p1", name="Tower", created_at="2026-01-01", created_by="me")


def _model(model_id: str, version: str, **overrides) -> ModelRecord:
    base = dict(id=model_id, name=model_id, role="reference", format="ifc", version=version)
    base.update(overrides)
    return ModelRecord(**base)


async def _federated(harness):
    await harness.load(federation_plugin)
    loader = _Loader()
    harness.kernel.capabilities.provide(ModelLoaderPortToken, loader)
    federation = harness.capability(FederationToken)
    await federation.open_project(PROJECT)
    return federation, loader


async def test_a_model_cannot_be_added_without_a_project(harness):
    await harness.load(federation_plugin)
    result = await harness.capability(FederationToken).add_model(_model("m1", "C01"))
    assert not result.ok and "No project is open" in result.error.message


async def test_one_unreadable_model_does_not_stop_a_project_opening(harness):
    federation, loader = await _federated(harness)
    for index in range(3):
        await federation.add_model(_model(f"m{index}", "C01"))
    loader.fail.add("m2")

    states = {state.model_id: state for state in (await federation.load_defaults()).value}
    assert states["m0"].phase == "loaded" and states["m1"].phase == "loaded"
    # Recorded on the state, not raised.
    assert states["m2"].phase == "failed" and "corrupt" in states["m2"].error


async def test_replacing_a_revision_keeps_the_model_id(harness):
    """Everything anchored to this model references it by id."""
    federation, loader = await _federated(harness)
    await federation.add_model(_model("m0", "C02", transform=(1.0,) * 16))
    await federation.load("m0")

    incoming = _model("WHATEVER-THE-FILE-SAYS", "C03")
    replaced = (await federation.replace_revision("m0", incoming)).value

    assert replaced.id == "m0"  # the id is the project's, not the incoming file's
    assert replaced.version == "C03"
    assert replaced.transform == (1.0,) * 16  # the project's placement survives the re-issue
    assert federation.state("m0").phase == "loaded"  # and it comes back loaded
    assert "m0" in loader.loaded


async def test_replacing_with_the_same_version_is_refused(harness):
    federation, _ = await _federated(harness)
    await federation.add_model(_model("m0", "C02"))
    result = await federation.replace_revision("m0", _model("m0", "C02"))
    assert not result.ok and "already at version" in result.error.message


async def test_re_adding_an_existing_model_points_at_replace_revision(harness):
    federation, _ = await _federated(harness)
    await federation.add_model(_model("m0", "C02"))
    result = await federation.add_model(_model("m0", "C03"))
    assert not result.ok and "replace_revision" in result.error.message


async def test_a_session_restores_which_models_were_loaded(harness):
    federation, _ = await _federated(harness)
    for index in range(3):
        await federation.add_model(_model(f"m{index}", "C01"))
    await federation.load("m0")
    await federation.load("m2")

    sessions = harness.capability(SessionStateToken)
    snapshot = (await sessions.capture()).value
    assert set(snapshot.loaded_model_ids) == {"m0", "m2"}

    await federation.unload("m0")
    await federation.load("m1")
    await sessions.restore(snapshot)
    # Reopening must not mean reloading twelve models and re-hiding nine of them.
    assert {s.model_id for s in federation.states() if s.phase == "loaded"} == {"m0", "m2"}


# ---------------------------------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------------------------------


class _Backend:
    def __init__(self) -> None:
        self.version = "C02"
        self.applied: list[EditOperation] = []
        self.externally_changed: set[str] = set()
        self.constraint_ok = True
        self.published: list[tuple[str, str]] = []

    async def apply(self, operations):
        self.applied.extend(operations)
        return ok(
            [
                op.element or ElementRef("m1", f"NEW-{len(self.applied)}-{index}")
                for index, op in enumerate(operations)
            ]
        )

    async def revert(self, operations):
        for op in operations:
            if op in self.applied:
                self.applied.remove(op)
        return ok(None)

    def current_version(self, model_id):
        return self.version if model_id == "m1" else None

    def changed_since(self, element, since_version):
        return element.global_id in self.externally_changed

    async def publish(self, model_id, version):
        self.published.append((model_id, version))
        return ok(None)

    def evaluate_constraint(self, constraint):
        return self.constraint_ok


async def _authoring(harness):
    await harness.load(authoring_plugin)
    backend = _Backend()
    harness.kernel.capabilities.provide(GeometryBackendToken, backend)
    return backend


def test_a_hosted_sketch_plane_resolves_to_its_level_plus_the_offset():
    plane = SketchPlane(level_id="L2", offset=0.9)
    resolved = resolve_sketch_plane(plane, [Level("L1", "Ground", 0.0), Level("L2", "First", 4.2)])
    assert resolved.origin[2] == pytest.approx(5.1)
    # The offset is kept, so the plane still tracks its level when the level moves.
    assert resolved.offset == 0.9


async def test_an_edit_outside_a_session_is_refused(harness):
    await _authoring(harness)
    result = await harness.capability(EditCommandToken).apply(
        [EditOperation(kind="create", ifc_class="IfcWall")]
    )
    assert not result.ok and "No authoring session" in result.error.message


async def test_a_second_session_with_unpublished_edits_is_refused(harness):
    await _authoring(harness)
    sessions = harness.capability(AuthoringSessionToken)
    await sessions.open("m1")
    await harness.capability(EditCommandToken).apply([EditOperation(kind="create")])
    result = await sessions.open("m1")
    assert not result.ok and "still open with unpublished edits" in result.error.message


async def test_a_broken_constraint_rolls_the_edit_back(harness):
    backend = await _authoring(harness)
    await harness.capability(AuthoringSessionToken).open("m1")
    edits = harness.capability(EditCommandToken)
    edits.add_constraint(
        ConstraintRecord(id="c1", kind="level", description="walls must sit on a level")
    )
    backend.constraint_ok = False

    result = await edits.apply([EditOperation(kind="create", ifc_class="IfcWall")])
    assert not result.ok and "rolled back" in result.error.message
    assert backend.applied == []  # actually reverted, not merely reported


async def test_undo_and_redo_go_through_the_backend(harness):
    backend = await _authoring(harness)
    await harness.capability(AuthoringSessionToken).open("m1")
    edits = harness.capability(EditCommandToken)
    history = harness.capability(EditHistoryToken)

    await edits.apply([EditOperation(kind="create", ifc_class="IfcWall")])
    assert len(backend.applied) == 1 and history.can_undo()

    await history.undo()
    assert backend.applied == [] and history.can_redo()

    await history.redo()
    assert len(backend.applied) == 1


async def test_coalescing_makes_a_drag_undo_once(harness):
    await _authoring(harness)
    await harness.capability(AuthoringSessionToken).open("m1")
    edits = harness.capability(EditCommandToken)
    history = harness.capability(EditHistoryToken)

    for _ in range(4):
        await edits.apply([EditOperation(kind="move", element=ElementRef("m1", "W1"))])
    assert len(history.entries()) == 4

    merged = (await history.coalesce("drag wall", [entry.id for entry in history.entries()])).value
    assert len(history.entries()) == 1 and len(merged.operations) == 4


async def test_discarding_a_session_reverts_its_geometry(harness):
    backend = await _authoring(harness)
    sessions = harness.capability(AuthoringSessionToken)
    session = (await sessions.open("m1")).value
    await harness.capability(EditCommandToken).apply([EditOperation(kind="create")])
    assert len(backend.applied) == 1

    await sessions.discard(session.id)
    # Not merely forgotten -- otherwise "discard" leaves the geometry exactly as it was.
    assert backend.applied == []


async def test_publishing_over_a_concurrent_change_is_refused(harness):
    backend = await _authoring(harness)
    sessions = harness.capability(AuthoringSessionToken)
    session = (await sessions.open("m1")).value
    await harness.capability(EditCommandToken).apply(
        [EditOperation(kind="modify", element=ElementRef("m1", "W1"))]
    )

    backend.externally_changed.add("W1")  # somebody else got there first
    publishing = harness.capability(PublishToken)

    preview = (await publishing.preview(session.id)).value
    assert [e.global_id for e in preview.conflicts] == ["W1"]

    refused = await publishing.publish(session.id, version="C03")
    assert not refused.ok and "changed since this session opened" in refused.error.message
    assert backend.published == []

    # Force is a deliberate act with a name, not the default.
    forced = await publishing.publish(session.id, version="C03", force=True)
    assert forced.ok and backend.published == [("m1", "C03")]


async def test_publishing_twice_is_refused(harness):
    await _authoring(harness)
    sessions = harness.capability(AuthoringSessionToken)
    session = (await sessions.open("m1")).value
    await harness.capability(EditCommandToken).apply([EditOperation(kind="create")])
    publishing = harness.capability(PublishToken)

    assert (await publishing.publish(session.id, version="C03")).ok
    again = await publishing.publish(session.id, version="C04")
    assert not again.ok and "already been published" in again.error.message


async def test_publishing_a_session_that_changed_nothing_is_refused(harness):
    await _authoring(harness)
    sessions = harness.capability(AuthoringSessionToken)
    session = (await sessions.open("m1")).value
    result = await harness.capability(PublishToken).publish(session.id, version="C03")
    assert not result.ok and "changed nothing" in result.error.message
