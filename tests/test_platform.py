"""Wave 4: interop, analytics, the shell's bookkeeping, and the engine bridge.

These four are the platform's edges -- where data arrives, where it is summarised, where it is
presented, and where it leaves for a game engine. Each test names the thing that would otherwise be
asserted on trust.
"""

from __future__ import annotations

import json

import pytest

from massingviser.kernel import err, ok
from massingviser.plugins.analytics import (
    ANALYTICS_COMMANDS,
    AnalyticsToken,
    MetricProviderToken,
    MetricValue,
    analytics_plugin,
    linear_forecast,
)
from massingviser.plugins.engine import (
    PayloadRef,
    RealityLayer,
    SceneNode,
    SceneNodeSourceToken,
    SceneRelationship,
    build_scene_package,
    create_scene_query,
    engine_plugin,
    to_manifest,
    validate_scene_package,
)
from massingviser.plugins.engine import SceneExportToken
from massingviser.plugins.interop import (
    ConnectorPolicy,
    ImportAdapterToken,
    ImportSummary,
    InteropToken,
    interop_plugin,
)
from massingviser.plugins.shell import ShellToken, StatusItem, shell_plugin


# ---------------------------------------------------------------------------------------------
# Interop
# ---------------------------------------------------------------------------------------------


class _IfcAdapter:
    format = "ifc"
    signatures = (b"ISO-10303-21;",)
    extensions = ("ifc",)

    async def read(self, payload):
        return ok(ImportSummary(format="ifc", records=payload.count(b"\n")))


class _ZipAdapter:
    format = "zip"
    signatures = (b"PK\x03\x04",)
    extensions = ("zip", "ifczip")

    async def read(self, payload):
        return ok(ImportSummary(format="zip", records=1))


async def _interop(harness):
    await harness.load(interop_plugin)
    harness.kernel.capabilities.provide(ImportAdapterToken, _IfcAdapter())
    harness.kernel.capabilities.provide(ImportAdapterToken, _ZipAdapter())
    return harness.capability(InteropToken)


async def test_detection_reads_the_bytes_not_the_extension(harness):
    service = await _interop(harness)
    detection = service.detect(b"ISO-10303-21;\nDATA;\n", "model.ifc")
    assert detection.format == "ifc" and detection.confidence == "certain"


async def test_a_mislabelled_file_is_detected_and_the_disagreement_surfaces(harness):
    """A `.ifc` that is really a zip, handed to an IFC parser, fails three layers down."""
    service = await _interop(harness)
    detection = service.detect(b"PK\x03\x04rest-of-the-zip", "model.ifc")
    assert detection.format == "zip"
    assert detection.claimed_format == "ifc"
    assert detection.disputed

    summary = (await service.import_payload(b"PK\x03\x04x", filename="model.ifc")).value
    assert summary.format == "zip"
    # The disagreement travels with the import rather than being resolved silently.
    assert any("claimed" in warning for warning in summary.warnings)


async def test_an_unrecognised_extension_is_reported_as_claimed_not_detected(harness):
    service = await _interop(harness)
    detection = service.detect(b"nothing recognisable here", "model.ifc")
    assert detection.confidence == "claimed"
    assert "the extension is the only evidence" in detection.detail


async def test_unknown_content_with_no_hint_is_refused(harness):
    service = await _interop(harness)
    result = await service.import_payload(b"???", filename=None)
    assert not result.ok and "No installed adapter recognises" in result.error.message


async def test_an_unknown_format_defaults_to_review_not_trusted(harness):
    """A governance default that permits is not governance."""
    service = await _interop(harness)
    assert service.policy("dwg").trust == "review"


async def test_a_blocked_format_is_refused_with_its_reason(harness):
    service = await _interop(harness)
    service.set_policy(
        ConnectorPolicy(format="zip", trust="blocked", reason="unvetted supplier archives")
    )
    result = await service.import_payload(b"PK\x03\x04x", filename="a.zip")
    assert not result.ok and result.error.code == "PERMISSION_DENIED"
    assert "unvetted supplier archives" in result.error.message


# ---------------------------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------------------------


class _Provider:
    namespace = "cost"

    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def collect(self):
        return (MetricValue("total", self.value, unit="GBP"),)


class _Broken:
    namespace = "broken"

    def collect(self):
        raise RuntimeError("provider exploded")


async def test_metrics_are_namespaced_on_collection(harness):
    """Two families can both publish "total" without silently overwriting each other."""
    await harness.load(analytics_plugin)
    harness.kernel.capabilities.provide(MetricProviderToken, _Provider())
    snapshot = (await harness.capability(AnalyticsToken).capture()).value
    assert [metric.key for metric in snapshot.values] == ["cost.total"]


async def test_one_broken_provider_does_not_blank_the_dashboard(harness):
    await harness.load(analytics_plugin)
    harness.kernel.capabilities.provide(MetricProviderToken, _Provider())
    harness.kernel.capabilities.provide(MetricProviderToken, _Broken())

    snapshot = (await harness.capability(AnalyticsToken).capture()).value
    assert len(snapshot.values) == 1
    assert [namespace for namespace, _ in snapshot.failed] == ["broken"]


async def test_history_accumulates_and_a_report_groups_by_namespace(harness):
    await harness.load(analytics_plugin)
    provider = _Provider()
    harness.kernel.capabilities.provide(MetricProviderToken, provider)
    analytics = harness.capability(AnalyticsToken)

    for value in (100.0, 110.0, 120.0):
        provider.value = value
        await analytics.capture()

    assert [sample.value for sample in analytics.history("cost.total")] == [100.0, 110.0, 120.0]
    report = (await analytics.report("Monthly")).value
    assert [section.title for section in report.sections] == ["cost"]


async def test_a_forecast_needs_at_least_two_samples(harness):
    await harness.load(analytics_plugin)
    harness.kernel.capabilities.provide(MetricProviderToken, _Provider())
    analytics = harness.capability(AnalyticsToken)
    await analytics.capture()
    result = await analytics.forecast("cost.total")
    assert not result.ok and "at least two" in result.error.message


async def test_a_forecast_carries_bounds_that_widen_with_distance(harness):
    """A single projected number invites a decision it cannot support."""
    await harness.load(analytics_plugin)
    provider = _Provider()
    harness.kernel.capabilities.provide(MetricProviderToken, provider)
    analytics = harness.capability(AnalyticsToken)
    for value in (100.0, 112.0, 119.0, 134.0, 141.0):
        provider.value = value
        await analytics.capture()

    forecast = (await analytics.forecast("cost.total", horizon=3)).value
    assert len(forecast.values) == len(forecast.lower) == len(forecast.upper) == 3
    assert all(lo < v < hi for lo, v, hi in zip(forecast.lower, forecast.values, forecast.upper))
    widths = [hi - lo for lo, hi in zip(forecast.lower, forecast.upper)]
    assert widths[0] < widths[-1]  # further out is less certain, and says so
    assert forecast.basis == 5


def test_a_perfectly_linear_history_says_its_interval_understates_uncertainty():
    values, lower, upper, caveat = linear_forecast([10.0, 12.0, 14.0, 16.0], 2, 0.95)
    assert values == pytest.approx([18.0, 20.0])
    assert "understates" in caveat


# ---------------------------------------------------------------------------------------------
# Shell bookkeeping
# ---------------------------------------------------------------------------------------------


async def test_the_shell_mirrors_registered_panels_without_opening_them(harness):
    from massingviser.plugins.massing import massing_plugin

    await harness.load(massing_plugin, shell_plugin)
    shell = harness.capability(ShellToken)
    panels = shell.sync_panels()
    assert [panel.id for panel in panels] == ["massing.panel"]
    # The registry says which panels exist; the shell says which are open.
    assert panels[0].open is False


async def test_toggling_a_panel_is_idempotent_per_call_and_reported(harness):
    from massingviser.plugins.massing import massing_plugin

    await harness.load(massing_plugin, shell_plugin)
    shell = harness.capability(ShellToken)
    shell.sync_panels()
    assert shell.toggle_panel("massing.panel").value.open is True
    assert shell.toggle_panel("massing.panel").value.open is False
    assert not shell.toggle_panel("nope").ok


async def test_an_error_notification_does_not_expire_by_default(harness):
    """An error with a timeout is an error nobody reads."""
    await harness.load(shell_plugin)
    shell = harness.capability(ShellToken)
    assert shell.notify("saved", severity="success").ttl == 6.0
    assert shell.notify("could not save", severity="error").ttl is None


async def test_dismissed_notifications_leave_the_default_listing(harness):
    await harness.load(shell_plugin)
    shell = harness.capability(ShellToken)
    note = shell.notify("hello")
    assert len(shell.notifications()) == 1
    shell.dismiss(note.id)
    assert shell.notifications() == ()
    assert len(shell.notifications(include_dismissed=True)) == 1


async def test_indeterminate_progress_is_distinguishable_from_zero(harness):
    """A spinner and a bar say different things."""
    await harness.load(shell_plugin)
    shell = harness.capability(ShellToken)
    spinner = shell.begin("Converting")
    assert spinner.fraction is None
    bar = shell.begin("Exporting", 0.0)
    assert bar.fraction == 0.0

    # Out-of-range progress is clamped, not fatal -- a rounding error must not fail the job.
    assert shell.report(bar.id, 1.02).value.fraction == 1.0
    assert len(shell.running()) == 2
    shell.finish(spinner.id)
    assert [task.id for task in shell.running()] == [bar.id]


async def test_the_palette_is_built_from_the_command_bus(harness):
    from massingviser.plugins.massing import massing_plugin

    await harness.load(massing_plugin, shell_plugin)
    shell = harness.capability(ShellToken)
    entries = shell.palette("story")
    assert entries and all("story" in e.title.lower() or "story" in e.command_id for e in entries)
    # Permissions travel, so a host does not offer an action that will be refused.
    assert any(entry.permission == "massing.edit" for entry in shell.palette())


async def test_status_items_replace_by_id(harness):
    await harness.load(shell_plugin)
    shell = harness.capability(ShellToken)
    shell.set_status(StatusItem(id="sync", text="Syncing"))
    shell.set_status(StatusItem(id="sync", text="Synced", severity="success"))
    assert len(shell.status()) == 1 and shell.status()[0].text == "Synced"


# ---------------------------------------------------------------------------------------------
# Engine bridge
# ---------------------------------------------------------------------------------------------


def _nodes():
    return [
        SceneNode(
            global_id="WALL-1",
            ifc_class="IfcWall",
            level_global_id="L1",
            parent_global_id="STOREY-1",
            property_sets={"Pset_WallCommon": {"FireRating": "60"}},
            relationships=(SceneRelationship("HostedBy", "STOREY-1"),),
            payload_id="geometry-0",
            transient_local_id=4711,
        ),
        SceneNode(global_id="WALL-2", ifc_class="IfcWall", level_global_id="L1"),
        SceneNode(global_id="SLAB-1", ifc_class="IfcSlab", level_global_id="L2"),
    ]


PAYLOAD = PayloadRef(
    id="geometry-0", role="geometry", path="payloads/geometry-0.glb",
    encoding="model/gltf-binary", byte_length=1024,
)


def test_duplicate_global_ids_are_refused():
    """The second would displace the first in the index, and the failure would surface in C++."""
    result = build_scene_package(
        generator="t",
        generated_at="2026-01-01",
        source_units="m",
        nodes=[SceneNode("W1", "IfcWall"), SceneNode("W1", "IfcSlab")],
    )
    assert not result.ok and "Duplicate GlobalId" in result.error.message


def test_a_node_referencing_an_undeclared_payload_is_refused():
    result = build_scene_package(
        generator="t",
        generated_at="2026-01-01",
        source_units="m",
        nodes=[SceneNode("W1", "IfcWall", payload_id="missing")],
    )
    assert not result.ok and "not declared" in result.error.message


def test_indexes_are_precomputed_by_the_exporter():
    package = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="m",
        nodes=_nodes(), payloads=[PAYLOAD],
    ).value
    assert package.indexes.by_class["IfcWall"] == (0, 1)
    assert package.indexes.by_level["L2"] == (2,)
    assert package.indexes.by_global_id["SLAB-1"] == 2


def test_semantics_travel_rather_than_being_flattened():
    package = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="m",
        nodes=_nodes(), payloads=[PAYLOAD],
    ).value
    query = create_scene_query(package)
    wall = query.by_global_id("WALL-1")
    # Property sets stay nested, relationships stay typed edges.
    assert wall.property_sets["Pset_WallCommon"]["FireRating"] == "60"
    assert wall.relationships[0].type == "HostedBy"
    assert [n.global_id for n in query.by_class("IfcWall")] == ["WALL-1", "WALL-2"]
    assert [n.global_id for n in query.by_level("L1")] == ["WALL-1", "WALL-2"]


def test_the_transient_local_id_never_reaches_the_index():
    package = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="m",
        nodes=_nodes(), payloads=[PAYLOAD],
    ).value
    assert 4711 not in package.indexes.by_global_id
    assert all(isinstance(key, str) for key in package.indexes.by_global_id)


def test_validation_reports_a_missing_payload_and_a_semantic_only_package():
    package = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="m",
        nodes=_nodes(), payloads=[PAYLOAD],
    ).value

    report = validate_scene_package(package, available_payloads=[])
    assert not report.ok and "not in the archive" in report.errors[0]

    semantic_only = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="m",
        nodes=[SceneNode("W1", "IfcWall")],
    ).value
    report = validate_scene_package(semantic_only)
    assert report.ok  # legitimate -- the viewer contracts hand out no mesh buffers
    assert any("semantic half only" in warning for warning in report.warnings)


def test_a_reality_layer_carries_its_measurable_flag():
    """So a splat arrives marked as something to render, not something to dimension."""
    package = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="m",
        nodes=[SceneNode("W1", "IfcWall")],
        reality_layers=[RealityLayer(id="scan", name="West elevation", measurable=False)],
    ).value
    assert package.reality_layers[0].measurable is False
    report = validate_scene_package(package)
    assert report.ok


def test_the_manifest_is_camel_case_so_one_importer_reads_both_implementations():
    package = build_scene_package(
        generator="t", generated_at="2026-01-01", source_units="mm",
        nodes=_nodes(), payloads=[PAYLOAD],
    ).value
    manifest = json.loads(json.dumps(to_manifest(package)))
    assert manifest["sourceUnits"] == "mm"
    assert manifest["nodes"][0]["globalId"] == "WALL-1"
    assert manifest["nodes"][0]["levelGlobalId"] == "L1"
    assert manifest["indexes"]["byClass"]["IfcWall"] == [0, 1]
    # The transient handle is not in the manifest at all.
    assert "transientLocalId" not in manifest["nodes"][0]


class _Source:
    def __init__(self, units="m", crs=None) -> None:
        self._units = units
        self._crs = crs

    def nodes(self):
        return _nodes()

    def payloads(self):
        return [PAYLOAD]

    def reality_layers(self):
        return ()

    def source_units(self):
        return self._units

    def crs(self):
        return self._crs


async def test_sources_disagreeing_about_units_are_refused(harness):
    """Guessing which source is right hides a problem that has to be fixed upstream."""
    await harness.load(engine_plugin)
    harness.kernel.capabilities.provide(SceneNodeSourceToken, _Source(units="m"))
    harness.kernel.capabilities.provide(SceneNodeSourceToken, _Source(units="mm"))
    result = await harness.capability(SceneExportToken).build()
    assert not result.ok and "disagree about units" in result.error.message


async def test_sources_declaring_different_crs_are_refused(harness):
    await harness.load(engine_plugin)
    harness.kernel.capabilities.provide(SceneNodeSourceToken, _Source(crs="EPSG:27700"))
    harness.kernel.capabilities.provide(SceneNodeSourceToken, _Source(crs="EPSG:4326"))
    result = await harness.capability(SceneExportToken).build()
    assert not result.ok and "different CRSs" in result.error.message


async def test_exporting_with_no_source_says_so(harness):
    await harness.load(engine_plugin)
    result = await harness.capability(SceneExportToken).build()
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"
