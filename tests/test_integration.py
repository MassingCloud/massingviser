"""The cross-plugin suite.

``massingifc``'s integration package exists to prove the capability families compose *through the
kernel* rather than through imports. This is the same test, against the same claim: geometry
becomes money and becomes a pinned issue, and no plugin in that chain imports another.
"""

from __future__ import annotations

import pytest

from massingviser import build_kernel
from massingviser.app import DEFAULT_PLUGINS
from massingviser.app import MASSING_MODEL_ID, MASSING_MODEL_VERSION
from massingviser.plugins.estimating import (
    ESTIMATING_COMMANDS,
    BoqToken,
    CashflowForecastToken,
    EstimateToken,
    QuantityTakeoffToken,
)
from massingviser.plugins.markup import MARKUP_COMMANDS, AnchorToken, IssueToken
from massingviser.plugins.massing import (
    MASSING_COMMANDS,
    MassingToken,
    MetricsToken,
    StoryToken,
)
from massingviser.schema import Money

FOOTPRINT = [(0, 0, 0), (40, 0, 0), (40, 24, 0), (0, 24, 0)]  # 960 m2


async def _scheme(kernel, storeys: int = 12, height: float = 3.6, color: str | None = None):
    profile = (
        await kernel.commands.execute(
            MASSING_COMMANDS.sketch_profile, {"points": FOOTPRINT, "name": "Plot"}
        )
    ).value
    mass = (
        await kernel.commands.execute(
            MASSING_COMMANDS.create_mass,
            {
                "name": "Block A",
                "profile_id": profile,
                "story_count": storeys,
                "story_height": height,
                "color": color,
            },
        )
    ).value
    return profile, mass


async def _price(kernel, rate_minor: int = 42_000, contingency: float = 7.5):
    await kernel.commands.execute(
        ESTIMATING_COMMANDS.add_rule,
        {
            "id": "rule-frame",
            "name": "Superstructure volume",
            "metric": "NetVolume",
            "unit": "m3",
            "filter": {"ifc_class": "IfcBuildingStorey"},
            "expression": "Area * Height",
        },
    )
    await kernel.commands.execute(
        ESTIMATING_COMMANDS.add_resource,
        {
            "id": "res-frame",
            "name": "Frame",
            "type": "material",
            "unit": "m3",
            "rate": Money(rate_minor, "GBP"),
        },
    )
    await kernel.commands.execute(
        ESTIMATING_COMMANDS.add_assembly,
        {
            "id": "asm-frame",
            "code": "SUPERSTRUCTURE",
            "name": "Frame allowance",
            "unit": "m3",
            "components": [{"resource_id": "res-frame", "factor": 1.0, "waste_percent": 5.0}],
            "overhead_percent": 12.0,
            "profit_percent": 6.0,
        },
    )
    await kernel.commands.execute(ESTIMATING_COMMANDS.run_takeoff, {})
    boq = (
        await kernel.commands.execute(
            ESTIMATING_COMMANDS.create_boq, {"name": "Concept bill", "currency": "GBP"}
        )
    ).value
    await kernel.commands.execute(
        ESTIMATING_COMMANDS.generate_boq,
        {"boq_id": boq.id, "assembly_by_code": {"SUPERSTRUCTURE": "asm-frame"}},
    )
    estimate = (
        await kernel.commands.execute(
            ESTIMATING_COMMANDS.create_estimate,
            {"name": "Concept estimate", "boq_id": boq.id, "contingency_percent": contingency},
        )
    ).value
    return boq, estimate


async def test_all_plugins_activate_in_one_kernel():
    kernel = build_kernel()
    report = await kernel.start()
    assert report.failed == () and report.skipped == ()
    # Structural, not enumerative. A list of ids copied into a test goes stale the moment a
    # family is added -- which it did, twice -- so this asserts the property instead: every plugin
    # the composition root registers came up, and every one of them is ours.
    assert len(report.activated) == len(DEFAULT_PLUGINS)
    assert all(plugin_id.startswith("massingviser.") for plugin_id in report.activated)
    assert all(record.status == "active" for record in kernel.plugins.list())
    await kernel.stop()


async def test_geometry_becomes_money_through_capabilities_alone():
    kernel = build_kernel()
    await kernel.start()
    _, mass = await _scheme(kernel, storeys=12, height=3.6)

    metrics = (await kernel.capabilities.get(MetricsToken).compute(mass.id)).value
    assert metrics.gross_floor_area == pytest.approx(960 * 12)

    _, estimate = await _price(kernel)

    quantity = kernel.capabilities.get(QuantityTakeoffToken).quantities()[0]
    # 960 m2 x 3.6 m x 12 storeys, measured storey by storey.
    assert quantity.quantity.value == pytest.approx(960 * 3.6 * 12)
    assert quantity.source.model_version == MASSING_MODEL_VERSION
    assert len(quantity.elements) == 12  # one per storey, not one per mass
    assert estimate.total.amount_minor > 0
    await kernel.stop()


async def test_the_bill_reconciles_with_its_own_lines():
    kernel = build_kernel()
    await kernel.start()
    await _scheme(kernel)
    boq, estimate = await _price(kernel, contingency=0.0)

    lines = kernel.capabilities.get(BoqToken).lines(boq.id)
    assert lines and all(line.total is not None for line in lines)
    assert sum(line.total.amount_minor for line in lines) == estimate.subtotal.amount_minor
    await kernel.stop()


async def test_contingency_is_applied_exactly():
    kernel = build_kernel()
    await kernel.start()
    await _scheme(kernel)
    _, estimate = await _price(kernel, contingency=10.0)
    assert estimate.total.amount_minor == estimate.subtotal.amount_minor + round(
        estimate.subtotal.amount_minor * 0.10
    )
    await kernel.stop()


async def test_cashflow_reaches_the_estimate_total_to_the_penny():
    kernel = build_kernel()
    await kernel.start()
    await _scheme(kernel)
    _, estimate = await _price(kernel)

    forecast = (
        await kernel.commands.execute(
            ESTIMATING_COMMANDS.generate_cashflow, {"estimate_id": estimate.id}
        )
    ).value
    assert forecast.periods[-1].cumulative_planned == estimate.total
    await kernel.stop()


async def test_editing_the_scheme_changes_the_price():
    kernel = build_kernel()
    await kernel.start()
    _, mass = await _scheme(kernel, storeys=10)
    _, first = await _price(kernel)

    await kernel.commands.execute(MASSING_COMMANDS.set_story_count, {"id": mass.id, "count": 20})
    _, second = await _price(kernel)

    # Doubling the storeys doubles the measured volume, so the estimate roughly doubles.
    assert second.total.amount_minor == pytest.approx(first.total.amount_minor * 2, rel=1e-6)
    await kernel.stop()


async def test_a_pin_on_a_mass_orphans_when_the_mass_is_deleted():
    kernel = build_kernel()
    await kernel.start()
    _, mass = await _scheme(kernel)

    markup = (
        await kernel.commands.execute(
            MARKUP_COMMANDS.create,
            {
                "kind": "pin",
                "model_id": MASSING_MODEL_ID,
                "element_ids": [mass.id],
                "text": "coordinate core",
            },
        )
    ).value
    await kernel.commands.execute(
        MARKUP_COMMANDS.anchor,
        {
            "markup_id": markup.id,
            "element": {"model_id": MASSING_MODEL_ID, "global_id": mass.id},
        },
    )
    alive = (
        await kernel.commands.execute(MARKUP_COMMANDS.reanchor, {"model_id": MASSING_MODEL_ID})
    ).value
    assert alive.resolved == 1 and alive.orphaned == ()

    await kernel.commands.execute(MASSING_COMMANDS.remove_mass, {"id": mass.id})
    gone = (
        await kernel.commands.execute(MARKUP_COMMANDS.reanchor, {"model_id": MASSING_MODEL_ID})
    ).value
    assert gone.orphaned == (markup.id,)
    await kernel.stop()


async def test_undo_reaches_across_the_whole_session():
    """Sketch, extrude, add floors, recolour -- then undo four times."""
    kernel = build_kernel()
    await kernel.start()
    _, mass = await _scheme(kernel, storeys=8, color="#4C78A8")
    masses = kernel.capabilities.get(MassingToken)

    await kernel.commands.execute(MASSING_COMMANDS.set_story_count, {"id": mass.id, "count": 14})
    await kernel.commands.execute(
        MASSING_COMMANDS.set_color, {"id": mass.id, "color": "#F58518"}
    )
    await kernel.commands.execute(MASSING_COMMANDS.duplicate_mass, {"id": mass.id})
    assert len(masses.list()) == 2

    await kernel.commands.undo()  # duplicate
    assert len(masses.list()) == 1
    await kernel.commands.undo()  # colour
    assert masses.get(mass.id).color == "#4C78A8"
    await kernel.commands.undo()  # story count
    assert masses.get(mass.id).story_count == 8
    await kernel.commands.undo()  # create
    assert masses.get(mass.id) is None
    await kernel.stop()


async def test_an_edit_with_nothing_to_restore_leaves_the_history_clean():
    """Setting a colour on a mass that had none records no undo step.

    Deliberate: the alternative is inventing a "previous" value and undoing to something the user
    never saw.
    """
    kernel = build_kernel()
    await kernel.start()
    _, mass = await _scheme(kernel, storeys=4)  # no colour
    depth = kernel.commands.history_size["undo"]

    await kernel.commands.execute(MASSING_COMMANDS.set_color, {"id": mass.id, "color": "#F58518"})
    assert kernel.commands.history_size["undo"] == depth

    # Once there *is* a previous colour, the inverse exists.
    await kernel.commands.execute(MASSING_COMMANDS.set_color, {"id": mass.id, "color": "#54A24B"})
    assert kernel.commands.history_size["undo"] == depth + 1
    await kernel.commands.undo()
    assert kernel.capabilities.get(MassingToken).get(mass.id).color == "#F58518"
    await kernel.stop()


async def test_state_survives_a_save_and_restore_round_trip():
    """A project's records go out through the state snapshot and come back intact."""
    from massingviser.kernel import MemoryStorageAdapter

    storage = MemoryStorageAdapter()
    kernel = build_kernel(storage=storage)
    await kernel.start()
    _, mass = await _scheme(kernel, storeys=9)
    snapshot = kernel.state.snapshot()
    await kernel.stop()

    revived = build_kernel(storage=storage)
    # Restoring *before* start is the load-order case the state store's pending map exists for.
    revived.state.restore(snapshot)
    await revived.start()

    restored = revived.capabilities.get(MassingToken).get(mass.id)
    assert restored is not None and restored.story_count == 9
    assert len(revived.capabilities.get(StoryToken).stories(mass.id)) == 9
    await revived.stop()


async def test_a_failing_plugin_does_not_take_the_others_down():
    from massingviser.sdk import define_plugin

    kernel = build_kernel()

    def explode(_context):
        raise RuntimeError("this plugin is broken")

    kernel.use(define_plugin(id="broken", version="1.0.0", activate=explode))
    report = await kernel.start()

    assert [plugin_id for plugin_id, _ in report.failed] == ["broken"]
    # Everything else still came up, and the capabilities they provide still resolve.
    assert kernel.capabilities.get(MassingToken) is not None
    assert kernel.capabilities.get(QuantityTakeoffToken) is not None
    assert kernel.plugins.status("broken") == "quarantined"
    await kernel.stop()


async def test_diagnostics_describe_the_assembled_platform():
    kernel = build_kernel()
    await kernel.start()
    diagnostics = kernel.diagnostics()

    assert diagnostics.api_version == "1.0.0"
    assert len(diagnostics.plugins) == len(DEFAULT_PLUGINS)
    assert diagnostics.commands > 50
    # Every capability family the shipped plugins provide.
    assert "massing.service" in diagnostics.capabilities
    assert "markup.anchors" in diagnostics.capabilities
    assert "estimating.takeoff" in diagnostics.capabilities
    assert "coordination.clash" in diagnostics.capabilities
    assert "planning.schedule" in diagnostics.capabilities
    assert "procurement.packages" in diagnostics.capabilities
    assert "panel" in diagnostics.ui_points
    await kernel.stop()
