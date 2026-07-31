from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...kernel import CommandDefinition, PluginContext, UIContribution
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    ESTIMATING_COMMANDS,
    ESTIMATING_PERMISSIONS,
    BoqToken,
    CashflowForecastToken,
    ClassificationMappingToken,
    CostAssemblyToken,
    EstimateToken,
    QuantityTakeoffToken,
)
from .services import (
    BoqServiceImpl,
    CashflowForecastServiceImpl,
    ClassificationMappingServiceImpl,
    CostAssemblyServiceImpl,
    EstimateServiceImpl,
    EstimatingRuntime,
    QuantityTakeoffServiceImpl,
    create_estimating_stores,
)

PLUGIN_ID = "massingviser.estimating-5d"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_estimating_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """5D estimating, packaged as a plugin.

    Depends on no other plugin. It reaches for a model element source and a schedule basis through
    *capabilities*, and reports honestly when neither is installed -- which is what lets the same
    build run headless in a test, against a massing model, or against a federated IFC project.
    """
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = EstimatingRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_estimating_stores(context)

        takeoff = QuantityTakeoffServiceImpl(runtime, stores)
        classification = ClassificationMappingServiceImpl(runtime, stores)
        assemblies = CostAssemblyServiceImpl(runtime, stores)
        boqs = BoqServiceImpl(runtime, stores, assemblies)
        estimates = EstimateServiceImpl(runtime, stores, boqs)
        cashflow = CashflowForecastServiceImpl(runtime, stores)

        context.capabilities.provide(QuantityTakeoffToken, takeoff, version=PLUGIN_VERSION)
        context.capabilities.provide(
            ClassificationMappingToken, classification, version=PLUGIN_VERSION
        )
        context.capabilities.provide(CostAssemblyToken, assemblies, version=PLUGIN_VERSION)
        context.capabilities.provide(BoqToken, boqs, version=PLUGIN_VERSION)
        context.capabilities.provide(EstimateToken, estimates, version=PLUGIN_VERSION)
        context.capabilities.provide(CashflowForecastToken, cashflow, version=PLUGIN_VERSION)

        async def add_rule(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await takeoff.add_rule(**dict(params)))

        async def run_takeoff(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await takeoff.run(
                    model_ids=params.get("model_ids"), rule_ids=params.get("rule_ids")
                )
            )

        async def add_resource(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await assemblies.upsert_resource(**dict(params)))

        async def add_assembly(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await assemblies.upsert_assembly(**dict(params)))

        async def create_boq(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await boqs.create(params["name"], params["currency"]))

        async def generate_boq(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await boqs.generate(
                    params["boq_id"], assembly_by_code=params.get("assembly_by_code", {})
                )
            )

        async def create_estimate(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await estimates.create(
                    params["name"],
                    params["boq_id"],
                    contingency_percent=params.get("contingency_percent", 0.0),
                )
            )

        async def recalculate(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await estimates.recalculate(params["estimate_id"]))

        async def issue(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await estimates.issue(params["estimate_id"]))

        async def generate_cashflow(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await cashflow.generate(params["estimate_id"], unit=params.get("unit", "month"))
            )

        for command in (
            CommandDefinition(
                id=ESTIMATING_COMMANDS.add_rule,
                title="Add takeoff rule",
                permission=ESTIMATING_PERMISSIONS.measure,
                handler=add_rule,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.run_takeoff,
                title="Run takeoff",
                permission=ESTIMATING_PERMISSIONS.measure,
                handler=run_takeoff,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.add_resource,
                title="Add resource",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=add_resource,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.add_assembly,
                title="Add assembly",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=add_assembly,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.create_boq,
                title="Create bill",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=create_boq,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.generate_boq,
                title="Generate bill",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=generate_boq,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.create_estimate,
                title="Create estimate",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=create_estimate,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.recalculate_estimate,
                title="Recalculate estimate",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=recalculate,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.issue_estimate,
                title="Issue estimate",
                permission=ESTIMATING_PERMISSIONS.issue,
                handler=issue,
            ),
            CommandDefinition(
                id=ESTIMATING_COMMANDS.generate_cashflow,
                title="Generate cashflow",
                permission=ESTIMATING_PERMISSIONS.price,
                handler=generate_cashflow,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="estimating.panel",
                point="panel",
                title="Cost",
                placement="right",
                order=40,
            )
        )
        context.ui.register(
            UIContribution(
                id="estimating.toolbar.takeoff",
                point="toolbar",
                title="Run takeoff",
                group="cost",
                order=20,
                command_id=ESTIMATING_COMMANDS.run_takeoff,
            )
        )

        context.logger.info("5D estimating capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="5D estimating",
        description="Takeoff, classification, composite rates, bills, estimates and cashflow.",
        permissions=[
            ESTIMATING_PERMISSIONS.measure,
            ESTIMATING_PERMISSIONS.price,
            ESTIMATING_PERMISSIONS.issue,
        ],
        activate=activate,
    )


estimating_plugin = create_estimating_plugin()
