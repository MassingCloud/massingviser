from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...kernel import CommandDefinition, CommandInvocation, PluginContext, UIContribution
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    COORDINATION_COMMANDS,
    COORDINATION_PERMISSIONS,
    ClashToken,
    IssueRoutingToken,
    ResponsibilityToken,
    RevisionDiffToken,
    ValidationToken,
)
from .services import (
    ClashServiceImpl,
    CoordinationRuntime,
    IssueRoutingServiceImpl,
    ResponsibilityMatrixServiceImpl,
    RevisionDiffServiceImpl,
    ValidationServiceImpl,
    create_coordination_stores,
)

PLUGIN_ID = "massingviser.coordination"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_coordination_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Clash, validation, routing and revision diff, packaged as a plugin.

    Owns the *workflow* and none of the geometry. The intersection test, the model snapshots and
    the issue tracker all arrive as capabilities, so the same build runs against a viewer, an IFC
    parser, or a set of fixtures in a test.
    """
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = CoordinationRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_coordination_stores(context)

        clashes = ClashServiceImpl(runtime, stores)
        validation = ValidationServiceImpl(runtime, stores)
        matrix = ResponsibilityMatrixServiceImpl(runtime, stores)
        routing = IssueRoutingServiceImpl(runtime, stores, matrix)
        diffs = RevisionDiffServiceImpl(runtime, stores)

        context.capabilities.provide(ClashToken, clashes, version=PLUGIN_VERSION)
        context.capabilities.provide(ValidationToken, validation, version=PLUGIN_VERSION)
        context.capabilities.provide(IssueRoutingToken, routing, version=PLUGIN_VERSION)
        context.capabilities.provide(RevisionDiffToken, diffs, version=PLUGIN_VERSION)
        context.capabilities.provide(ResponsibilityToken, matrix, version=PLUGIN_VERSION)

        async def define_test(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await clashes.define_test(**dict(params)))

        async def run_test(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await clashes.run(params["test_id"]))

        async def set_status(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            clash_id = params["id"]
            existing = stores.clashes.get(clash_id)
            previous = existing.status if existing else None
            _unwrap(await clashes.set_status(clash_id, params["status"], params.get("note")))
            return {"id": clash_id, "previous": previous}

        async def promote(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await clashes.promote_to_issue(params["id"], params.get("assignee")))

        async def run_validation(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await validation.run(
                    rule_ids=params.get("rule_ids"), model_ids=params.get("model_ids")
                )
            )

        async def compare(params: Mapping[str, Any], _ctx: Any) -> Any:
            if params.get("from_version") is None:
                return _unwrap(await diffs.compare_to_previous(params["model_id"]))
            return _unwrap(
                await diffs.compare(
                    params["model_id"], params["from_version"], params["to_version"]
                )
            )

        async def route(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await routing.route(params.get("issue_ids")))

        for command in (
            CommandDefinition(
                id=COORDINATION_COMMANDS.define_clash_test,
                title="Define clash test",
                permission=COORDINATION_PERMISSIONS.manage_rules,
                handler=define_test,
            ),
            CommandDefinition(
                id=COORDINATION_COMMANDS.run_clash_test,
                title="Run clash test",
                permission=COORDINATION_PERMISSIONS.run_tests,
                handler=run_test,
            ),
            CommandDefinition(
                id=COORDINATION_COMMANDS.set_clash_status,
                title="Triage clash",
                permission=COORDINATION_PERMISSIONS.triage,
                handler=set_status,
                # Triage is the expensive human input in this workflow, so a mis-click has to be
                # reversible.
                create_inverse=lambda _params, result: (
                    None
                    if result["previous"] is None
                    else CommandInvocation(
                        COORDINATION_COMMANDS.set_clash_status,
                        {"id": result["id"], "status": result["previous"]},
                    )
                ),
            ),
            CommandDefinition(
                id=COORDINATION_COMMANDS.promote_clash_to_issue,
                title="Raise issue from clash",
                permission=COORDINATION_PERMISSIONS.triage,
                handler=promote,
            ),
            CommandDefinition(
                id=COORDINATION_COMMANDS.run_validation,
                title="Run validation",
                permission=COORDINATION_PERMISSIONS.run_tests,
                handler=run_validation,
            ),
            CommandDefinition(
                id=COORDINATION_COMMANDS.compare_revisions,
                title="Compare revisions",
                handler=compare,
            ),
            CommandDefinition(
                id=COORDINATION_COMMANDS.route_issues,
                title="Route issues",
                permission=COORDINATION_PERMISSIONS.triage,
                handler=route,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="coordination.panel",
                point="panel",
                title="Coordination",
                placement="right",
                order=35,
            )
        )
        context.ui.register(
            UIContribution(
                id="coordination.toolbar.clash",
                point="toolbar",
                title="Run clash test",
                group="review",
                order=20,
                command_id=COORDINATION_COMMANDS.run_clash_test,
            )
        )

        context.logger.info("Coordination capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Coordination",
        description="Clash with stable signatures, validation, issue routing and revision diff.",
        permissions=[
            COORDINATION_PERMISSIONS.run_tests,
            COORDINATION_PERMISSIONS.triage,
            COORDINATION_PERMISSIONS.manage_rules,
        ],
        activate=activate,
    )


coordination_plugin = create_coordination_plugin()
