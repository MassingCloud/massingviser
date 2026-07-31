from __future__ import annotations

from typing import Any, Mapping

from ...kernel import CommandDefinition, PluginContext, UIContribution
from ...schema import ElementRef
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    PLANNING_COMMANDS,
    PLANNING_PERMISSIONS,
    PlannedActualToken,
    ScheduleImportToken,
    TaskModelLinkToken,
    TimelinePlaybackToken,
)
from .services import (
    PlannedActualComparisonServiceImpl,
    PlanningRuntime,
    ScheduleImportServiceImpl,
    TaskModelLinkServiceImpl,
    TimelinePlaybackServiceImpl,
    create_planning_stores,
)

PLUGIN_ID = "massingviser.planning-4d"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def _elements(raw: Any) -> list[ElementRef]:
    return [
        item if isinstance(item, ElementRef) else ElementRef(**dict(item)) for item in raw or ()
    ]


def create_planning_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """4D planning, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = PlanningRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_planning_stores(context)

        schedule = ScheduleImportServiceImpl(runtime, stores)
        links = TaskModelLinkServiceImpl(runtime, stores)
        playback = TimelinePlaybackServiceImpl(runtime, stores)
        progress = PlannedActualComparisonServiceImpl(runtime, stores)

        context.capabilities.provide(ScheduleImportToken, schedule, version=PLUGIN_VERSION)
        context.capabilities.provide(TaskModelLinkToken, links, version=PLUGIN_VERSION)
        context.capabilities.provide(TimelinePlaybackToken, playback, version=PLUGIN_VERSION)
        context.capabilities.provide(PlannedActualToken, progress, version=PLUGIN_VERSION)

        async def import_schedule(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await schedule.import_schedule(params["payload"], params.get("format", "csv"))
            )

        async def reimport(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await schedule.reimport(params["payload"], params.get("format", "csv")))

        async def link_selection(params: Mapping[str, Any], _ctx: Any) -> Any:
            options = (
                {"ifc_relationship": params["ifc_relationship"]}
                if params.get("ifc_relationship")
                else {}
            )
            return _unwrap(
                await links.link(
                    params["task_id"],
                    _elements(params.get("elements")),
                    params.get("behaviour", "construct"),
                    **options,
                )
            )

        async def link_by_rule(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await links.link_by_rule(
                    params["task_id"],
                    params["model_id"],
                    params.get("filter", {}),
                    params.get("behaviour", "construct"),
                )
            )

        async def reresolve(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await links.reresolve(params.get("model_id")))

        async def seek(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await playback.seek(params["at"]))

        async def compare(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await progress.compare(params["data_date"], params.get("task_ids"))
            )

        for command in (
            CommandDefinition(
                id=PLANNING_COMMANDS.import_schedule,
                title="Import programme",
                permission=PLANNING_PERMISSIONS.import_schedule,
                handler=import_schedule,
            ),
            CommandDefinition(
                id=PLANNING_COMMANDS.reimport_schedule,
                title="Re-import programme",
                permission=PLANNING_PERMISSIONS.import_schedule,
                handler=reimport,
            ),
            CommandDefinition(
                id=PLANNING_COMMANDS.link_selection,
                title="Link selection to task",
                permission=PLANNING_PERMISSIONS.edit_links,
                handler=link_selection,
            ),
            CommandDefinition(
                id=PLANNING_COMMANDS.link_by_rule,
                title="Link by rule",
                permission=PLANNING_PERMISSIONS.edit_links,
                handler=link_by_rule,
            ),
            CommandDefinition(
                id=PLANNING_COMMANDS.reresolve_links,
                title="Re-resolve links",
                permission=PLANNING_PERMISSIONS.edit_links,
                handler=reresolve,
            ),
            CommandDefinition(id=PLANNING_COMMANDS.seek, title="Seek timeline", handler=seek),
            CommandDefinition(
                id=PLANNING_COMMANDS.compare_progress,
                title="Compare progress",
                handler=compare,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="planning.panel", point="panel", title="Programme", placement="right", order=45
            )
        )
        context.logger.info("4D planning capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="4D planning",
        description="Schedule import and re-import, rule-based model links, playback, progress.",
        permissions=[PLANNING_PERMISSIONS.import_schedule, PLANNING_PERMISSIONS.edit_links],
        activate=activate,
    )


planning_plugin = create_planning_plugin()
