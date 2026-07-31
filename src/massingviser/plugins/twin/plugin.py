from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...kernel import CommandDefinition, PluginContext, UIContribution
from ...schema import TwinObjectRecord
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    TWIN_COMMANDS,
    TWIN_PERMISSIONS,
    PointPair,
    TwinAlignmentToken,
    TwinObservationToken,
    TwinPromotionToken,
    TwinRegistryToken,
    TwinTimelineToken,
)
from .services import (
    TwinAlignmentServiceImpl,
    TwinObservationServiceImpl,
    TwinPromotionServiceImpl,
    TwinRegistryServiceImpl,
    TwinRuntime,
    TwinTimelineServiceImpl,
    create_twin_stores,
)

PLUGIN_ID = "massingviser.digital-twin"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_twin_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Captured reality, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = TwinRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_twin_stores(context)

        registry = TwinRegistryServiceImpl(runtime, stores)
        alignment = TwinAlignmentServiceImpl(runtime, stores)
        observations = TwinObservationServiceImpl(runtime, stores)
        timelines = TwinTimelineServiceImpl(runtime, stores, observations)
        promotion = TwinPromotionServiceImpl(runtime, stores)

        context.capabilities.provide(TwinRegistryToken, registry, version=PLUGIN_VERSION)
        context.capabilities.provide(TwinAlignmentToken, alignment, version=PLUGIN_VERSION)
        context.capabilities.provide(TwinObservationToken, observations, version=PLUGIN_VERSION)
        context.capabilities.provide(TwinTimelineToken, timelines, version=PLUGIN_VERSION)
        context.capabilities.provide(TwinPromotionToken, promotion, version=PLUGIN_VERSION)

        async def register(params: Mapping[str, Any], _ctx: Any) -> Any:
            record = params["record"]
            if not isinstance(record, TwinObjectRecord):
                record = TwinObjectRecord(**dict(record))
            return _unwrap(await registry.register(record))

        async def materialise(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await registry.materialise(params["twin_object_id"]))

        async def align_by_points(params: Mapping[str, Any], _ctx: Any) -> Any:
            pairs = [
                pair if isinstance(pair, PointPair) else PointPair(**dict(pair))
                for pair in params["pairs"]
            ]
            return _unwrap(
                await alignment.align_by_points(
                    params["twin_object_id"],
                    pairs,
                    allow_scale=params.get("allow_scale", False),
                )
            )

        async def set_transform(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await alignment.set_transform(params["twin_object_id"], params["transform"])
            )

        async def record_observation(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await observations.record(**dict(params)))

        async def promote(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await promotion.promote(
                    params["twin_object_id"], params["target"], **params.get("options", {})
                )
            )

        for command in (
            CommandDefinition(
                id=TWIN_COMMANDS.register,
                title="Register twin object",
                permission=TWIN_PERMISSIONS.register,
                handler=register,
            ),
            CommandDefinition(
                id=TWIN_COMMANDS.materialise, title="Materialise twin object", handler=materialise
            ),
            CommandDefinition(
                id=TWIN_COMMANDS.align_by_points,
                title="Align by control points",
                permission=TWIN_PERMISSIONS.align,
                handler=align_by_points,
            ),
            CommandDefinition(
                id=TWIN_COMMANDS.set_transform,
                title="Set twin transform",
                permission=TWIN_PERMISSIONS.align,
                handler=set_transform,
            ),
            CommandDefinition(
                id=TWIN_COMMANDS.record_observation,
                title="Record observation",
                handler=record_observation,
            ),
            CommandDefinition(
                id=TWIN_COMMANDS.promote,
                title="Promote twin object",
                permission=TWIN_PERMISSIONS.promote,
                handler=promote,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="twin.panel", point="panel", title="Reality", placement="left", order=55
            )
        )
        context.logger.info("Digital twin capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Digital twin",
        description="Registry with pluggable factories, planar Procrustes alignment, "
        "observations, timelines and gated promotion.",
        permissions=[TWIN_PERMISSIONS.register, TWIN_PERMISSIONS.align, TWIN_PERMISSIONS.promote],
        activate=activate,
    )


twin_plugin = create_twin_plugin()
