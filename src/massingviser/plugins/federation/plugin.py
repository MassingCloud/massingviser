from __future__ import annotations

from typing import Any, Mapping

from ...kernel import CommandDefinition, PluginContext, UIContribution
from ...schema import ModelRecord, ProjectRecord
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    FEDERATION_COMMANDS,
    FEDERATION_PERMISSIONS,
    FederationToken,
    SessionStateToken,
)
from .services import (
    FederationRuntime,
    FederationServiceImpl,
    SessionStateServiceImpl,
    create_federation_stores,
)

PLUGIN_ID = "massingviser.federation"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def _as(record_type: Any, value: Any) -> Any:
    return value if isinstance(value, record_type) else record_type(**dict(value))


def create_federation_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Project composition and per-model load state, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = FederationRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_federation_stores(context)

        federation = FederationServiceImpl(runtime, stores)
        sessions = SessionStateServiceImpl(runtime, stores, federation)

        context.capabilities.provide(FederationToken, federation, version=PLUGIN_VERSION)
        context.capabilities.provide(SessionStateToken, sessions, version=PLUGIN_VERSION)

        async def open_project(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await federation.open_project(_as(ProjectRecord, params["project"])))

        async def close_project(_params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await federation.close_project())

        async def add_model(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await federation.add_model(_as(ModelRecord, params["model"])))

        async def replace_revision(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await federation.replace_revision(
                    params["model_id"], _as(ModelRecord, params["model"])
                )
            )

        async def save_session(_params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await sessions.capture())

        async def restore_session(params: Mapping[str, Any], _ctx: Any) -> Any:
            state = stores.sessions.get(params["session_id"])
            if state is None:
                raise KeyError("No such session: " + str(params["session_id"]))
            return _unwrap(await sessions.restore(state))

        for command in (
            CommandDefinition(
                id=FEDERATION_COMMANDS.open_project,
                title="Open project",
                permission=FEDERATION_PERMISSIONS.open_project,
                handler=open_project,
            ),
            CommandDefinition(
                id=FEDERATION_COMMANDS.close_project,
                title="Close project",
                handler=close_project,
            ),
            CommandDefinition(
                id=FEDERATION_COMMANDS.add_model,
                title="Add model",
                permission=FEDERATION_PERMISSIONS.manage_models,
                handler=add_model,
            ),
            CommandDefinition(
                id=FEDERATION_COMMANDS.replace_revision,
                title="Replace revision",
                permission=FEDERATION_PERMISSIONS.manage_models,
                handler=replace_revision,
            ),
            CommandDefinition(
                id=FEDERATION_COMMANDS.save_session,
                title="Save session",
                handler=save_session,
            ),
            CommandDefinition(
                id=FEDERATION_COMMANDS.restore_session,
                title="Restore session",
                handler=restore_session,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="federation.panel", point="panel", title="Models", placement="left", order=10
            )
        )
        context.logger.info("Federation capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Federation",
        description="Project composition, per-model load state, id-preserving revision "
        "replacement and session state.",
        permissions=[FEDERATION_PERMISSIONS.manage_models, FEDERATION_PERMISSIONS.open_project],
        activate=activate,
    )


federation_plugin = create_federation_plugin()
