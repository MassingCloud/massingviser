from __future__ import annotations

from typing import Any, Mapping

from ...kernel import CommandDefinition, PluginContext, UIContribution
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    AUTHORING_COMMANDS,
    AUTHORING_PERMISSIONS,
    AuthoringSessionToken,
    EditCommandToken,
    EditHistoryToken,
    EditOperation,
    PublishToken,
    SketchPlane,
)
from .services import (
    AuthoringRuntime,
    AuthoringSessionServiceImpl,
    EditCommandServiceImpl,
    EditHistoryServiceImpl,
    PublishServiceImpl,
    create_authoring_stores,
)

PLUGIN_ID = "massingviser.authoring"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_authoring_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Edit sessions, reversible history and conflict-checked publish, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = AuthoringRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_authoring_stores(context)

        sessions = AuthoringSessionServiceImpl(runtime, stores)
        edits = EditCommandServiceImpl(runtime, stores, sessions)
        history = EditHistoryServiceImpl(runtime, stores, sessions)
        publishing = PublishServiceImpl(runtime, stores, sessions)

        context.capabilities.provide(AuthoringSessionToken, sessions, version=PLUGIN_VERSION)
        context.capabilities.provide(EditCommandToken, edits, version=PLUGIN_VERSION)
        context.capabilities.provide(EditHistoryToken, history, version=PLUGIN_VERSION)
        context.capabilities.provide(PublishToken, publishing, version=PLUGIN_VERSION)

        async def open_session(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await sessions.open(params["model_id"]))

        async def discard_session(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await sessions.discard(params["session_id"]))

        async def apply_edit(params: Mapping[str, Any], _ctx: Any) -> Any:
            operations = [
                op if isinstance(op, EditOperation) else EditOperation(**dict(op))
                for op in params["operations"]
            ]
            return _unwrap(await edits.apply(operations))

        async def publish(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await publishing.publish(
                    params["session_id"],
                    version=params["version"],
                    force=params.get("force", False),
                )
            )

        async def undo(_params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await history.undo())

        async def redo(_params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await history.redo())

        def set_plane(params: Mapping[str, Any], _ctx: Any) -> Any:
            plane = params.get("plane")
            if not isinstance(plane, SketchPlane):
                plane = SketchPlane(**dict(plane or {}))
            sessions.set_sketch_plane(plane)
            return sessions.sketch_plane()

        for command in (
            CommandDefinition(
                id=AUTHORING_COMMANDS.open_session,
                title="Open edit session",
                permission=AUTHORING_PERMISSIONS.edit,
                handler=open_session,
            ),
            CommandDefinition(
                id=AUTHORING_COMMANDS.discard_session,
                title="Discard edit session",
                permission=AUTHORING_PERMISSIONS.edit,
                handler=discard_session,
            ),
            CommandDefinition(
                id=AUTHORING_COMMANDS.apply_edit,
                title="Apply edit",
                permission=AUTHORING_PERMISSIONS.edit,
                handler=apply_edit,
            ),
            CommandDefinition(
                id=AUTHORING_COMMANDS.publish,
                title="Publish edits",
                permission=AUTHORING_PERMISSIONS.publish,
                handler=publish,
            ),
            CommandDefinition(id=AUTHORING_COMMANDS.undo, title="Undo edit", handler=undo),
            CommandDefinition(id=AUTHORING_COMMANDS.redo, title="Redo edit", handler=redo),
            CommandDefinition(
                id=AUTHORING_COMMANDS.set_sketch_plane,
                title="Set sketch plane",
                permission=AUTHORING_PERMISSIONS.edit,
                handler=set_plane,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="authoring.panel", point="panel", title="Authoring", placement="left", order=15
            )
        )
        context.logger.info("Authoring capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Authoring",
        description="Edit sessions, sketch planes, reversible history, constraints and "
        "conflict-checked publish.",
        permissions=[AUTHORING_PERMISSIONS.edit, AUTHORING_PERMISSIONS.publish],
        activate=activate,
    )


authoring_plugin = create_authoring_plugin()
