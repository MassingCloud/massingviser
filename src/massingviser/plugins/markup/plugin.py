from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...kernel import CommandDefinition, CommandInvocation, PluginContext, UIContribution
from ...schema import ElementRef, Id, MarkupRecord
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    MARKUP_COMMANDS,
    MARKUP_PERMISSIONS,
    AnchorToken,
    CommentToken,
    IssueToken,
    MarkupToken,
    ReviewToken,
)
from .services import (
    AnchorServiceImpl,
    CommentServiceImpl,
    IssueServiceImpl,
    MarkupRuntime,
    MarkupServiceImpl,
    ReviewServiceImpl,
    create_markup_stores,
)

PLUGIN_ID = "massingviser.markup"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_markup_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Markup, issues and review, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = MarkupRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_markup_stores(context)

        markups = MarkupServiceImpl(runtime, stores)
        anchors = AnchorServiceImpl(runtime, stores)
        issues = IssueServiceImpl(runtime, stores)
        comments = CommentServiceImpl(runtime, stores)
        review = ReviewServiceImpl(runtime, stores)

        context.capabilities.provide(MarkupToken, markups, version=PLUGIN_VERSION)
        context.capabilities.provide(AnchorToken, anchors, version=PLUGIN_VERSION)
        context.capabilities.provide(IssueToken, issues, version=PLUGIN_VERSION)
        context.capabilities.provide(CommentToken, comments, version=PLUGIN_VERSION)
        context.capabilities.provide(ReviewToken, review, version=PLUGIN_VERSION)

        restore_buffer: dict[Id, MarkupRecord] = {}

        async def create_markup(params: Mapping[str, Any], _ctx: Any) -> MarkupRecord:
            return _unwrap(await markups.create(**dict(params)))

        async def remove_markup(params: Mapping[str, Any], _ctx: Any) -> MarkupRecord | None:
            markup_id = params["id"]
            snapshot = markups.get(markup_id)
            _unwrap(await markups.remove(markup_id))
            if snapshot is not None:
                restore_buffer[markup_id] = snapshot
            return snapshot

        def restore_markup(params: Mapping[str, Any], _ctx: Any) -> None:
            buffered = restore_buffer.get(params["id"])
            if buffered is not None:
                markups.restore(buffered)

        async def anchor_markup(params: Mapping[str, Any], _ctx: Any) -> Any:
            element = params.get("element")
            if isinstance(element, Mapping):
                element = ElementRef(**dict(element))
            return _unwrap(
                await anchors.anchor(
                    params["markup_id"],
                    element=element,
                    world_position=params.get("world_position"),
                )
            )

        async def reanchor(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await anchors.reanchor(params["model_id"]))

        async def create_issue(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await issues.create(**dict(params)))

        async def transition_issue(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            issue_id = params["id"]
            existing = issues.get(issue_id)
            previous = existing.status if existing else None
            _unwrap(await issues.transition(issue_id, params["status"], params.get("note")))
            return {"id": issue_id, "previous": previous}

        async def assign_issue(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await issues.assign(params["id"], params["assignee"]))

        async def post_comment(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await comments.post(
                    params["subject_id"], params.get("subject_kind", "issue"), params["body"]
                )
            )

        async def take_snapshot(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await review.snapshot(params.get("name")))

        async def start_session(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await review.start_session(params["name"], params.get("participants", ()))
            )

        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.create,
                title="Add markup",
                permission=MARKUP_PERMISSIONS.create,
                handler=create_markup,
                create_inverse=lambda _params, record: CommandInvocation(
                    MARKUP_COMMANDS.remove, {"id": record.id}
                ),
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.remove,
                title="Delete markup",
                permission=MARKUP_PERMISSIONS.edit,
                handler=remove_markup,
                create_inverse=lambda params, _result: CommandInvocation(
                    MARKUP_COMMANDS.restore, {"id": params["id"]}
                ),
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.restore,
                title="Restore markup",
                permission=MARKUP_PERMISSIONS.edit,
                handler=restore_markup,
                create_inverse=lambda params, _result: CommandInvocation(
                    MARKUP_COMMANDS.remove, {"id": params["id"]}
                ),
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.anchor,
                title="Anchor markup",
                permission=MARKUP_PERMISSIONS.edit,
                handler=anchor_markup,
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.reanchor,
                title="Re-anchor after revision",
                permission=MARKUP_PERMISSIONS.edit,
                handler=reanchor,
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.create_issue,
                title="Raise issue",
                permission=MARKUP_PERMISSIONS.create,
                handler=create_issue,
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.transition_issue,
                title="Change issue status",
                permission=MARKUP_PERMISSIONS.edit,
                handler=transition_issue,
                # The inverse only makes sense if the reverse transition is itself legal; the state
                # machine rejects it otherwise, which is the correct outcome rather than a bug.
                create_inverse=lambda _params, result: (
                    None
                    if result["previous"] is None
                    else CommandInvocation(
                        MARKUP_COMMANDS.transition_issue,
                        {"id": result["id"], "status": result["previous"]},
                    )
                ),
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.assign_issue,
                title="Assign issue",
                permission=MARKUP_PERMISSIONS.assign,
                handler=assign_issue,
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.post_comment,
                title="Post comment",
                permission=MARKUP_PERMISSIONS.create,
                handler=post_comment,
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.snapshot, title="Take review snapshot", handler=take_snapshot
            )
        )
        context.commands.register(
            CommandDefinition(
                id=MARKUP_COMMANDS.start_session,
                title="Start review session",
                handler=start_session,
            )
        )

        context.ui.register(
            UIContribution(
                id="markup.panel", point="panel", title="Issues", placement="right", order=30
            )
        )
        context.ui.register(
            UIContribution(
                id="markup.toolbar.pin",
                point="toolbar",
                title="Add pin",
                group="review",
                order=10,
                command_id=MARKUP_COMMANDS.create,
            )
        )

        context.logger.info("Markup capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Markup",
        description="Pins, redlines, GlobalId anchoring, issues, threads and review snapshots.",
        permissions=[
            MARKUP_PERMISSIONS.create,
            MARKUP_PERMISSIONS.edit,
            MARKUP_PERMISSIONS.assign,
            MARKUP_PERMISSIONS.close,
        ],
        activate=activate,
    )


markup_plugin = create_markup_plugin()
