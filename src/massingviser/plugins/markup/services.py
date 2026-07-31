from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import (
    AnchorReference,
    CommentRecord,
    CommentThread,
    ElementRef,
    Id,
    IssueRecord,
    MarkupRecord,
    MarkupStatus,
    ReviewSession,
    ReviewSnapshot,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    ISSUE_TRANSITIONS,
    MARKUP_EVENTS,
    ElementResolverToken,
    MarkupQuery,
    ReanchorReport,
    ViewpointProviderToken,
)


@dataclass(frozen=True)
class MarkupStores:
    markups: RecordStore[MarkupRecord]
    anchors: RecordStore[AnchorReference]
    issues: RecordStore[IssueRecord]
    threads: RecordStore[CommentThread]
    snapshots: RecordStore[ReviewSnapshot]
    sessions: RecordStore[ReviewSession]


@dataclass(frozen=True)
class MarkupRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_markup_stores(context: PluginContext) -> MarkupStores:
    return MarkupStores(
        markups=create_record_store(context.state, "markups"),
        anchors=create_record_store(context.state, "anchors"),
        issues=create_record_store(context.state, "issues"),
        threads=create_record_store(context.state, "threads"),
        snapshots=create_record_store(context.state, "snapshots"),
        sessions=create_record_store(context.state, "sessions"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


class MarkupServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MarkupRuntime, stores: MarkupStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def create(self, **markup: Any) -> Result[MarkupRecord, KernelError]:
        element_ids = tuple(markup.get("element_ids", ()))
        # An anchored markup must name GlobalIds, not viewer handles. Catching an integer here is
        # the difference between one clear error and a project's worth of pins that survive exactly
        # one session.
        for element_id in element_ids:
            if not isinstance(element_id, str):
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f"element_ids must be IFC GlobalIds (strings); got {element_id!r}. "
                        "A transient viewer id here would be lost on the next model load.",
                        {"elementId": element_id},
                    )
                )

        record = MarkupRecord(
            id=self._runtime.ids.next("markup"),
            kind=markup.get("kind", "pin"),
            created_at=self._runtime.clock.iso(),
            created_by=markup.get("created_by") or self._runtime.context.permissions.identity.id,
            viewpoint_id=markup.get("viewpoint_id"),
            model_id=markup.get("model_id"),
            element_ids=element_ids,
            world_position=markup.get("world_position"),
            screen_space=markup.get("screen_space"),
            text=markup.get("text"),
            status=markup.get("status", "open"),
            assignee=markup.get("assignee"),
        )
        self._stores.markups.add(record)
        self._runtime.context.events.emit(MARKUP_EVENTS.created, {"record": record})
        return ok(record)

    async def update(self, id: Id, changes: Mapping[str, Any]) -> Result[MarkupRecord, KernelError]:
        updated = self._stores.markups.update(id, dict(changes))
        if updated is None:
            return err(_not_found("markup", id))
        self._runtime.context.events.emit(MARKUP_EVENTS.updated, {"record": updated})
        return ok(updated)

    async def remove(self, id: Id) -> Result[None, KernelError]:
        if not self._stores.markups.remove(id):
            return err(_not_found("markup", id))
        self._stores.anchors.remove_where(lambda anchor: anchor.markup_id == id)
        self._runtime.context.events.emit(MARKUP_EVENTS.removed, {"id": id})
        return ok(None)

    def restore(self, record: MarkupRecord) -> Result[MarkupRecord, KernelError]:
        if self._stores.markups.has(record.id):
            return ok(record)
        self._stores.markups.add(record)
        self._runtime.context.events.emit(MARKUP_EVENTS.created, {"record": record})
        return ok(record)

    def get(self, id: Id) -> MarkupRecord | None:
        return self._stores.markups.get(id)

    def query(self, query: MarkupQuery | None = None) -> tuple[MarkupRecord, ...]:
        if query is None:
            return self._stores.markups.all()
        return tuple(
            record
            for record in self._stores.markups.all()
            if (query.model_id is None or record.model_id == query.model_id)
            and (query.kind is None or record.kind == query.kind)
            and (query.status is None or record.status == query.status)
            and (query.element_id is None or query.element_id in record.element_ids)
            and (query.viewpoint_id is None or record.viewpoint_id == query.viewpoint_id)
            and (query.assignee is None or record.assignee == query.assignee)
        )


class AnchorServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MarkupRuntime, stores: MarkupStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def anchor(
        self,
        markup_id: Id,
        *,
        element: ElementRef | None = None,
        world_position: Sequence[float] | None = None,
    ) -> Result[AnchorReference, KernelError]:
        markup = self._stores.markups.get(markup_id)
        if markup is None:
            return err(_not_found("markup", markup_id))
        if element is None and world_position is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "An anchor needs an element or a world position.",
                    {"markupId": markup_id},
                )
            )

        resolver = self._runtime.context.capabilities.get(ElementResolverToken)
        resolved = True
        if element is not None and resolver is not None:
            resolved = resolver.exists(element.model_id, element.global_id)

        record = AnchorReference(
            id=self._runtime.ids.next("anchor"),
            markup_id=markup_id,
            resolved=resolved,
            model_id=element.model_id if element else markup.model_id,
            global_id=element.global_id if element else None,
            world_position=tuple(world_position) if world_position else markup.world_position,
            resolved_at=self._runtime.clock.iso() if resolved else None,
        )
        # One anchor per markup: two would make "where is this pin" ambiguous.
        self._stores.anchors.remove_where(lambda anchor: anchor.markup_id == markup_id)
        self._stores.anchors.add(record)
        self._runtime.context.events.emit(MARKUP_EVENTS.anchored, {"record": record})
        return ok(record)

    def resolve(self, markup_id: Id) -> AnchorReference | None:
        return self._stores.anchors.find(lambda anchor: anchor.markup_id == markup_id)

    async def reanchor(self, model_id: Id) -> Result[ReanchorReport, KernelError]:
        """Re-check every anchor in a model after a revision.

        Never relocates an anchor. An element that no longer exists produces an orphan, reported by
        id, because moving the pin somewhere plausible is how a review comment silently ends up
        attached to the wrong wall.
        """
        resolver = self._runtime.context.capabilities.get(ElementResolverToken)
        if resolver is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "Re-anchoring needs an element resolver; without one there is no way to tell "
                    "a moved element from a deleted one.",
                    {"modelId": model_id},
                )
            )

        candidates = self._stores.anchors.query(
            lambda anchor: anchor.model_id == model_id and anchor.global_id is not None
        )
        resolved_count = 0
        orphaned: list[Id] = []
        stamp = self._runtime.clock.iso()

        for anchor in candidates:
            exists = resolver.exists(model_id, anchor.global_id or "")
            if exists:
                resolved_count += 1
                if not anchor.resolved:
                    self._stores.anchors.update(anchor.id, {"resolved": True, "resolved_at": stamp})
                continue
            orphaned.append(anchor.markup_id)
            if anchor.resolved:
                self._stores.anchors.update(anchor.id, {"resolved": False, "resolved_at": None})

        report = ReanchorReport(
            resolved=resolved_count, orphaned=tuple(orphaned), checked=len(candidates)
        )
        if orphaned:
            self._runtime.context.events.emit(
                MARKUP_EVENTS.orphaned, {"modelId": model_id, "markupIds": tuple(orphaned)}
            )
        return ok(report)

    def orphaned(self) -> tuple[AnchorReference, ...]:
        return self._stores.anchors.query(lambda anchor: not anchor.resolved)


class IssueServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MarkupRuntime, stores: MarkupStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def create(self, **issue: Any) -> Result[IssueRecord, KernelError]:
        identity = self._runtime.context.permissions.identity.id
        record = IssueRecord(
            id=self._runtime.ids.next("issue"),
            title=issue["title"],
            status=issue.get("status", "open"),
            reporter=issue.get("reporter") or identity,
            created_at=self._runtime.clock.iso(),
            created_by=issue.get("created_by") or identity,
            description=issue.get("description"),
            priority=issue.get("priority"),
            assignee=issue.get("assignee"),
            due_date=issue.get("due_date"),
            markup_ids=tuple(issue.get("markup_ids", ())),
            viewpoint_id=issue.get("viewpoint_id"),
            responsibility=issue.get("responsibility"),
            labels=tuple(issue.get("labels", ())),
        )
        self._stores.issues.add(record)
        self._runtime.context.events.emit(MARKUP_EVENTS.issue_created, {"record": record})
        return ok(record)

    async def update(self, id: Id, changes: Mapping[str, Any]) -> Result[IssueRecord, KernelError]:
        # `status` moves only through `transition`, which is where the state machine lives.
        safe = {key: value for key, value in changes.items() if key != "status"}
        updated = self._stores.issues.update(id, safe)
        return ok(updated) if updated else err(_not_found("issue", id))

    async def transition(
        self, id: Id, status: MarkupStatus, note: str | None = None
    ) -> Result[IssueRecord, KernelError]:
        issue = self._stores.issues.get(id)
        if issue is None:
            return err(_not_found("issue", id))
        if status == issue.status:
            return ok(issue)

        allowed = ISSUE_TRANSITIONS.get(issue.status, ())
        if status not in allowed:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'An issue cannot go from "{issue.status}" to "{status}". '
                    f"Allowed: {', '.join(allowed) or 'none'}.",
                    {"issueId": id, "from": issue.status, "to": status, "allowed": list(allowed)},
                )
            )

        changes: dict[str, Any] = {"status": status}
        changes["closed_at"] = self._runtime.clock.iso() if status == "closed" else None
        updated = self._stores.issues.update(id, changes)
        if updated is None:
            return err(_not_found("issue", id))

        if note:
            thread = self._ensure_thread(id, "issue")
            self._stores.threads.update(
                thread.id,
                {
                    "comments": (
                        *thread.comments,
                        CommentRecord(
                            id=self._runtime.ids.next("comment"),
                            body=note,
                            author_id=self._runtime.context.permissions.identity.id,
                            created_at=self._runtime.clock.iso(),
                        ),
                    )
                },
            )

        self._runtime.context.events.emit(
            MARKUP_EVENTS.issue_transitioned,
            {"issueId": id, "from": issue.status, "to": status},
        )
        return ok(updated)

    async def assign(self, id: Id, assignee: Id) -> Result[IssueRecord, KernelError]:
        updated = self._stores.issues.update(id, {"assignee": assignee})
        return ok(updated) if updated else err(_not_found("issue", id))

    def get(self, id: Id) -> IssueRecord | None:
        return self._stores.issues.get(id)

    def query(self, **filter: Any) -> tuple[IssueRecord, ...]:
        status = filter.get("status")
        assignee = filter.get("assignee")
        priority = filter.get("priority")
        responsibility = filter.get("responsibility")
        return tuple(
            issue
            for issue in self._stores.issues.all()
            if (status is None or issue.status == status)
            and (assignee is None or issue.assignee == assignee)
            and (priority is None or issue.priority == priority)
            and (responsibility is None or issue.responsibility == responsibility)
        )

    def _ensure_thread(self, subject_id: Id, subject_kind: str) -> CommentThread:
        existing = self._stores.threads.find(
            lambda thread: thread.subject_id == subject_id and thread.subject_kind == subject_kind
        )
        if existing is not None:
            return existing
        thread = CommentThread(
            id=self._runtime.ids.next("thread"),
            subject_id=subject_id,
            subject_kind=subject_kind,
            created_at=self._runtime.clock.iso(),
        )
        self._stores.threads.add(thread)
        return thread


class CommentServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MarkupRuntime, stores: MarkupStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def thread(self, subject_id: Id, subject_kind: str) -> CommentThread | None:
        return self._stores.threads.find(
            lambda thread: thread.subject_id == subject_id and thread.subject_kind == subject_kind
        )

    async def post(
        self, subject_id: Id, subject_kind: str, body: str
    ) -> Result[CommentRecord, KernelError]:
        if not body.strip():
            return err(KernelError("COMMAND_FAILED", "A comment needs a body.", {}))

        thread = self.thread(subject_id, subject_kind)
        if thread is None:
            thread = CommentThread(
                id=self._runtime.ids.next("thread"),
                subject_id=subject_id,
                subject_kind=subject_kind,
                created_at=self._runtime.clock.iso(),
            )
            self._stores.threads.add(thread)

        comment = CommentRecord(
            id=self._runtime.ids.next("comment"),
            body=body,
            author_id=self._runtime.context.permissions.identity.id,
            created_at=self._runtime.clock.iso(),
        )
        self._stores.threads.update(thread.id, {"comments": (*thread.comments, comment)})
        self._runtime.context.events.emit(
            MARKUP_EVENTS.comment_posted, {"threadId": thread.id, "comment": comment}
        )
        return ok(comment)

    async def edit(self, comment_id: Id, body: str) -> Result[CommentRecord, KernelError]:
        for thread in self._stores.threads.all():
            for index, comment in enumerate(thread.comments):
                if comment.id != comment_id:
                    continue
                # `edited_at` is set rather than `created_at` overwritten: a review record that
                # hides when a comment changed is not a review record.
                edited = replace(comment, body=body, edited_at=self._runtime.clock.iso())
                comments = list(thread.comments)
                comments[index] = edited
                self._stores.threads.update(thread.id, {"comments": tuple(comments)})
                return ok(edited)
        return err(_not_found("comment", comment_id))

    async def resolve_thread(self, thread_id: Id) -> Result[CommentThread, KernelError]:
        updated = self._stores.threads.update(thread_id, {"resolved": True})
        return ok(updated) if updated else err(_not_found("thread", thread_id))


class ReviewServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MarkupRuntime, stores: MarkupStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def snapshot(self, name: str | None = None) -> Result[ReviewSnapshot, KernelError]:
        """Freeze what the model looked like at review time.

        The captured model versions are what make a snapshot evidential rather than decorative:
        reopening a six-month-old review must show the geometry the reviewer actually saw.
        """
        provider = self._runtime.context.capabilities.get(ViewpointProviderToken)
        viewpoint_id = ""
        if provider is not None:
            captured = await provider.capture(name)
            if not captured.ok:
                return err(captured.error)
            viewpoint_id = captured.value.get("id", "")

        record = ReviewSnapshot(
            id=self._runtime.ids.next("snap"),
            viewpoint_id=viewpoint_id,
            created_at=self._runtime.clock.iso(),
            created_by=self._runtime.context.permissions.identity.id,
            name=name,
            markup_ids=tuple(markup.id for markup in self._stores.markups.all()),
        )
        self._stores.snapshots.add(record)
        self._runtime.context.events.emit(MARKUP_EVENTS.snapshot_taken, {"record": record})
        return ok(record)

    def snapshots(self) -> tuple[ReviewSnapshot, ...]:
        return self._stores.snapshots.all()

    async def start_session(
        self, name: str, participants: Sequence[Id] = ()
    ) -> Result[ReviewSession, KernelError]:
        record = ReviewSession(
            id=self._runtime.ids.next("session"),
            name=name,
            started_at=self._runtime.clock.iso(),
            participants=tuple(participants),
        )
        self._stores.sessions.add(record)
        return ok(record)

    async def end_session(self, session_id: Id) -> Result[ReviewSession, KernelError]:
        session = self._stores.sessions.get(session_id)
        if session is None:
            return err(_not_found("review session", session_id))
        if session.ended_at is not None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Review session "{session.name}" has already ended.',
                    {"sessionId": session_id},
                )
            )
        updated = self._stores.sessions.update(session_id, {"ended_at": self._runtime.clock.iso()})
        return ok(updated) if updated else err(_not_found("review session", session_id))

    def sessions(self) -> tuple[ReviewSession, ...]:
        return self._stores.sessions.all()
