"""``massingviser.plugins.markup`` -- pins, redlines, issues, threads and review.

Anchoring is the part that has to be right. A markup exists to say something about a *place in a
model*, and models get re-issued. Everything here keys on IFC GlobalId, and a markup whose element
disappears becomes explicitly **orphaned** rather than silently relocating to the origin -- which
is what "all the pins moved" looks like from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
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


@runtime_checkable
class ElementResolver(Protocol):
    """Tells markup whether an element still exists.

    Provided by whatever holds the model. Without it, re-anchoring cannot distinguish "the element
    moved" from "the element was deleted", and both look like a pin in the wrong place.
    """

    def exists(self, model_id: Id, global_id: str) -> bool: ...
    def global_ids(self, model_id: Id) -> Sequence[str]: ...


ElementResolverToken: CapabilityToken[ElementResolver] = create_capability_token(
    "markup.element-resolver"
)


@runtime_checkable
class ViewpointProvider(Protocol):
    async def capture(self, name: str | None = None) -> Result[Mapping[str, Any], KernelError]: ...
    async def apply(self, viewpoint_id: Id) -> Result[None, KernelError]: ...


ViewpointProviderToken: CapabilityToken[ViewpointProvider] = create_capability_token(
    "markup.viewpoint-provider"
)


#: The issue lifecycle, as a state machine rather than a free-text field.
#:
#: Stated explicitly because the illegal transitions are the interesting ones: reopening a *closed*
#: issue has to go back through review, and jumping straight from ``open`` to ``closed`` skips the
#: verification step that the whole workflow exists to enforce.
ISSUE_TRANSITIONS: Mapping[MarkupStatus, tuple[MarkupStatus, ...]] = MappingProxyType(
    {
        "open": ("in-review", "resolved"),
        "in-review": ("open", "resolved"),
        "resolved": ("in-review", "closed"),
        "closed": ("in-review",),
    }
)


@dataclass(frozen=True)
class MarkupQuery:
    model_id: Id | None = None
    kind: str | None = None
    status: MarkupStatus | None = None
    element_id: str | None = None
    viewpoint_id: Id | None = None
    assignee: Id | None = None


@runtime_checkable
class MarkupService(Protocol):
    async def create(self, **markup: Any) -> Result[MarkupRecord, KernelError]: ...
    async def update(
        self, id: Id, changes: Mapping[str, Any]
    ) -> Result[MarkupRecord, KernelError]: ...
    async def remove(self, id: Id) -> Result[None, KernelError]: ...
    def get(self, id: Id) -> MarkupRecord | None: ...
    def query(self, query: MarkupQuery | None = None) -> tuple[MarkupRecord, ...]: ...


MarkupToken: CapabilityToken[MarkupService] = create_capability_token("markup.service")


@dataclass(frozen=True)
class ReanchorReport:
    resolved: int
    orphaned: tuple[Id, ...]
    checked: int


@runtime_checkable
class AnchorService(Protocol):
    async def anchor(
        self,
        markup_id: Id,
        *,
        element: ElementRef | None = None,
        world_position: Sequence[float] | None = None,
    ) -> Result[AnchorReference, KernelError]: ...
    def resolve(self, markup_id: Id) -> AnchorReference | None: ...
    #: Re-checks every anchor in a model after a revision. Reports orphans; never relocates one.
    async def reanchor(self, model_id: Id) -> Result[ReanchorReport, KernelError]: ...
    def orphaned(self) -> tuple[AnchorReference, ...]: ...


AnchorToken: CapabilityToken[AnchorService] = create_capability_token("markup.anchors")


@runtime_checkable
class IssueService(Protocol):
    async def create(self, **issue: Any) -> Result[IssueRecord, KernelError]: ...
    async def update(
        self, id: Id, changes: Mapping[str, Any]
    ) -> Result[IssueRecord, KernelError]: ...
    async def transition(
        self, id: Id, status: MarkupStatus, note: str | None = None
    ) -> Result[IssueRecord, KernelError]: ...
    async def assign(self, id: Id, assignee: Id) -> Result[IssueRecord, KernelError]: ...
    def get(self, id: Id) -> IssueRecord | None: ...
    def query(self, **filter: Any) -> tuple[IssueRecord, ...]: ...


IssueToken: CapabilityToken[IssueService] = create_capability_token("markup.issues")


@runtime_checkable
class CommentService(Protocol):
    def thread(self, subject_id: Id, subject_kind: str) -> CommentThread | None: ...
    async def post(
        self, subject_id: Id, subject_kind: str, body: str
    ) -> Result[CommentRecord, KernelError]: ...
    async def edit(self, comment_id: Id, body: str) -> Result[CommentRecord, KernelError]: ...
    async def resolve_thread(self, thread_id: Id) -> Result[CommentThread, KernelError]: ...


CommentToken: CapabilityToken[CommentService] = create_capability_token("markup.comments")


@runtime_checkable
class ReviewService(Protocol):
    async def snapshot(self, name: str | None = None) -> Result[ReviewSnapshot, KernelError]: ...
    def snapshots(self) -> tuple[ReviewSnapshot, ...]: ...
    async def start_session(
        self, name: str, participants: Sequence[Id] = ()
    ) -> Result[ReviewSession, KernelError]: ...
    async def end_session(self, session_id: Id) -> Result[ReviewSession, KernelError]: ...
    def sessions(self) -> tuple[ReviewSession, ...]: ...


ReviewToken: CapabilityToken[ReviewService] = create_capability_token("markup.review")


class MARKUP_COMMANDS:
    create = "markup.create"
    remove = "markup.remove"
    restore = "markup.restore"
    anchor = "markup.anchor"
    reanchor = "markup.reanchor"
    create_issue = "markup.issue.create"
    transition_issue = "markup.issue.transition"
    assign_issue = "markup.issue.assign"
    post_comment = "markup.comment.post"
    snapshot = "markup.review.snapshot"
    start_session = "markup.review.start"


class MARKUP_PERMISSIONS:
    create = "markup.create"
    edit = "markup.edit"
    assign = "markup.issue.assign"
    close = "markup.issue.close"


class MARKUP_EVENTS:
    created = "markup.created"
    updated = "markup.updated"
    removed = "markup.removed"
    anchored = "markup.anchored"
    orphaned = "markup.orphaned"
    issue_created = "markup.issue.created"
    issue_transitioned = "markup.issue.transitioned"
    comment_posted = "markup.comment.posted"
    snapshot_taken = "markup.review.snapshot-taken"
