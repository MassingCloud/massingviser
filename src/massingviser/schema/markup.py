from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .common import Id, IsoTimestamp

MarkupKind = Literal["pin", "redline", "cloud", "note", "measurement-ref", "snapshot-link"]
MarkupStatus = Literal["open", "in-review", "resolved", "closed"]
IssuePriority = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class MarkupRecord:
    """A single piece of markup.

    Consumed by coordination, review and field plugins, so its contract is one of the few that must
    not drift.
    """

    id: Id
    kind: MarkupKind
    created_at: IsoTimestamp
    created_by: Id
    viewpoint_id: Id | None = None
    model_id: Id | None = None
    #: IFC GlobalIds. Not viewer-local ids -- a markup anchored to a transient id is a lost markup.
    element_ids: tuple[str, ...] = ()
    world_position: tuple[float, float, float] | None = None
    screen_space: tuple[float, float] | None = None
    text: str | None = None
    status: MarkupStatus = "open"
    assignee: Id | None = None


@dataclass(frozen=True)
class AnchorReference:
    """How a markup stays attached to what it is about.

    Separated from ``MarkupRecord`` because anchoring is the part that degrades: a pin placed on an
    element that a later model revision deletes must become *orphaned*, not silently relocate to
    the origin. ``resolved`` records whether the anchor still binds.
    """

    id: Id
    markup_id: Id
    resolved: bool
    model_id: Id | None = None
    #: IFC GlobalId of the anchored element. Never a transient viewer id -- see ``ElementRef``.
    global_id: str | None = None
    world_position: tuple[float, float, float] | None = None
    #: Position relative to the anchored element, so the markup survives the element moving.
    local_offset: tuple[float, float, float] | None = None
    resolved_at: IsoTimestamp | None = None


@dataclass(frozen=True)
class IssueRecord:
    id: Id
    title: str
    status: MarkupStatus
    reporter: Id
    created_at: IsoTimestamp
    created_by: Id
    description: str | None = None
    priority: IssuePriority | None = None
    assignee: Id | None = None
    due_date: IsoTimestamp | None = None
    markup_ids: tuple[Id, ...] = ()
    viewpoint_id: Id | None = None
    #: Discipline or package this issue is routed to, e.g. ``"MEP"``, ``"Structure"``.
    responsibility: str | None = None
    labels: tuple[str, ...] = ()
    closed_at: IsoTimestamp | None = None


@dataclass(frozen=True)
class CommentRecord:
    id: Id
    body: str
    author_id: Id
    created_at: IsoTimestamp
    edited_at: IsoTimestamp | None = None
    attachments: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CommentThread:
    id: Id
    #: Record this thread hangs off -- an issue, a markup, a clash, a cost line.
    subject_id: Id
    subject_kind: str
    created_at: IsoTimestamp
    comments: tuple[CommentRecord, ...] = ()
    resolved: bool = False


@dataclass(frozen=True)
class ModelVersionRef:
    model_id: Id
    version: str


@dataclass(frozen=True)
class ReviewSnapshot:
    """A frozen record of what the model looked like at review time.

    The stored ``model_versions`` are what make a snapshot evidential rather than decorative:
    reopening a six-month-old review must show the geometry the reviewer actually saw, not today's.
    """

    id: Id
    viewpoint_id: Id
    created_at: IsoTimestamp
    created_by: Id
    name: str | None = None
    image_uri: str | None = None
    model_versions: tuple[ModelVersionRef, ...] = ()
    markup_ids: tuple[Id, ...] = ()


@dataclass(frozen=True)
class ReviewSession:
    id: Id
    name: str
    started_at: IsoTimestamp
    participants: tuple[Id, ...] = ()
    snapshot_ids: tuple[Id, ...] = ()
    issue_ids: tuple[Id, ...] = ()
    ended_at: IsoTimestamp | None = None
