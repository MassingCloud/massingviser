from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .common import ElementRef, Id, IsoTimestamp

ClashKind = Literal["hard", "clearance", "duplicate", "workflow"]
ClashStatus = Literal["new", "active", "reviewed", "approved", "resolved", "ignored"]


@dataclass(frozen=True)
class ClashRecord:
    id: Id
    test_id: Id
    kind: ClashKind
    status: ClashStatus
    a: ElementRef
    b: ElementRef
    #: Stable hash of the participating elements and geometry.
    #:
    #: This is what makes a clash test re-runnable: without it every run produces fresh ids and the
    #: triage work done last week is lost, which is the single most common way clash workflows fail
    #: in practice.
    signature: str
    first_seen_at: IsoTimestamp
    last_seen_at: IsoTimestamp
    point: tuple[float, float, float] | None = None
    #: Penetration depth, or shortfall against the required clearance, in project units.
    distance: float | None = None
    issue_id: Id | None = None
    assignee: Id | None = None


@dataclass(frozen=True)
class ClashTestRecord:
    id: Id
    name: str
    kind: ClashKind
    #: Minimum separation for a clearance test, in project units.
    tolerance: float = 0.0
    selection_a: tuple[Id, ...] = ()
    selection_b: tuple[Id, ...] = ()
    last_run_at: IsoTimestamp | None = None
    clash_count: int | None = None


ValidationSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ValidationRuleRecord:
    id: Id
    name: str
    severity: ValidationSeverity
    description: str | None = None
    #: Rule source -- a standard, a company handbook, or a custom check.
    standard: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class ValidationResultRecord:
    id: Id
    rule_id: Id
    severity: ValidationSeverity
    message: str
    checked_at: IsoTimestamp
    element: ElementRef | None = None
    issue_id: Id | None = None


ChangeKind = Literal["added", "removed", "modified", "moved"]


@dataclass(frozen=True)
class RevisionDiffEntry:
    """One element's difference between two model revisions."""

    element: ElementRef
    kind: ChangeKind
    changed_properties: tuple[str, ...] = ()
    #: Quantity delta, when the change affects measured quantities -- the 5D hand-off.
    quantity_delta: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RevisionDiffRecord:
    id: Id
    model_id: Id
    from_version: str
    to_version: str
    computed_at: IsoTimestamp
    entries: tuple[RevisionDiffEntry, ...] = ()


@dataclass(frozen=True)
class ResponsibilityRecord:
    """Who answers for a scope. Drives issue routing and package ownership."""

    id: Id
    scope: str
    discipline: str
    organisation: str | None = None
    responsible_id: Id | None = None
    accountable_id: Id | None = None
    consulted_ids: tuple[Id, ...] = ()
    informed_ids: tuple[Id, ...] = ()
