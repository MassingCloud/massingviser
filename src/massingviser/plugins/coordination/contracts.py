"""``massingviser.plugins.coordination`` -- clash, validation, routing and revision diff.

The defining property is that a re-run **accumulates knowledge instead of discarding it**. Clash
detection that produces fresh ids every run throws away last week's triage, which is the single most
common way clash workflows fail in practice. Everything here is built around a stable signature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    ClashKind,
    ClashRecord,
    ClashStatus,
    ClashTestRecord,
    ElementRef,
    Id,
    ResponsibilityRecord,
    RevisionDiffRecord,
    ValidationResultRecord,
    ValidationRuleRecord,
    ValidationSeverity,
)


@dataclass(frozen=True)
class RawClash:
    """One intersection, as the geometry engine reports it -- before it has an identity."""

    a: ElementRef
    b: ElementRef
    point: tuple[float, float, float] | None = None
    distance: float | None = None


@runtime_checkable
class ClashEngine(Protocol):
    """Geometry. Deliberately a port.

    Coordination owns the *workflow* -- signatures, triage, ageing, routing -- and knows nothing
    about how an intersection is found. A BVH in Python, a native library, or a viewer's own
    collision pass all satisfy this.
    """

    def intersect(
        self,
        a: Sequence[ElementRef],
        b: Sequence[ElementRef],
        kind: ClashKind,
        tolerance: float,
    ) -> Sequence[RawClash]: ...


ClashEngineToken: CapabilityToken[ClashEngine] = create_capability_token(
    "coordination.clash-engine"
)


@dataclass(frozen=True)
class SnapshotElement:
    """An element as it stood in one revision of one model."""

    global_id: str
    ifc_class: str
    properties: Mapping[str, Any] = field(default_factory=dict)
    position: tuple[float, float, float] | None = None
    quantities: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class ModelSnapshotSource(Protocol):
    """Frozen element sets per model revision -- what makes a diff possible at all.

    Without stored snapshots, "what changed between C02 and C03" can only be answered if both
    revisions happen to be loaded, which is exactly when nobody needs to ask.
    """

    def snapshot(self, model_id: Id, version: str) -> Sequence[SnapshotElement] | None: ...
    def model_ids(self) -> Sequence[Id]: ...
    def versions(self, model_id: Id) -> Sequence[str]: ...


ModelSnapshotToken: CapabilityToken[ModelSnapshotSource] = create_capability_token(
    "coordination.model-snapshot"
)


@dataclass(frozen=True)
class ClashRunSummary:
    test_id: Id
    total: int
    #: Seen for the first time in this run.
    created: int
    #: Seen before, and whose triage was carried forward.
    persisted: int
    #: Previously seen and no longer occurring. Marked resolved, never deleted.
    resolved: int
    run_at: str


@runtime_checkable
class ClashService(Protocol):
    async def define_test(self, **test: Any) -> Result[ClashTestRecord, KernelError]: ...
    async def run(self, test_id: Id) -> Result[ClashRunSummary, KernelError]: ...
    def results(
        self, test_id: Id, *, status: ClashStatus | None = None
    ) -> tuple[ClashRecord, ...]: ...
    async def set_status(
        self, clash_id: Id, status: ClashStatus, note: str | None = None
    ) -> Result[ClashRecord, KernelError]: ...
    async def promote_to_issue(
        self, clash_id: Id, assignee: Id | None = None
    ) -> Result[Id, KernelError]: ...
    def tests(self) -> tuple[ClashTestRecord, ...]: ...


ClashToken: CapabilityToken[ClashService] = create_capability_token("coordination.clash")


@dataclass(frozen=True)
class ValidationFinding:
    message: str
    element: ElementRef | None = None
    severity: ValidationSeverity | None = None


@runtime_checkable
class ValidationRule(Protocol):
    """A model-quality check, contributed as a capability.

    Many-to-one on purpose: every plugin that knows a rule registers one, and the service
    aggregates. A rule set baked into coordination would mean editing this package to add a check
    somebody else's discipline cares about.
    """

    @property
    def descriptor(self) -> ValidationRuleRecord: ...
    def check(
        self, model_id: Id, elements: Sequence[SnapshotElement]
    ) -> Sequence[ValidationFinding]: ...


ValidationRuleToken: CapabilityToken[ValidationRule] = create_capability_token("coordination.rule")


@dataclass(frozen=True)
class ValidationRunSummary:
    rules_run: int
    checked: int
    errors: int
    warnings: int
    #: Rules that raised. Reported, never allowed to abort the run -- one bad rule must not stop
    #: a model being validated against the other twenty.
    failed_rules: tuple[tuple[Id, str], ...] = ()


@runtime_checkable
class ValidationService(Protocol):
    def rules(self) -> tuple[ValidationRuleRecord, ...]: ...
    def set_enabled(self, rule_id: Id, enabled: bool) -> None: ...
    async def run(
        self, *, rule_ids: Sequence[Id] | None = None, model_ids: Sequence[Id] | None = None
    ) -> Result[ValidationRunSummary, KernelError]: ...
    def results(
        self, *, severity: ValidationSeverity | None = None
    ) -> tuple[ValidationResultRecord, ...]: ...


ValidationToken: CapabilityToken[ValidationService] = create_capability_token(
    "coordination.validation"
)


@dataclass(frozen=True)
class RoutingRule:
    """Sends an issue to whoever answers for it.

    Matching is on discipline and label rather than on a person: people move between projects and
    a rule naming one goes stale silently, whereas a discipline outlives the roster.
    """

    id: Id
    #: Matched against the issue's `responsibility`.
    discipline: str | None = None
    #: All of these must be present on the issue.
    labels: tuple[str, ...] = ()
    #: Assigned when the rule matches. Resolved through the responsibility matrix when absent.
    assignee: Id | None = None
    priority: int = 0


@dataclass(frozen=True)
class RoutingSummary:
    routed: int
    #: Issues no rule matched. Reported rather than dumped on a default owner, who then ignores
    #: everything they are sent because most of it is not theirs.
    unrouted: tuple[Id, ...] = ()


@runtime_checkable
class IssueDirectory(Protocol):
    """The slice of an issue tracker that routing needs.

    Declared here, by capability *id*, rather than imported from the markup package. A capability
    token is a contract identified by a string, so naming ``"markup.issues"`` couples coordination
    to the **contract** and not to the package that happens to satisfy it -- which is the whole
    point of the registry, and what keeps `test_no_capability_plugin_imports_another` green.

    Any plugin providing that id satisfies this: markup does today, a Jira bridge could tomorrow.
    """

    def query(self, **filter: Any) -> Sequence[Any]: ...
    async def assign(self, id: Id, assignee: Id) -> Result[Any, KernelError]: ...


IssueDirectoryToken: CapabilityToken[IssueDirectory] = create_capability_token("markup.issues")


@runtime_checkable
class IssueRoutingService(Protocol):
    async def add_rule(self, **rule: Any) -> Result[RoutingRule, KernelError]: ...
    async def remove_rule(self, rule_id: Id) -> Result[None, KernelError]: ...
    def rules(self) -> tuple[RoutingRule, ...]: ...
    async def route(
        self, issue_ids: Sequence[Id] | None = None
    ) -> Result[RoutingSummary, KernelError]: ...


IssueRoutingToken: CapabilityToken[IssueRoutingService] = create_capability_token(
    "coordination.routing"
)


@runtime_checkable
class RevisionDiffService(Protocol):
    async def compare(
        self, model_id: Id, from_version: str, to_version: str
    ) -> Result[RevisionDiffRecord, KernelError]: ...
    async def compare_to_previous(
        self, model_id: Id
    ) -> Result[RevisionDiffRecord, KernelError]: ...
    def get(self, diff_id: Id) -> RevisionDiffRecord | None: ...
    def list(self) -> tuple[RevisionDiffRecord, ...]: ...


RevisionDiffToken: CapabilityToken[RevisionDiffService] = create_capability_token(
    "coordination.diff"
)


@runtime_checkable
class ResponsibilityMatrixService(Protocol):
    def entries(self) -> tuple[ResponsibilityRecord, ...]: ...
    async def upsert(self, **record: Any) -> Result[ResponsibilityRecord, KernelError]: ...
    async def remove(self, record_id: Id) -> Result[None, KernelError]: ...
    def responsible_for(self, discipline: str) -> ResponsibilityRecord | None: ...


ResponsibilityToken: CapabilityToken[ResponsibilityMatrixService] = create_capability_token(
    "coordination.responsibility"
)


class COORDINATION_COMMANDS:
    define_clash_test = "coordination.clash.define"
    run_clash_test = "coordination.clash.run"
    set_clash_status = "coordination.clash.set-status"
    promote_clash_to_issue = "coordination.clash.promote"
    run_validation = "coordination.validation.run"
    compare_revisions = "coordination.diff.compare"
    route_issues = "coordination.routing.run"


class COORDINATION_PERMISSIONS:
    run_tests = "coordination.run"
    triage = "coordination.triage"
    manage_rules = "coordination.rules.manage"


class COORDINATION_EVENTS:
    clash_run_completed = "coordination.clash.run-completed"
    clash_status_changed = "coordination.clash.status-changed"
    clash_promoted = "coordination.clash.promoted"
    validation_completed = "coordination.validation.completed"
    diff_computed = "coordination.diff.computed"
    issues_routed = "coordination.routing.completed"
