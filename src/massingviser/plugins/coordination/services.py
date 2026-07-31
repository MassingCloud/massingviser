from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import (
    ClashRecord,
    ClashStatus,
    ClashTestRecord,
    ElementRef,
    Id,
    ResponsibilityRecord,
    RevisionDiffEntry,
    RevisionDiffRecord,
    ValidationResultRecord,
    ValidationRuleRecord,
    ValidationSeverity,
    element_key,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    COORDINATION_EVENTS,
    ClashEngineToken,
    ClashRunSummary,
    IssueDirectoryToken,
    ModelSnapshotToken,
    RoutingRule,
    RoutingSummary,
    SnapshotElement,
    ValidationRuleToken,
    ValidationRunSummary,
)

#: Triage state a re-run must carry forward rather than discard.
PRESERVED_STATUSES: tuple[ClashStatus, ...] = ("reviewed", "approved", "ignored", "active")


def clash_signature(test_id: Id, a: ElementRef, b: ElementRef) -> str:
    """A stable identity for one clash, independent of run order.

    Endpoints are sorted before hashing, so the same pair of elements produces the same signature
    whichever selection each happened to land in. This is the property that decides whether a
    weekly clash cycle accumulates knowledge or discards it.
    """
    left = element_key(a)
    right = element_key(b)
    first, second = (left, right) if left <= right else (right, left)
    return hashlib.sha256(f"{test_id}|{first}|{second}".encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CoordinationStores:
    tests: RecordStore[ClashTestRecord]
    clashes: RecordStore[ClashRecord]
    rules: RecordStore[ValidationRuleRecord]
    results: RecordStore[ValidationResultRecord]
    diffs: RecordStore[RevisionDiffRecord]
    responsibilities: RecordStore[ResponsibilityRecord]
    routing: RecordStore[RoutingRule]


@dataclass(frozen=True)
class CoordinationRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_coordination_stores(context: PluginContext) -> CoordinationStores:
    return CoordinationStores(
        tests=create_record_store(context.state, "tests"),
        clashes=create_record_store(context.state, "clashes"),
        rules=create_record_store(context.state, "rules"),
        results=create_record_store(context.state, "results"),
        diffs=create_record_store(context.state, "diffs"),
        responsibilities=create_record_store(context.state, "responsibilities"),
        routing=create_record_store(context.state, "routing"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


# ---------------------------------------------------------------------------------------------
# Clash
# ---------------------------------------------------------------------------------------------


class ClashServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: CoordinationRuntime, stores: CoordinationStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def define_test(self, **test: Any) -> Result[ClashTestRecord, KernelError]:
        tolerance = test.get("tolerance", 0.0)
        kind = test.get("kind", "hard")
        if kind == "clearance" and tolerance <= 0:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "A clearance test needs a positive tolerance; with zero it is a hard clash "
                    "test wearing the wrong name.",
                    {"tolerance": tolerance},
                )
            )
        record = ClashTestRecord(
            id=test.get("id") or self._runtime.ids.next("clashtest"),
            name=test["name"],
            kind=kind,
            tolerance=tolerance,
            selection_a=tuple(test.get("selection_a", ())),
            selection_b=tuple(test.get("selection_b", ())),
        )
        self._stores.tests.add(record)
        return ok(record)

    def tests(self) -> tuple[ClashTestRecord, ...]:
        return self._stores.tests.all()

    async def run(self, test_id: Id) -> Result[ClashRunSummary, KernelError]:
        test = self._stores.tests.get(test_id)
        if test is None:
            return err(_not_found("clash test", test_id))

        engine = self._runtime.context.capabilities.get(ClashEngineToken)
        if engine is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No clash engine is installed. Coordination owns the workflow; a plugin must "
                    "provide the geometry through ClashEngineToken.",
                    {"testId": test_id},
                )
            )

        selection_a = [ElementRef(model_id=m, global_id=m) for m in test.selection_a]
        selection_b = [ElementRef(model_id=m, global_id=m) for m in test.selection_b]
        raw = engine.intersect(selection_a, selection_b, test.kind, test.tolerance)

        run_at = self._runtime.clock.iso()
        previous = {
            clash.signature: clash
            for clash in self._stores.clashes.query(lambda c: c.test_id == test_id)
        }

        seen: set[str] = set()
        created = 0
        persisted = 0

        for candidate in raw:
            signature = clash_signature(test_id, candidate.a, candidate.b)
            seen.add(signature)
            existing = previous.get(signature)

            if existing is not None:
                # Everything a human decided about this clash survives; only the observation
                # updates. A re-run that reset triage would make the whole cycle pointless.
                changes: dict[str, Any] = {"last_seen_at": run_at}
                if existing.status not in PRESERVED_STATUSES:
                    changes["status"] = "active"
                if candidate.point is not None:
                    changes["point"] = candidate.point
                if candidate.distance is not None:
                    changes["distance"] = candidate.distance
                self._stores.clashes.update(existing.id, changes)
                persisted += 1
                continue

            self._stores.clashes.add(
                ClashRecord(
                    id=self._runtime.ids.next("clash"),
                    test_id=test_id,
                    kind=test.kind,
                    status="new",
                    a=candidate.a,
                    b=candidate.b,
                    signature=signature,
                    first_seen_at=run_at,
                    last_seen_at=run_at,
                    point=candidate.point,
                    distance=candidate.distance,
                )
            )
            created += 1

        # Clashes that no longer occur become resolved rather than being deleted -- the record that
        # one existed, and what was decided about it, is the audit trail.
        resolved = 0
        for signature, clash in previous.items():
            if signature in seen or clash.status in ("resolved", "ignored"):
                continue
            self._stores.clashes.update(clash.id, {"status": "resolved", "last_seen_at": run_at})
            resolved += 1

        total = len(self._stores.clashes.query(lambda c: c.test_id == test_id))
        self._stores.tests.update(test_id, {"last_run_at": run_at, "clash_count": total})

        summary = ClashRunSummary(
            test_id=test_id,
            total=total,
            created=created,
            persisted=persisted,
            resolved=resolved,
            run_at=run_at,
        )
        self._runtime.context.events.emit(
            COORDINATION_EVENTS.clash_run_completed, {"summary": summary}
        )
        return ok(summary)

    def results(
        self, test_id: Id, *, status: ClashStatus | None = None
    ) -> tuple[ClashRecord, ...]:
        return self._stores.clashes.query(
            lambda clash: clash.test_id == test_id and (status is None or clash.status == status)
        )

    async def set_status(
        self, clash_id: Id, status: ClashStatus, note: str | None = None
    ) -> Result[ClashRecord, KernelError]:
        existing = self._stores.clashes.get(clash_id)
        if existing is None:
            return err(_not_found("clash", clash_id))
        updated = self._stores.clashes.update(clash_id, {"status": status})
        self._runtime.context.events.emit(
            COORDINATION_EVENTS.clash_status_changed,
            {"clashId": clash_id, "from": existing.status, "to": status, "note": note},
        )
        return ok(updated) if updated else err(_not_found("clash", clash_id))

    async def promote_to_issue(
        self, clash_id: Id, assignee: Id | None = None
    ) -> Result[Id, KernelError]:
        """Hand a clash to the markup plugin as an issue.

        Routed through the command bus rather than an import: coordination has no dependency on
        markup, and a deployment without it simply cannot promote -- which is reported, not
        crashed.
        """
        clash = self._stores.clashes.get(clash_id)
        if clash is None:
            return err(_not_found("clash", clash_id))
        if clash.issue_id:
            return ok(clash.issue_id)  # idempotent: promoting twice must not raise two issues

        if not self._runtime.context.commands.has("markup.issue.create"):
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No markup capability is installed, so a clash cannot become an issue.",
                    {"clashId": clash_id},
                )
            )

        created = await self._runtime.context.commands.execute(
            "markup.issue.create",
            {
                "title": f"Clash: {clash.a.global_id} vs {clash.b.global_id}",
                "description": (
                    f"{clash.kind} clash from test {clash.test_id}"
                    + (f", overlap {clash.distance:.3f} m" if clash.distance else "")
                ),
                "priority": "high" if clash.kind == "hard" else "medium",
                "assignee": assignee,
                "labels": ("clash", clash.kind),
            },
        )
        if not created.ok:
            return err(created.error)

        issue_id = created.value.id
        self._stores.clashes.update(clash_id, {"issue_id": issue_id, "assignee": assignee})
        self._runtime.context.events.emit(
            COORDINATION_EVENTS.clash_promoted, {"clashId": clash_id, "issueId": issue_id}
        )
        return ok(issue_id)


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


class ValidationServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: CoordinationRuntime, stores: CoordinationStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def _providers(self) -> list[Any]:
        return [
            provider.value
            for provider in self._runtime.context.capabilities.get_all(ValidationRuleToken)
        ]

    def _sync_descriptors(self) -> None:
        """Mirror contributed rules into records so they can be enabled and disabled."""
        for rule in self._providers():
            descriptor = rule.descriptor
            if not self._stores.rules.has(descriptor.id):
                self._stores.rules.add(descriptor)

    def rules(self) -> tuple[ValidationRuleRecord, ...]:
        self._sync_descriptors()
        return self._stores.rules.all()

    def set_enabled(self, rule_id: Id, enabled: bool) -> None:
        self._sync_descriptors()
        self._stores.rules.update(rule_id, {"enabled": enabled})

    async def run(
        self, *, rule_ids: Sequence[Id] | None = None, model_ids: Sequence[Id] | None = None
    ) -> Result[ValidationRunSummary, KernelError]:
        snapshots = self._runtime.context.capabilities.get(ModelSnapshotToken)
        if snapshots is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "Validation needs a model snapshot source to check against.",
                    {},
                )
            )
        self._sync_descriptors()

        wanted = set(rule_ids) if rule_ids is not None else None
        active = [
            rule
            for rule in self._providers()
            if (wanted is None or rule.descriptor.id in wanted)
            and (self._stores.rules.get(rule.descriptor.id) or rule.descriptor).enabled
        ]
        models = list(model_ids) if model_ids is not None else list(snapshots.model_ids())

        checked_at = self._runtime.clock.iso()
        checked = 0
        errors = 0
        warnings = 0
        failed: list[tuple[Id, str]] = []
        produced: list[ValidationResultRecord] = []

        for model_id in models:
            versions = list(snapshots.versions(model_id))
            elements = snapshots.snapshot(model_id, versions[-1]) if versions else None
            if not elements:
                continue
            checked += len(elements)

            for rule in active:
                try:
                    findings = rule.check(model_id, elements)
                except Exception as thrown:  # noqa: BLE001
                    # One bad rule must not stop a model being validated against the other twenty.
                    failed.append((rule.descriptor.id, str(thrown)))
                    continue
                for finding in findings:
                    severity: ValidationSeverity = (
                        finding.severity or rule.descriptor.severity
                    )
                    if severity == "error":
                        errors += 1
                    elif severity == "warning":
                        warnings += 1
                    produced.append(
                        ValidationResultRecord(
                            id=self._runtime.ids.next("finding"),
                            rule_id=rule.descriptor.id,
                            severity=severity,
                            message=finding.message,
                            checked_at=checked_at,
                            element=finding.element,
                        )
                    )

        ran = {rule.descriptor.id for rule in active}
        # A re-run supersedes the previous findings for the rules that ran, rather than stacking.
        self._stores.results.remove_where(lambda r: r.rule_id in ran)
        self._stores.results.add_many(produced)

        summary = ValidationRunSummary(
            rules_run=len(active),
            checked=checked,
            errors=errors,
            warnings=warnings,
            failed_rules=tuple(failed),
        )
        self._runtime.context.events.emit(
            COORDINATION_EVENTS.validation_completed, {"summary": summary}
        )
        return ok(summary)

    def results(
        self, *, severity: ValidationSeverity | None = None
    ) -> tuple[ValidationResultRecord, ...]:
        if severity is None:
            return self._stores.results.all()
        return self._stores.results.query(lambda r: r.severity == severity)


# ---------------------------------------------------------------------------------------------
# Routing and the responsibility matrix
# ---------------------------------------------------------------------------------------------


class ResponsibilityMatrixServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: CoordinationRuntime, stores: CoordinationStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def entries(self) -> tuple[ResponsibilityRecord, ...]:
        return self._stores.responsibilities.all()

    async def upsert(self, **record: Any) -> Result[ResponsibilityRecord, KernelError]:
        record_id = record.get("id")
        if record_id and self._stores.responsibilities.has(record_id):
            updated = self._stores.responsibilities.update(
                record_id, {k: v for k, v in record.items() if k != "id"}
            )
            return ok(updated) if updated else err(_not_found("responsibility", record_id))
        created = ResponsibilityRecord(
            id=record_id or self._runtime.ids.next("resp"),
            scope=record["scope"],
            discipline=record["discipline"],
            organisation=record.get("organisation"),
            responsible_id=record.get("responsible_id"),
            accountable_id=record.get("accountable_id"),
            consulted_ids=tuple(record.get("consulted_ids", ())),
            informed_ids=tuple(record.get("informed_ids", ())),
        )
        self._stores.responsibilities.add(created)
        return ok(created)

    async def remove(self, record_id: Id) -> Result[None, KernelError]:
        return (
            ok(None)
            if self._stores.responsibilities.remove(record_id)
            else err(_not_found("responsibility", record_id))
        )

    def responsible_for(self, discipline: str) -> ResponsibilityRecord | None:
        return self._stores.responsibilities.find(lambda r: r.discipline == discipline)


class IssueRoutingServiceImpl:
    __slots__ = ("_runtime", "_stores", "_matrix")

    def __init__(
        self,
        runtime: CoordinationRuntime,
        stores: CoordinationStores,
        matrix: ResponsibilityMatrixServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._matrix = matrix

    async def add_rule(self, **rule: Any) -> Result[RoutingRule, KernelError]:
        record = RoutingRule(
            id=rule.get("id") or self._runtime.ids.next("route"),
            discipline=rule.get("discipline"),
            labels=tuple(rule.get("labels", ())),
            assignee=rule.get("assignee"),
            priority=rule.get("priority", 0),
        )
        if record.discipline is None and not record.labels:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "A routing rule that matches on nothing would capture every issue.",
                    {},
                )
            )
        self._stores.routing.add(record)
        return ok(record)

    async def remove_rule(self, rule_id: Id) -> Result[None, KernelError]:
        return ok(None) if self._stores.routing.remove(rule_id) else err(
            _not_found("routing rule", rule_id)
        )

    def rules(self) -> tuple[RoutingRule, ...]:
        return tuple(sorted(self._stores.routing.all(), key=lambda r: -r.priority))

    async def route(
        self, issue_ids: Sequence[Id] | None = None
    ) -> Result[RoutingSummary, KernelError]:
        issues_service = self._runtime.context.capabilities.get(IssueDirectoryToken)
        if issues_service is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND", "No markup capability, so there are no issues to route.", {}
                )
            )

        candidates = [
            issue
            for issue in issues_service.query()
            if (issue_ids is None or issue.id in set(issue_ids)) and issue.assignee is None
        ]

        routed = 0
        unrouted: list[Id] = []
        for issue in candidates:
            assignee = None
            for rule in self.rules():
                if rule.discipline is not None and issue.responsibility != rule.discipline:
                    continue
                if rule.labels and not set(rule.labels).issubset(set(issue.labels)):
                    continue
                assignee = rule.assignee
                if assignee is None and rule.discipline is not None:
                    entry = self._matrix.responsible_for(rule.discipline)
                    assignee = entry.responsible_id if entry else None
                if assignee is not None:
                    break
            if assignee is None:
                # Reported rather than dumped on a default owner, who then ignores everything they
                # are sent because most of it is not theirs.
                unrouted.append(issue.id)
                continue
            await issues_service.assign(issue.id, assignee)
            routed += 1

        summary = RoutingSummary(routed=routed, unrouted=tuple(unrouted))
        self._runtime.context.events.emit(
            COORDINATION_EVENTS.issues_routed, {"summary": summary}
        )
        return ok(summary)


# ---------------------------------------------------------------------------------------------
# Revision diff
# ---------------------------------------------------------------------------------------------


class RevisionDiffServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: CoordinationRuntime, stores: CoordinationStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def compare(
        self, model_id: Id, from_version: str, to_version: str
    ) -> Result[RevisionDiffRecord, KernelError]:
        snapshots = self._runtime.context.capabilities.get(ModelSnapshotToken)
        if snapshots is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "A diff needs stored snapshots of both revisions; without them the question "
                    "can only be answered when both happen to be loaded.",
                    {"modelId": model_id},
                )
            )

        before = snapshots.snapshot(model_id, from_version)
        after = snapshots.snapshot(model_id, to_version)
        missing = [
            version
            for version, snapshot in ((from_version, before), (to_version, after))
            if snapshot is None
        ]
        if missing:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'No snapshot of "{model_id}" at version(s) {", ".join(missing)}.',
                    {"modelId": model_id, "missing": missing},
                )
            )

        old = {element.global_id: element for element in before or ()}
        new = {element.global_id: element for element in after or ()}
        entries: list[RevisionDiffEntry] = []

        for global_id, element in new.items():
            reference = ElementRef(model_id=model_id, global_id=global_id)
            previous = old.get(global_id)
            if previous is None:
                entries.append(
                    RevisionDiffEntry(
                        element=reference,
                        kind="added",
                        quantity_delta=dict(element.quantities),
                    )
                )
                continue

            moved = (
                previous.position is not None
                and element.position is not None
                and previous.position != element.position
            )
            changed = tuple(
                sorted(
                    key
                    for key in set(previous.properties) | set(element.properties)
                    if previous.properties.get(key) != element.properties.get(key)
                )
            )
            delta = {
                metric: element.quantities.get(metric, 0.0) - previous.quantities.get(metric, 0.0)
                for metric in set(previous.quantities) | set(element.quantities)
                if element.quantities.get(metric, 0.0) != previous.quantities.get(metric, 0.0)
            }
            if not moved and not changed and not delta:
                continue
            entries.append(
                RevisionDiffEntry(
                    # "moved" is reported distinctly from "modified": a relocation and a
                    # respecification have entirely different downstream consequences.
                    element=reference,
                    kind="moved" if moved and not changed else "modified",
                    changed_properties=changed,
                    quantity_delta=delta,
                )
            )

        for global_id, element in old.items():
            if global_id in new:
                continue
            entries.append(
                RevisionDiffEntry(
                    element=ElementRef(model_id=model_id, global_id=global_id),
                    kind="removed",
                    # Negative, so the 5D hand-off can add deltas without special-casing removal.
                    quantity_delta={
                        metric: -value for metric, value in element.quantities.items()
                    },
                )
            )

        record = RevisionDiffRecord(
            id=self._runtime.ids.next("diff"),
            model_id=model_id,
            from_version=from_version,
            to_version=to_version,
            computed_at=self._runtime.clock.iso(),
            entries=tuple(entries),
        )
        self._stores.diffs.add(record)
        self._runtime.context.events.emit(
            COORDINATION_EVENTS.diff_computed, {"record": record}
        )
        return ok(record)

    async def compare_to_previous(self, model_id: Id) -> Result[RevisionDiffRecord, KernelError]:
        snapshots = self._runtime.context.capabilities.get(ModelSnapshotToken)
        if snapshots is None:
            return err(
                KernelError("CAPABILITY_NOT_FOUND", "No model snapshot source is installed.", {})
            )
        versions = list(snapshots.versions(model_id))
        if len(versions) < 2:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'"{model_id}" has only {len(versions)} revision(s); there is nothing to '
                    "compare it to.",
                    {"modelId": model_id},
                )
            )
        return await self.compare(model_id, versions[-2], versions[-1])

    def get(self, diff_id: Id) -> RevisionDiffRecord | None:
        return self._stores.diffs.get(diff_id)

    def list(self) -> tuple[RevisionDiffRecord, ...]:
        return self._stores.diffs.all()
