from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import (
    DEFAULT_TASK_IFC_RELATIONSHIP,
    ElementRef,
    Id,
    IsoTimestamp,
    ProgressComparisonRecord,
    ScheduleTaskRecord,
    SelectionRule,
    SimulationSettings,
    TaskDependencyRecord,
    TaskLinkBehaviour,
    TaskModelLinkRecord,
    element_key,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .calendars import WorkCalendar, calendar_or_default
from .contracts import (
    PLANNING_EVENTS,
    ElementFilterSourceToken,
    ReimportSummary,
    ReresolveSummary,
    ScheduleFormat,
    ScheduleImportSummary,
)
from .formats import (
    ScheduleParseError,
    flatten_predecessors,
    parse_mspdi,
    parse_xer,
)

_REQUIRED_COLUMNS = ("id", "name", "planned_start", "planned_finish")


@dataclass(frozen=True)
class PlanningStores:
    tasks: RecordStore[ScheduleTaskRecord]
    dependencies: RecordStore[TaskDependencyRecord]
    links: RecordStore[TaskModelLinkRecord]
    settings: RecordStore[SimulationSettings]
    progress: RecordStore[ProgressComparisonRecord]


@dataclass(frozen=True)
class PlanningRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_planning_stores(context: PluginContext) -> PlanningStores:
    return PlanningStores(
        tasks=create_record_store(context.state, "tasks"),
        dependencies=create_record_store(context.state, "dependencies"),
        links=create_record_store(context.state, "links"),
        settings=create_record_store(context.state, "settings"),
        progress=create_record_store(context.state, "progress"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


def parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO timestamp, tolerating a trailing ``Z`` and a bare date.

    A programme export is somebody else's file format; being strict here means refusing a
    perfectly readable schedule over a timezone suffix.
    """
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------------------------
# Schedule import
# ---------------------------------------------------------------------------------------------


class ScheduleImportServiceImpl:
    """Reads a programme, and re-reads it without losing the model links.

    Four formats, all reduced to the same row shape before anything downstream sees them: ``csv``
    and ``json`` for interchange, ``xer`` for Primavera P6 and ``mspdi`` for MS Project. What the
    two vendor readers cover, and what they deliberately leave in the file, is documented in
    ``planning.formats``.
    """

    __slots__ = ("_runtime", "_stores", "_calendars")

    def __init__(self, runtime: PlanningRuntime, stores: PlanningStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._calendars: dict[str, WorkCalendar] = {}

    def supported_formats(self) -> tuple[ScheduleFormat, ...]:
        return ("csv", "json", "xer", "mspdi")

    def _rows(
        self, payload: str | bytes, format: ScheduleFormat
    ) -> Result[list[Mapping[str, Any]], KernelError]:
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
        if format == "json":
            try:
                parsed = json.loads(text)
            except ValueError as thrown:
                return err(
                    KernelError("COMMAND_FAILED", f"Schedule is not valid JSON: {thrown}", {})
                )
            rows = parsed.get("tasks", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(rows, list):
                return err(
                    KernelError(
                        "COMMAND_FAILED", "Expected a list of tasks or {'tasks': [...]}.", {}
                    )
                )
            return ok(rows)
        if format == "csv":
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                return err(KernelError("COMMAND_FAILED", "The CSV has no header row.", {}))
            missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f"Schedule CSV is missing column(s): {', '.join(missing)}.",
                        {"missing": missing},
                    )
                )
            return ok(list(reader))
        if format in ("xer", "mspdi"):
            parser = parse_xer if format == "xer" else parse_mspdi
            try:
                rows, calendars = parser(payload)
                # Held on the service rather than on each record: a calendar is shared by many
                # tasks, and copying it onto every one would be the same object stored a thousand
                # times and able to disagree with itself.
                self._calendars = dict(calendars)
                return ok(rows)
            except ScheduleParseError as thrown:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f"Could not read the {format.upper()} programme: {thrown}",
                        {"format": format},
                    )
                )
        return err(
            KernelError(
                "COMMAND_FAILED", f'Unsupported schedule format "{format}".', {"format": format}
            )
        )

    def _to_task(self, row: Mapping[str, Any]) -> tuple[ScheduleTaskRecord | None, str | None]:
        external = str(row.get("id") or row.get("external_id") or "").strip()
        name = str(row.get("name") or "").strip()
        start = parse_timestamp(str(row.get("planned_start") or ""))
        finish = parse_timestamp(str(row.get("planned_finish") or ""))

        if not external:
            return None, "row has no id"
        if not name:
            return None, f"{external}: no name"
        if start is None or finish is None:
            return None, f"{external}: unreadable planned dates"
        if finish < start:
            # A task that finishes before it starts poisons every duration computed from it.
            return None, f"{external}: finishes before it starts"

        percent = row.get("percent_complete")
        actual_start = parse_timestamp(str(row.get("actual_start") or ""))
        actual_finish = parse_timestamp(str(row.get("actual_finish") or ""))

        return (
            ScheduleTaskRecord(
                # The source programme's own id *is* the identity, so a re-export matches rather
                # than duplicating. A generated id would make every re-import a fresh programme.
                id=external,
                external_id=external,
                name=name,
                planned_start=_iso(start),
                planned_finish=_iso(finish),
                wbs_code=str(row["wbs_code"]) if row.get("wbs_code") else None,
                parent_id=str(row["parent_id"]) if row.get("parent_id") else None,
                actual_start=_iso(actual_start) if actual_start else None,
                actual_finish=_iso(actual_finish) if actual_finish else None,
                percent_complete=float(percent) if percent not in (None, "") else None,
                critical=(
                    row["critical"]
                    if isinstance(row.get("critical"), bool)
                    else str(row.get("critical", "")).lower() in ("1", "true", "yes")
                ),
                calendar_id=str(row["calendar_id"]) if row.get("calendar_id") else None,
            ),
            None,
        )

    def _read(
        self, payload: str | bytes, format: ScheduleFormat
    ) -> Result[
        tuple[list[ScheduleTaskRecord], list[TaskDependencyRecord], list[tuple[str, str]]],
        KernelError,
    ]:
        rows = self._rows(payload, format)
        if not rows.ok:
            return err(rows.error)

        tasks: list[ScheduleTaskRecord] = []
        rejected: list[tuple[str, str]] = []
        for row in rows.value:
            task, reason = self._to_task(row)
            if task is None:
                rejected.append((str(row.get("id", "?")), reason or "unreadable"))
            else:
                tasks.append(task)

        known = {task.id for task in tasks}
        dependencies: list[TaskDependencyRecord] = []
        # A CSV row names at most one predecessor; a real programme names several, so the vendor
        # readers put them in a list. Both shapes are flattened to the same pairs here.
        pairs = [
            {
                "successor": str(row.get("id") or "").strip(),
                "predecessor": str(row.get("predecessor") or "").strip(),
                "type": str(row.get("dependency_type") or "FS").upper(),
                "lag": float(row.get("lag") or 0.0),
            }
            for row in rows.value
            if str(row.get("predecessor") or "").strip()
        ] + flatten_predecessors(rows.value)

        for pair in pairs:
            predecessor = str(pair.get("predecessor") or "").strip()
            if not predecessor:
                continue
            successor = str(pair.get("successor") or "").strip()
            if predecessor not in known or successor not in known:
                # A dependency naming a task that is not in the file is a broken programme, not a
                # reason to drop the whole import.
                rejected.append((successor, f"dependency on unknown task {predecessor}"))
                continue
            dependencies.append(
                TaskDependencyRecord(
                    id=f"{predecessor}->{successor}",
                    predecessor_id=predecessor,
                    successor_id=successor,
                    type=str(pair.get("type") or "FS").upper(),  # type: ignore[arg-type]
                    lag=float(pair.get("lag") or 0.0),
                )
            )
        return ok((tasks, dependencies, rejected))

    async def import_schedule(
        self, payload: str | bytes, format: ScheduleFormat
    ) -> Result[ScheduleImportSummary, KernelError]:
        read = self._read(payload, format)
        if not read.ok:
            return err(read.error)
        tasks, dependencies, rejected = read.value

        self._stores.tasks.clear()
        self._stores.dependencies.clear()
        self._stores.tasks.add_many(tasks)
        self._stores.dependencies.add_many(dependencies)

        summary = ScheduleImportSummary(
            tasks=len(tasks),
            dependencies=len(dependencies),
            format=format,
            rejected=tuple(rejected),
        )
        self._runtime.context.events.emit(PLANNING_EVENTS.schedule_imported, {"summary": summary})
        return ok(summary)

    async def reimport(
        self, payload: str | bytes, format: ScheduleFormat
    ) -> Result[ReimportSummary, KernelError]:
        """Re-read a programme, keeping every model link whose task still exists.

        This is the operation that decides whether 4D survives contact with a project. A programme
        is re-issued weekly; if re-importing means re-linking, nobody re-imports, and the model
        silently drifts away from the plan.
        """
        read = self._read(payload, format)
        if not read.ok:
            return err(read.error)
        tasks, dependencies, rejected = read.value

        before = {task.id: task for task in self._stores.tasks.all()}
        incoming = {task.id: task for task in tasks}

        added = sum(1 for task_id in incoming if task_id not in before)
        removed = sum(1 for task_id in before if task_id not in incoming)
        updated = sum(
            1 for task_id, task in incoming.items() if task_id in before and before[task_id] != task
        )

        self._stores.tasks.clear()
        self._stores.dependencies.clear()
        self._stores.tasks.add_many(tasks)
        self._stores.dependencies.add_many(dependencies)

        # Links survive by task id. Those whose task vanished are reported rather than deleted --
        # somebody has to decide whether the work moved or was cancelled.
        orphaned = tuple(
            link.id for link in self._stores.links.all() if link.task_id not in incoming
        )

        summary = ReimportSummary(
            tasks=len(tasks),
            dependencies=len(dependencies),
            format=format,
            rejected=tuple(rejected),
            added=added,
            updated=updated,
            removed=removed,
            orphaned_links=orphaned,
        )
        self._runtime.context.events.emit(PLANNING_EVENTS.schedule_imported, {"summary": summary})
        return ok(summary)

    def calendars(self) -> Mapping[str, WorkCalendar]:
        """Working calendars from the last import, keyed as the source programme named them."""
        return dict(self._calendars)

    def calendar_for(self, task: ScheduleTaskRecord) -> WorkCalendar:
        """The calendar a task is scheduled against, or the assumed five-day week."""
        return calendar_or_default(self._calendars, getattr(task, "calendar_id", None))

    def tasks(self, **filter: Any) -> tuple[ScheduleTaskRecord, ...]:
        parent_id = filter.get("parent_id")
        critical = filter.get("critical")
        return tuple(
            sorted(
                (
                    task
                    for task in self._stores.tasks.all()
                    if (parent_id is None or task.parent_id == parent_id)
                    and (critical is None or task.critical == critical)
                ),
                key=lambda task: task.planned_start,
            )
        )

    def dependencies(self, task_id: Id | None = None) -> tuple[TaskDependencyRecord, ...]:
        if task_id is None:
            return self._stores.dependencies.all()
        return self._stores.dependencies.query(
            lambda d: d.predecessor_id == task_id or d.successor_id == task_id
        )

    async def export(self, format: ScheduleFormat) -> Result[str, KernelError]:
        tasks = self.tasks()
        predecessors = {d.successor_id: d for d in self._stores.dependencies.all()}
        rows = [
            {
                "id": task.id,
                "name": task.name,
                "planned_start": task.planned_start,
                "planned_finish": task.planned_finish,
                "actual_start": task.actual_start or "",
                "actual_finish": task.actual_finish or "",
                "percent_complete": task.percent_complete
                if task.percent_complete is not None
                else "",
                "wbs_code": task.wbs_code or "",
                "parent_id": task.parent_id or "",
                "critical": "true" if task.critical else "false",
                "predecessor": predecessors[task.id].predecessor_id
                if task.id in predecessors
                else "",
                "dependency_type": predecessors[task.id].type if task.id in predecessors else "",
                "lag": predecessors[task.id].lag if task.id in predecessors else "",
            }
            for task in tasks
        ]
        if format == "json":
            return ok(json.dumps({"tasks": rows}, indent=2))
        if format == "csv":
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer, fieldnames=list(rows[0]) if rows else list(_REQUIRED_COLUMNS)
            )
            writer.writeheader()
            writer.writerows(rows)
            return ok(buffer.getvalue())
        return err(KernelError("COMMAND_FAILED", f'Unsupported schedule format "{format}".', {}))


# ---------------------------------------------------------------------------------------------
# Model links
# ---------------------------------------------------------------------------------------------


class TaskModelLinkServiceImpl:
    __slots__ = ("_runtime", "_stores", "_calendars")

    def __init__(self, runtime: PlanningRuntime, stores: PlanningStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._calendars: dict[str, WorkCalendar] = {}

    async def link(
        self,
        task_id: Id,
        elements: Sequence[ElementRef],
        behaviour: TaskLinkBehaviour,
        **options: Any,
    ) -> Result[TaskModelLinkRecord, KernelError]:
        if not self._stores.tasks.has(task_id):
            return err(_not_found("task", task_id))
        if not elements:
            return err(
                KernelError(
                    "COMMAND_FAILED", "A link with no elements links nothing.", {"taskId": task_id}
                )
            )
        record = TaskModelLinkRecord(
            id=self._runtime.ids.next("link"),
            task_id=task_id,
            behaviour=behaviour,
            elements=tuple(elements),
            ifc_relationship=options.get("ifc_relationship", DEFAULT_TASK_IFC_RELATIONSHIP),
            resolved_at=self._runtime.clock.iso(),
            link_source="manual",
        )
        self._stores.links.add(record)
        self._runtime.context.events.emit(PLANNING_EVENTS.links_changed, {"record": record})
        return ok(record)

    async def link_by_rule(
        self,
        task_id: Id,
        model_id: Id,
        filter: Mapping[str, Any],
        behaviour: TaskLinkBehaviour,
        **options: Any,
    ) -> Result[TaskModelLinkRecord, KernelError]:
        if not self._stores.tasks.has(task_id):
            return err(_not_found("task", task_id))
        source = self._runtime.context.capabilities.get(ElementFilterSourceToken)
        if source is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "Rule-based linking needs an element filter source to resolve against.",
                    {"taskId": task_id},
                )
            )
        matched = tuple(source.match(model_id, filter))
        record = TaskModelLinkRecord(
            id=self._runtime.ids.next("link"),
            task_id=task_id,
            behaviour=behaviour,
            elements=matched,
            ifc_relationship=options.get("ifc_relationship", DEFAULT_TASK_IFC_RELATIONSHIP),
            # The rule is kept, not just its result. That is what lets the link re-resolve against
            # the next revision instead of being rebuilt by hand.
            selection_rule=SelectionRule(model_id=model_id, filter=dict(filter)),
            resolved_at=self._runtime.clock.iso(),
            link_source="rule",
        )
        self._stores.links.add(record)
        self._runtime.context.events.emit(PLANNING_EVENTS.links_changed, {"record": record})
        return ok(record)

    async def unlink(self, link_id: Id) -> Result[None, KernelError]:
        return ok(None) if self._stores.links.remove(link_id) else err(_not_found("link", link_id))

    def links(self, task_id: Id | None = None) -> tuple[TaskModelLinkRecord, ...]:
        if task_id is None:
            return self._stores.links.all()
        return self._stores.links.query(lambda link: link.task_id == task_id)

    async def reresolve(self, model_id: Id | None = None) -> Result[ReresolveSummary, KernelError]:
        source = self._runtime.context.capabilities.get(ElementFilterSourceToken)
        if source is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND", "Re-resolving needs an element filter source.", {}
                )
            )
        stamp = self._runtime.clock.iso()
        resolved = 0
        unmatched: list[Id] = []

        for link in self._stores.links.all():
            rule = link.selection_rule
            if rule is None or (model_id is not None and rule.model_id != model_id):
                continue  # a hand-picked link has no rule to re-run
            matched = tuple(source.match(rule.model_id, rule.filter))
            if not matched:
                # Kept and reported. Deleting it would silently drop the plan's coverage of work
                # that probably still exists under a different classification.
                unmatched.append(link.id)
                continue
            self._stores.links.update(link.id, {"elements": matched, "resolved_at": stamp})
            resolved += 1

        summary = ReresolveSummary(resolved=resolved, unmatched=tuple(unmatched))
        self._runtime.context.events.emit(PLANNING_EVENTS.links_changed, {"summary": summary})
        return ok(summary)

    async def unlinked_elements(self, model_id: Id) -> Result[tuple[ElementRef, ...], KernelError]:
        source = self._runtime.context.capabilities.get(ElementFilterSourceToken)
        if source is None:
            return err(
                KernelError("CAPABILITY_NOT_FOUND", "No element filter source is installed.", {})
            )
        everything = source.match(model_id, {})
        linked = {
            element_key(element) for link in self._stores.links.all() for element in link.elements
        }
        return ok(tuple(e for e in everything if element_key(e) not in linked))


# ---------------------------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------------------------


class TimelinePlaybackServiceImpl:
    __slots__ = ("_runtime", "_stores", "_current")

    def __init__(self, runtime: PlanningRuntime, stores: PlanningStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._current: IsoTimestamp | None = None

    async def configure(self, **settings: Any) -> Result[SimulationSettings, KernelError]:
        record = SimulationSettings(
            id=settings.get("id") or self._runtime.ids.next("sim"),
            name=settings.get("name", "Simulation"),
            from_time=settings["from_time"],
            to_time=settings["to_time"],
            step_unit=settings.get("step_unit", "week"),
            show_planned=settings.get("show_planned", True),
            show_actual=settings.get("show_actual", False),
        )
        if parse_timestamp(record.to_time) is None or parse_timestamp(record.from_time) is None:
            return err(KernelError("COMMAND_FAILED", "Simulation bounds are unreadable.", {}))
        self._stores.settings.clear()
        self._stores.settings.add(record)
        return ok(record)

    async def seek(self, at: IsoTimestamp) -> Result[None, KernelError]:
        if parse_timestamp(at) is None:
            return err(KernelError("COMMAND_FAILED", f'"{at}" is not a readable date.', {"at": at}))
        self._current = at
        self._runtime.context.events.emit(PLANNING_EVENTS.playback_seeked, {"at": at})
        return ok(None)

    def current_date(self) -> IsoTimestamp | None:
        return self._current

    async def state_at(
        self, at: IsoTimestamp
    ) -> Result[Mapping[str, tuple[ElementRef, ...]], KernelError]:
        """What the model looks like at an instant, grouped by behaviour.

        Grouped rather than flattened to a visibility list because the four behaviours render
        differently: constructed work appears, demolished work disappears, temporary work appears
        and then goes, and existing work was always there.
        """
        moment = parse_timestamp(at)
        if moment is None:
            return err(KernelError("COMMAND_FAILED", f'"{at}" is not a readable date.', {"at": at}))

        buckets: dict[str, list[ElementRef]] = {
            "construct": [],
            "demolish": [],
            "temporary": [],
            "existing": [],
        }
        for link in self._stores.links.all():
            task = self._stores.tasks.get(link.task_id)
            if task is None:
                continue
            start = parse_timestamp(task.actual_start or task.planned_start)
            finish = parse_timestamp(task.actual_finish or task.planned_finish)
            if start is None or finish is None:
                continue

            if link.behaviour == "existing":
                buckets["existing"].extend(link.elements)
            elif link.behaviour == "temporary":
                # Present only while its task is running.
                if start <= moment <= finish:
                    buckets["temporary"].extend(link.elements)
            elif link.behaviour == "construct":
                if moment >= finish:
                    buckets["construct"].extend(link.elements)
            elif link.behaviour == "demolish":
                if moment >= finish:
                    buckets["demolish"].extend(link.elements)

        return ok({name: tuple(elements) for name, elements in buckets.items()})


# ---------------------------------------------------------------------------------------------
# Planned versus actual
# ---------------------------------------------------------------------------------------------


class PlannedActualComparisonServiceImpl:
    __slots__ = ("_runtime", "_stores", "_calendars")

    def __init__(self, runtime: PlanningRuntime, stores: PlanningStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._calendars: dict[str, WorkCalendar] = {}

    async def compare(
        self, data_date: IsoTimestamp, task_ids: Sequence[Id] | None = None
    ) -> Result[tuple[ProgressComparisonRecord, ...], KernelError]:
        moment = parse_timestamp(data_date)
        if moment is None:
            return err(KernelError("COMMAND_FAILED", f'"{data_date}" is not a readable date.', {}))

        wanted = set(task_ids) if task_ids is not None else None
        produced: list[ProgressComparisonRecord] = []

        for task in self._stores.tasks.all():
            if wanted is not None and task.id not in wanted:
                continue
            start = parse_timestamp(task.planned_start)
            finish = parse_timestamp(task.planned_finish)
            if start is None or finish is None:
                continue

            span = (finish - start).total_seconds()
            if moment <= start:
                planned = 0.0
            elif moment >= finish or span <= 0:
                planned = 1.0
            else:
                planned = (moment - start).total_seconds() / span

            actual = task.percent_complete
            if actual is None:
                actual = 1.0 if task.actual_finish else (0.0 if not task.actual_start else 0.5)

            # Variance in days, signed so positive is ahead. Expressed against the task's own
            # duration rather than a calendar month, because a two-day task three days late is a
            # different problem from a two-year task three days late.
            variance_days = ((actual - planned) * span / 86400.0) if span > 0 else 0.0

            produced.append(
                ProgressComparisonRecord(
                    id=self._runtime.ids.next("progress"),
                    task_id=task.id,
                    data_date=data_date,
                    planned_percent=round(planned, 6),
                    actual_percent=round(actual, 6),
                    schedule_variance_days=round(variance_days, 3),
                )
            )

        self._stores.progress.remove_where(lambda p: p.data_date == data_date)
        self._stores.progress.add_many(produced)
        self._runtime.context.events.emit(
            PLANNING_EVENTS.progress_compared, {"dataDate": data_date, "count": len(produced)}
        )
        return ok(tuple(produced))

    async def behind_schedule(
        self, data_date: IsoTimestamp, threshold_days: float = 0.0
    ) -> Result[tuple[ProgressComparisonRecord, ...], KernelError]:
        compared = await self.compare(data_date)
        if not compared.ok:
            return err(compared.error)
        return ok(
            tuple(
                record
                for record in compared.value
                if (record.schedule_variance_days or 0.0) < -abs(threshold_days)
            )
        )
