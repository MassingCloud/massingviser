"""``massingviser.plugins.planning`` -- 4D: schedule, model links, playback, planned-versus-actual.

The hard part of 4D is not animation, it is **surviving a re-issue**. A programme is re-exported
weekly and a model monthly, and links made by hand are lost on both events unless the identity they
rest on is stable. Tasks key on the source programme's own id; links key on IFC GlobalId and keep
the rule that produced them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    ElementRef,
    Id,
    IsoTimestamp,
    ProgressComparisonRecord,
    ScheduleTaskRecord,
    SimulationSettings,
    TaskDependencyRecord,
    TaskLinkBehaviour,
    TaskModelLinkRecord,
)

ScheduleFormat = Literal["csv", "json", "xer", "mspdi"]


@runtime_checkable
class ElementFilterSource(Protocol):
    """Resolves a selection rule to elements.

    Rule-based linking is what makes a link survive a model re-issue: the rule re-resolves against
    the new revision, where a hand-picked list would simply be stale.
    """

    def match(self, model_id: Id, filter: Mapping[str, Any]) -> Sequence[ElementRef]: ...
    def model_ids(self) -> Sequence[Id]: ...


ElementFilterSourceToken: CapabilityToken[ElementFilterSource] = create_capability_token(
    "planning.element-filter"
)


@dataclass(frozen=True)
class ScheduleImportSummary:
    tasks: int
    dependencies: int
    format: ScheduleFormat
    #: Rows the importer could not read. Named, never dropped silently -- a programme that imports
    #: "successfully" while missing a trade is worse than one that refuses.
    rejected: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReimportSummary(ScheduleImportSummary):
    added: int = 0
    updated: int = 0
    removed: int = 0
    #: Links whose task disappeared from the new programme. Reported so somebody decides, rather
    #: than deleted along with the task.
    orphaned_links: tuple[Id, ...] = ()


@runtime_checkable
class ScheduleImportService(Protocol):
    def supported_formats(self) -> tuple[ScheduleFormat, ...]: ...
    async def import_schedule(
        self, payload: str | bytes, format: ScheduleFormat
    ) -> Result[ScheduleImportSummary, KernelError]: ...
    async def reimport(
        self, payload: str | bytes, format: ScheduleFormat
    ) -> Result[ReimportSummary, KernelError]: ...
    def tasks(self, **filter: Any) -> tuple[ScheduleTaskRecord, ...]: ...
    def dependencies(self, task_id: Id | None = None) -> tuple[TaskDependencyRecord, ...]: ...
    async def export(self, format: ScheduleFormat) -> Result[str, KernelError]: ...


ScheduleImportToken: CapabilityToken[ScheduleImportService] = create_capability_token(
    "planning.schedule"
)


@dataclass(frozen=True)
class ReresolveSummary:
    resolved: int
    #: Links whose rule now matches nothing. The link is kept and reported, not deleted.
    unmatched: tuple[Id, ...] = ()


@runtime_checkable
class TaskModelLinkService(Protocol):
    async def link(
        self,
        task_id: Id,
        elements: Sequence[ElementRef],
        behaviour: TaskLinkBehaviour,
        **options: Any,
    ) -> Result[TaskModelLinkRecord, KernelError]: ...
    async def link_by_rule(
        self,
        task_id: Id,
        model_id: Id,
        filter: Mapping[str, Any],
        behaviour: TaskLinkBehaviour,
        **options: Any,
    ) -> Result[TaskModelLinkRecord, KernelError]: ...
    async def unlink(self, link_id: Id) -> Result[None, KernelError]: ...
    def links(self, task_id: Id | None = None) -> tuple[TaskModelLinkRecord, ...]: ...
    async def reresolve(
        self, model_id: Id | None = None
    ) -> Result[ReresolveSummary, KernelError]: ...
    #: Elements no task claims. The 4D equivalent of an unpriced line, and just as expensive.
    async def unlinked_elements(
        self, model_id: Id
    ) -> Result[tuple[ElementRef, ...], KernelError]: ...


TaskModelLinkToken: CapabilityToken[TaskModelLinkService] = create_capability_token(
    "planning.links"
)


@runtime_checkable
class TimelinePlaybackService(Protocol):
    async def configure(self, **settings: Any) -> Result[SimulationSettings, KernelError]: ...
    async def seek(self, at: IsoTimestamp) -> Result[None, KernelError]: ...
    def current_date(self) -> IsoTimestamp | None: ...
    #: What the model looks like at an instant, grouped by what each task does to its elements.
    async def state_at(
        self, at: IsoTimestamp
    ) -> Result[Mapping[str, tuple[ElementRef, ...]], KernelError]: ...


TimelinePlaybackToken: CapabilityToken[TimelinePlaybackService] = create_capability_token(
    "planning.playback"
)


@runtime_checkable
class PlannedActualComparisonService(Protocol):
    async def compare(
        self, data_date: IsoTimestamp, task_ids: Sequence[Id] | None = None
    ) -> Result[tuple[ProgressComparisonRecord, ...], KernelError]: ...
    async def behind_schedule(
        self, data_date: IsoTimestamp, threshold_days: float = 0.0
    ) -> Result[tuple[ProgressComparisonRecord, ...], KernelError]: ...


PlannedActualToken: CapabilityToken[PlannedActualComparisonService] = create_capability_token(
    "planning.planned-actual"
)


class PLANNING_COMMANDS:
    import_schedule = "planning.schedule.import"
    reimport_schedule = "planning.schedule.reimport"
    link_selection = "planning.link.selection"
    link_by_rule = "planning.link.rule"
    reresolve_links = "planning.link.reresolve"
    seek = "planning.playback.seek"
    compare_progress = "planning.compare"


class PLANNING_PERMISSIONS:
    import_schedule = "planning.import"
    edit_links = "planning.link.edit"


class PLANNING_EVENTS:
    schedule_imported = "planning.schedule.imported"
    links_changed = "planning.links.changed"
    playback_seeked = "planning.playback.seeked"
    progress_compared = "planning.progress.compared"
