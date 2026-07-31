from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .common import ElementRef, Id, IsoTimestamp

TaskConstraint = Literal["asap", "start-no-earlier-than", "finish-no-later-than", "must-start-on"]


@dataclass(frozen=True)
class ScheduleTaskRecord:
    id: Id
    name: str
    planned_start: IsoTimestamp
    planned_finish: IsoTimestamp
    #: Id from the source programme (P6, MS Project, XER). Kept for round-tripping.
    external_id: str | None = None
    wbs_code: str | None = None
    parent_id: Id | None = None
    actual_start: IsoTimestamp | None = None
    actual_finish: IsoTimestamp | None = None
    #: 0..1. Distinct from earned quantity, which lives on the 5D side.
    percent_complete: float | None = None
    constraint: TaskConstraint | None = None
    critical: bool = False
    total_float: float | None = None
    resource_ids: tuple[Id, ...] = ()


DependencyType = Literal["FS", "SS", "FF", "SF"]


@dataclass(frozen=True)
class TaskDependencyRecord:
    id: Id
    predecessor_id: Id
    successor_id: Id
    type: DependencyType = "FS"
    #: Lag in days; negative expresses a lead.
    lag: float = 0.0


TaskLinkBehaviour = Literal["construct", "demolish", "temporary", "existing"]

#: How the link maps to IFC when exported.
#:
#: ``IfcRelAssignsToProduct`` is the correct relationship for the products a task **produces** --
#: the task appears in ``RelatedObjects`` and the constructed product is the ``RelatingProduct``.
#:
#: ``IfcRelAssignsToProcess`` is for what a task **consumes**: labour, plant, materials as
#: resources. The two are easy to transpose and the result still validates, so the intended meaning
#: is recorded explicitly rather than inferred at export time.
TaskIfcRelationship = Literal["IfcRelAssignsToProduct", "IfcRelAssignsToProcess"]

#: A 4D link almost always names a task's *output*. Resource assignment is the exception and must
#: be stated.
DEFAULT_TASK_IFC_RELATIONSHIP: TaskIfcRelationship = "IfcRelAssignsToProduct"


@dataclass(frozen=True)
class SelectionRule:
    model_id: Id
    filter: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskModelLinkRecord:
    """Binds model elements to a schedule activity.

    Selection is stored as a *rule* as well as a resolved element list. Re-issuing a model would
    otherwise break every link -- the rule lets the link re-resolve against the new revision
    instead of being rebuilt by hand. Resolved elements carry IFC GlobalIds, so re-resolution is a
    correctness backstop rather than the only thing holding the link together.
    """

    id: Id
    task_id: Id
    behaviour: TaskLinkBehaviour
    elements: tuple[ElementRef, ...] = ()
    ifc_relationship: TaskIfcRelationship = DEFAULT_TASK_IFC_RELATIONSHIP
    selection_rule: SelectionRule | None = None
    resolved_at: IsoTimestamp | None = None
    #: How the link was made -- hand-picked, rule-matched, or imported with the programme.
    link_source: Literal["manual", "rule", "imported"] | None = None


@dataclass(frozen=True)
class SimulationSettings:
    id: Id
    name: str
    from_time: IsoTimestamp
    to_time: IsoTimestamp
    step_unit: Literal["day", "week", "month"] = "week"
    show_planned: bool = True
    show_actual: bool = False


@dataclass(frozen=True)
class ProgressComparisonRecord:
    """Planned-versus-actual comparison for a single activity."""

    id: Id
    task_id: Id
    data_date: IsoTimestamp
    planned_percent: float
    actual_percent: float
    #: Positive means ahead of programme, in days.
    schedule_variance_days: float | None = None
    earned_quantity: float | None = None
    notes: str | None = None
