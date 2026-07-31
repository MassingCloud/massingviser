from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .common import ElementRef, Id, IsoTimestamp, Money

PackageStatus = Literal["draft", "issued", "tendering", "awarded", "in-progress", "complete"]


@dataclass(frozen=True)
class ProcurementPackageRecord:
    """A scope of work bought as one unit -- the bridge from estimate to execution."""

    id: Id
    code: str
    name: str
    status: PackageStatus
    created_at: IsoTimestamp
    boq_line_ids: tuple[Id, ...] = ()
    elements: tuple[ElementRef, ...] = ()
    task_ids: tuple[Id, ...] = ()
    budget: Money | None = None
    awarded_value: Money | None = None
    vendor_id: Id | None = None
    required_on_site: IsoTimestamp | None = None
    lead_time_days: int | None = None


@dataclass(frozen=True)
class VendorRecord:
    id: Id
    name: str
    trade: str | None = None
    contact_email: str | None = None
    prequalified: bool = False
    rating: float | None = None


@dataclass(frozen=True)
class VendorScopeRecord:
    id: Id
    package_id: Id
    vendor_id: Id
    inclusions: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    quoted_value: Money | None = None
    submitted_at: IsoTimestamp | None = None


FieldState = Literal["not-started", "in-progress", "installed", "inspected", "accepted", "rework"]


@dataclass(frozen=True)
class FieldStatusRecord:
    """Observed state of real work on real elements.

    The point at which 5D stops being an estimate: quantities installed here are what earned value
    and progress claims are computed from, so this record is deliberately element-level rather than
    a percentage typed against a task.
    """

    id: Id
    element: ElementRef
    state: FieldState
    observed_at: IsoTimestamp
    observed_by: Id
    package_id: Id | None = None
    task_id: Id | None = None
    quantity_installed: float | None = None
    unit: str | None = None
    photo_uris: tuple[str, ...] = ()
    notes: str | None = None


InspectionOutcome = Literal["pass", "fail", "pass-with-comments", "not-inspected"]


@dataclass(frozen=True)
class InspectionRecord:
    id: Id
    name: str
    outcome: InspectionOutcome
    inspected_at: IsoTimestamp
    inspected_by: Id
    checklist_id: Id | None = None
    package_id: Id | None = None
    elements: tuple[ElementRef, ...] = ()
    issue_ids: tuple[Id, ...] = ()
    signature_uri: str | None = None


@dataclass(frozen=True)
class InstallProgressRecord:
    id: Id
    package_id: Id
    data_date: IsoTimestamp
    quantity_installed: float
    quantity_total: float
    unit: str
    percent_complete: float
    earned_value: Money | None = None
