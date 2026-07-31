"""``massingviser.plugins.procurement`` -- packages, vendors, field status, inspection, earned value.

Where 5D stops being an estimate. Progress is recorded **per element**, not as a percentage typed
against a task, because that is the only form an earned-value claim can be defended in: "78%" is an
opinion, "these 412 elements are installed and these 96 are not" is a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    ElementRef,
    FieldState,
    FieldStatusRecord,
    Id,
    InspectionRecord,
    InstallProgressRecord,
    IsoTimestamp,
    Money,
    PackageStatus,
    ProcurementPackageRecord,
    VendorRecord,
    VendorScopeRecord,
)


@dataclass(frozen=True)
class PackageBoqLine:
    """The slice of a bill line a package needs, without importing the estimating package."""

    id: Id
    description: str
    total: Money | None = None
    classification_code: str | None = None
    quantity: float | None = None
    unit: str | None = None


@runtime_checkable
class BoqLineSource(Protocol):
    """Priced lines, supplied by whatever owns the bill.

    Declared by capability id rather than imported, so procurement runs against this platform's
    estimating plugin, an imported bill, or a fixture.
    """

    def lines(self, line_ids: Sequence[Id] | None = None) -> Sequence[PackageBoqLine]: ...


BoqLineSourceToken: CapabilityToken[BoqLineSource] = create_capability_token(
    "procurement.boq-lines"
)


@runtime_checkable
class PackageService(Protocol):
    async def create(self, **pkg: Any) -> Result[ProcurementPackageRecord, KernelError]: ...
    async def update(
        self, package_id: Id, changes: Mapping[str, Any]
    ) -> Result[ProcurementPackageRecord, KernelError]: ...
    async def set_status(
        self, package_id: Id, status: PackageStatus
    ) -> Result[ProcurementPackageRecord, KernelError]: ...
    async def from_boq_lines(
        self, boq_line_ids: Sequence[Id], name: str, code: str
    ) -> Result[ProcurementPackageRecord, KernelError]: ...
    def list(self, **filter: Any) -> tuple[ProcurementPackageRecord, ...]: ...
    def get(self, package_id: Id) -> ProcurementPackageRecord | None: ...
    #: Priced lines no package covers. Every one is scope somebody will discover on site.
    async def uncovered_scope(self) -> Result[tuple[Id, ...], KernelError]: ...


PackageToken: CapabilityToken[PackageService] = create_capability_token("procurement.packages")


@dataclass(frozen=True)
class VendorComparison:
    vendor_id: Id
    vendor_name: str
    quoted: Money | None
    #: Signed difference against the package budget. Negative is under.
    variance: Money | None
    inclusions: int
    exclusions: int
    #: Comparability warning. A cheaper bid that excludes more is not a cheaper bid.
    caveat: str | None = None


@runtime_checkable
class VendorScopeService(Protocol):
    def vendors(self) -> tuple[VendorRecord, ...]: ...
    async def upsert_vendor(self, **vendor: Any) -> Result[VendorRecord, KernelError]: ...
    def scopes(self, package_id: Id) -> tuple[VendorScopeRecord, ...]: ...
    async def submit_scope(self, **scope: Any) -> Result[VendorScopeRecord, KernelError]: ...
    async def compare(
        self, package_id: Id
    ) -> Result[tuple[VendorComparison, ...], KernelError]: ...
    async def award(
        self, package_id: Id, vendor_id: Id, value: Money
    ) -> Result[ProcurementPackageRecord, KernelError]: ...


VendorScopeToken: CapabilityToken[VendorScopeService] = create_capability_token(
    "procurement.vendor-scope"
)


@runtime_checkable
class FieldStatusService(Protocol):
    async def record(self, **status: Any) -> Result[FieldStatusRecord, KernelError]: ...
    async def record_many(
        self, statuses: Sequence[Mapping[str, Any]]
    ) -> Result[int, KernelError]: ...
    #: The latest observation for an element. History is kept; this is the state.
    def current(self, element: ElementRef) -> FieldStatusRecord | None: ...
    def query(self, **filter: Any) -> tuple[FieldStatusRecord, ...]: ...


FieldStatusToken: CapabilityToken[FieldStatusService] = create_capability_token("field.status")


@runtime_checkable
class InspectionService(Protocol):
    async def create(self, **inspection: Any) -> Result[InspectionRecord, KernelError]: ...
    #: Records a failure and raises an issue per finding, so a defect cannot be closed by
    #: forgetting about it.
    async def fail(
        self, inspection_id: Id, findings: Sequence[Mapping[str, Any]]
    ) -> Result[tuple[Id, ...], KernelError]: ...
    def list(self, **filter: Any) -> tuple[InspectionRecord, ...]: ...


InspectionToken: CapabilityToken[InspectionService] = create_capability_token("field.inspection")


@runtime_checkable
class InstallProgressService(Protocol):
    async def compute(
        self, package_id: Id, data_date: IsoTimestamp
    ) -> Result[InstallProgressRecord, KernelError]: ...
    def history(self, package_id: Id) -> tuple[InstallProgressRecord, ...]: ...
    async def earned_value(
        self, package_id: Id, data_date: IsoTimestamp
    ) -> Result[Money, KernelError]: ...


InstallProgressToken: CapabilityToken[InstallProgressService] = create_capability_token(
    "field.progress"
)


class PROCUREMENT_COMMANDS:
    create_package = "procurement.package.create"
    package_from_boq = "procurement.package.from-boq"
    award_package = "procurement.package.award"
    record_field_status = "field.status.record"
    create_inspection = "field.inspection.create"
    compute_progress = "field.progress.compute"


class PROCUREMENT_PERMISSIONS:
    manage_packages = "procurement.package.manage"
    award = "procurement.award"
    record_field = "field.record"
    inspect = "field.inspect"


class PROCUREMENT_EVENTS:
    package_created = "procurement.package.created"
    package_awarded = "procurement.package.awarded"
    field_status_recorded = "field.status.recorded"
    inspection_completed = "field.inspection.completed"
    progress_computed = "field.progress.computed"


#: States that count as work in place for earned value.
#:
#: ``installed`` counts and ``rework`` does not, deliberately: work that has to be redone has not
#: been earned, however complete it looked when it was installed.
EARNING_STATES: tuple[FieldState, ...] = ("installed", "inspected", "accepted")
