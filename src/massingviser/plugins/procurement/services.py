from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import (
    ElementRef,
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
    element_key,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    EARNING_STATES,
    PROCUREMENT_EVENTS,
    BoqLineSourceToken,
    VendorComparison,
)


@dataclass(frozen=True)
class ProcurementStores:
    packages: RecordStore[ProcurementPackageRecord]
    vendors: RecordStore[VendorRecord]
    scopes: RecordStore[VendorScopeRecord]
    statuses: RecordStore[FieldStatusRecord]
    inspections: RecordStore[InspectionRecord]
    progress: RecordStore[InstallProgressRecord]


@dataclass(frozen=True)
class ProcurementRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_procurement_stores(context: PluginContext) -> ProcurementStores:
    return ProcurementStores(
        packages=create_record_store(context.state, "packages"),
        vendors=create_record_store(context.state, "vendors"),
        scopes=create_record_store(context.state, "scopes"),
        statuses=create_record_store(context.state, "statuses"),
        inspections=create_record_store(context.state, "inspections"),
        progress=create_record_store(context.state, "progress"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


#: Package lifecycle. Awarding is a one-way door and reopening a tender is a new package.
_PACKAGE_TRANSITIONS: Mapping[str, tuple[str, ...]] = {
    "draft": ("issued",),
    "issued": ("tendering", "draft"),
    "tendering": ("awarded", "issued"),
    "awarded": ("in-progress",),
    "in-progress": ("complete",),
    "complete": (),
}


class PackageServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: ProcurementRuntime, stores: ProcurementStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def create(self, **pkg: Any) -> Result[ProcurementPackageRecord, KernelError]:
        code = pkg.get("code")
        if not code:
            return err(KernelError("COMMAND_FAILED", "A package needs a code.", {}))
        if self._stores.packages.find(lambda p: p.code == code):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'A package with code "{code}" already exists. Codes appear on orders and '
                    "invoices, so two packages sharing one is a reconciliation problem.",
                    {"code": code},
                )
            )
        record = ProcurementPackageRecord(
            id=pkg.get("id") or self._runtime.ids.next("pkg"),
            code=code,
            name=pkg["name"],
            status=pkg.get("status", "draft"),
            created_at=self._runtime.clock.iso(),
            boq_line_ids=tuple(pkg.get("boq_line_ids", ())),
            elements=tuple(pkg.get("elements", ())),
            task_ids=tuple(pkg.get("task_ids", ())),
            budget=pkg.get("budget"),
            vendor_id=pkg.get("vendor_id"),
            required_on_site=pkg.get("required_on_site"),
            lead_time_days=pkg.get("lead_time_days"),
        )
        self._stores.packages.add(record)
        self._runtime.context.events.emit(
            PROCUREMENT_EVENTS.package_created, {"record": record}
        )
        return ok(record)

    async def update(
        self, package_id: Id, changes: Mapping[str, Any]
    ) -> Result[ProcurementPackageRecord, KernelError]:
        # Status and award value move through their own guarded paths.
        safe = {k: v for k, v in changes.items() if k not in ("status", "awarded_value", "id")}
        updated = self._stores.packages.update(package_id, safe)
        return ok(updated) if updated else err(_not_found("package", package_id))

    async def set_status(
        self, package_id: Id, status: PackageStatus
    ) -> Result[ProcurementPackageRecord, KernelError]:
        existing = self._stores.packages.get(package_id)
        if existing is None:
            return err(_not_found("package", package_id))
        if status == existing.status:
            return ok(existing)
        allowed = _PACKAGE_TRANSITIONS.get(existing.status, ())
        if status not in allowed:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'A package cannot go from "{existing.status}" to "{status}". '
                    f"Allowed: {', '.join(allowed) or 'none'}.",
                    {"packageId": package_id, "from": existing.status, "to": status},
                )
            )
        updated = self._stores.packages.update(package_id, {"status": status})
        return ok(updated) if updated else err(_not_found("package", package_id))

    async def from_boq_lines(
        self, boq_line_ids: Sequence[Id], name: str, code: str
    ) -> Result[ProcurementPackageRecord, KernelError]:
        """Build a package from priced lines, carrying the budget across.

        The budget is the sum of the lines it was built from, not a number somebody types. That is
        what makes the eventual award comparable to the estimate.
        """
        source = self._runtime.context.capabilities.get(BoqLineSourceToken)
        if source is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No bill of quantities is available to build a package from.",
                    {},
                )
            )
        lines = list(source.lines(boq_line_ids))
        if not lines:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "None of those line ids resolved to a priced line.",
                    {"lineIds": list(boq_line_ids)},
                )
            )

        priced = [line for line in lines if line.total is not None]
        budget: Money | None = None
        if priced:
            currency = priced[0].total.currency  # type: ignore[union-attr]
            mixed = [line.id for line in priced if line.total.currency != currency]  # type: ignore[union-attr]
            if mixed:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        "The selected lines are priced in more than one currency.",
                        {"lines": mixed},
                    )
                )
            budget = Money(sum(line.total.amount_minor for line in priced), currency)  # type: ignore[union-attr]

        return await self.create(
            code=code,
            name=name,
            boq_line_ids=tuple(line.id for line in lines),
            budget=budget,
        )

    def list(self, **filter: Any) -> tuple[ProcurementPackageRecord, ...]:
        status = filter.get("status")
        vendor_id = filter.get("vendor_id")
        return self._stores.packages.query(
            lambda p: (status is None or p.status == status)
            and (vendor_id is None or p.vendor_id == vendor_id)
        )

    def get(self, package_id: Id) -> ProcurementPackageRecord | None:
        return self._stores.packages.get(package_id)

    async def uncovered_scope(self) -> Result[tuple[Id, ...], KernelError]:
        source = self._runtime.context.capabilities.get(BoqLineSourceToken)
        if source is None:
            return err(
                KernelError("CAPABILITY_NOT_FOUND", "No bill of quantities is available.", {})
            )
        covered = {
            line_id for package in self._stores.packages.all() for line_id in package.boq_line_ids
        }
        return ok(tuple(line.id for line in source.lines() if line.id not in covered))


class VendorScopeServiceImpl:
    __slots__ = ("_runtime", "_stores", "_packages")

    def __init__(
        self,
        runtime: ProcurementRuntime,
        stores: ProcurementStores,
        packages: PackageServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._packages = packages

    def vendors(self) -> tuple[VendorRecord, ...]:
        return self._stores.vendors.all()

    async def upsert_vendor(self, **vendor: Any) -> Result[VendorRecord, KernelError]:
        vendor_id = vendor.get("id")
        if vendor_id and self._stores.vendors.has(vendor_id):
            updated = self._stores.vendors.update(
                vendor_id, {k: v for k, v in vendor.items() if k != "id"}
            )
            return ok(updated) if updated else err(_not_found("vendor", vendor_id))
        record = VendorRecord(
            id=vendor_id or self._runtime.ids.next("vendor"),
            name=vendor["name"],
            trade=vendor.get("trade"),
            contact_email=vendor.get("contact_email"),
            prequalified=vendor.get("prequalified", False),
            rating=vendor.get("rating"),
        )
        self._stores.vendors.add(record)
        return ok(record)

    def scopes(self, package_id: Id) -> tuple[VendorScopeRecord, ...]:
        return self._stores.scopes.query(lambda s: s.package_id == package_id)

    async def submit_scope(self, **scope: Any) -> Result[VendorScopeRecord, KernelError]:
        package_id = scope["package_id"]
        if not self._stores.packages.has(package_id):
            return err(_not_found("package", package_id))
        if not self._stores.vendors.has(scope["vendor_id"]):
            return err(_not_found("vendor", scope["vendor_id"]))

        record = VendorScopeRecord(
            id=scope.get("id") or self._runtime.ids.next("scope"),
            package_id=package_id,
            vendor_id=scope["vendor_id"],
            inclusions=tuple(scope.get("inclusions", ())),
            exclusions=tuple(scope.get("exclusions", ())),
            quoted_value=scope.get("quoted_value"),
            submitted_at=scope.get("submitted_at") or self._runtime.clock.iso(),
        )
        # One live scope per vendor per package; a resubmission replaces the previous bid.
        self._stores.scopes.remove_where(
            lambda s: s.package_id == package_id and s.vendor_id == record.vendor_id
        )
        self._stores.scopes.add(record)
        return ok(record)

    async def compare(self, package_id: Id) -> Result[tuple[VendorComparison, ...], KernelError]:
        """Rank bids, and say when a ranking is not trustworthy.

        A bid that excludes more than its rivals is not comparable on price alone, and the single
        most expensive procurement mistake is awarding on a number that covers less work. The
        caveat is attached rather than the bid being silently reordered.
        """
        package = self._stores.packages.get(package_id)
        if package is None:
            return err(_not_found("package", package_id))

        scopes = self.scopes(package_id)
        if not scopes:
            return ok(())

        exclusion_counts = [len(scope.exclusions) for scope in scopes]
        fewest_exclusions = min(exclusion_counts)

        comparisons: list[VendorComparison] = []
        for scope in scopes:
            vendor = self._stores.vendors.get(scope.vendor_id)
            variance: Money | None = None
            if scope.quoted_value is not None and package.budget is not None:
                if scope.quoted_value.currency != package.budget.currency:
                    return err(
                        KernelError(
                            "COMMAND_FAILED",
                            f"Bid is {scope.quoted_value.currency} but the budget is "
                            f"{package.budget.currency}.",
                            {"packageId": package_id, "vendorId": scope.vendor_id},
                        )
                    )
                variance = Money(
                    scope.quoted_value.amount_minor - package.budget.amount_minor,
                    package.budget.currency,
                )

            caveat = None
            if len(scope.exclusions) > fewest_exclusions:
                caveat = (
                    f"excludes {len(scope.exclusions) - fewest_exclusions} more item(s) than the "
                    "most inclusive bid -- not comparable on price alone"
                )
            elif scope.quoted_value is None:
                caveat = "no price submitted"

            comparisons.append(
                VendorComparison(
                    vendor_id=scope.vendor_id,
                    vendor_name=vendor.name if vendor else scope.vendor_id,
                    quoted=scope.quoted_value,
                    variance=variance,
                    inclusions=len(scope.inclusions),
                    exclusions=len(scope.exclusions),
                    caveat=caveat,
                )
            )

        # Unpriced bids sort last rather than first, which is what a naive ascending sort on None
        # would do.
        comparisons.sort(
            key=lambda c: (c.quoted is None, c.quoted.amount_minor if c.quoted else 0)
        )
        return ok(tuple(comparisons))

    async def award(
        self, package_id: Id, vendor_id: Id, value: Money
    ) -> Result[ProcurementPackageRecord, KernelError]:
        package = self._stores.packages.get(package_id)
        if package is None:
            return err(_not_found("package", package_id))
        if package.status == "awarded" or package.vendor_id:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Package "{package.code}" is already awarded to {package.vendor_id}. '
                    "Re-awarding is a new package, not an edit.",
                    {"packageId": package_id},
                )
            )
        if not self._stores.vendors.has(vendor_id):
            return err(_not_found("vendor", vendor_id))
        if package.budget is not None and value.currency != package.budget.currency:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Award is {value.currency} but the budget is {package.budget.currency}.",
                    {"packageId": package_id},
                )
            )

        if package.status == "draft":
            issued = await self._packages.set_status(package_id, "issued")
            if not issued.ok:
                return err(issued.error)
        if self._stores.packages.get(package_id).status == "issued":  # type: ignore[union-attr]
            tendering = await self._packages.set_status(package_id, "tendering")
            if not tendering.ok:
                return err(tendering.error)

        updated = self._stores.packages.update(
            package_id, {"status": "awarded", "vendor_id": vendor_id, "awarded_value": value}
        )
        self._runtime.context.events.emit(
            PROCUREMENT_EVENTS.package_awarded,
            {"packageId": package_id, "vendorId": vendor_id, "value": value},
        )
        return ok(updated) if updated else err(_not_found("package", package_id))


class FieldStatusServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: ProcurementRuntime, stores: ProcurementStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def record(self, **status: Any) -> Result[FieldStatusRecord, KernelError]:
        element = status["element"]
        if not isinstance(element, ElementRef):
            element = ElementRef(**dict(element))
        record = FieldStatusRecord(
            id=self._runtime.ids.next("field"),
            element=element,
            state=status["state"],
            observed_at=status.get("observed_at") or self._runtime.clock.iso(),
            observed_by=status.get("observed_by")
            or self._runtime.context.permissions.identity.id,
            package_id=status.get("package_id"),
            task_id=status.get("task_id"),
            quantity_installed=status.get("quantity_installed"),
            unit=status.get("unit"),
            photo_uris=tuple(status.get("photo_uris", ())),
            notes=status.get("notes"),
        )
        # History is append-only. Superseding rather than overwriting is what lets a progress claim
        # be re-checked against what was actually observed, and when.
        self._stores.statuses.add(record)
        self._runtime.context.events.emit(
            PROCUREMENT_EVENTS.field_status_recorded, {"record": record}
        )
        return ok(record)

    async def record_many(
        self, statuses: Sequence[Mapping[str, Any]]
    ) -> Result[int, KernelError]:
        count = 0
        for status in statuses:
            result = await self.record(**dict(status))
            if not result.ok:
                return err(result.error)
            count += 1
        return ok(count)

    def current(self, element: ElementRef) -> FieldStatusRecord | None:
        key = element_key(element)
        matching = [
            (record.observed_at, index, record)
            for index, record in enumerate(self._stores.statuses.all())
            if element_key(record.element) == key
        ]
        if not matching:
            return None
        # Ties break on insertion order, not arbitrarily. `datetime.now()` resolves to about 15 ms
        # on Windows, so two observations recorded in the same breath -- an inspection failing an
        # element that was marked installed moments earlier -- share a timestamp exactly. A plain
        # `max` on the timestamp returns the *first* of those, which is the superseded state, and
        # the defect silently fails to move earned value.
        return max(matching, key=lambda entry: (entry[0], entry[1]))[2]

    def query(self, **filter: Any) -> tuple[FieldStatusRecord, ...]:
        package_id = filter.get("package_id")
        state = filter.get("state")
        task_id = filter.get("task_id")
        latest_only = filter.get("latest_only", True)

        records = self._stores.statuses.query(
            lambda r: (package_id is None or r.package_id == package_id)
            and (state is None or r.state == state)
            and (task_id is None or r.task_id == task_id)
        )
        if not latest_only:
            return records
        # Same tie-breaking as `current`: later insertion wins when timestamps collide.
        newest: dict[str, tuple[str, int, FieldStatusRecord]] = {}
        order = {id(record): index for index, record in enumerate(self._stores.statuses.all())}
        for record in records:
            key = element_key(record.element)
            entry = (record.observed_at, order.get(id(record), 0), record)
            if key not in newest or entry[:2] > newest[key][:2]:
                newest[key] = entry
        return tuple(entry[2] for entry in newest.values())


class InspectionServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: ProcurementRuntime, stores: ProcurementStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def create(self, **inspection: Any) -> Result[InspectionRecord, KernelError]:
        elements = tuple(
            e if isinstance(e, ElementRef) else ElementRef(**dict(e))
            for e in inspection.get("elements", ())
        )
        record = InspectionRecord(
            id=inspection.get("id") or self._runtime.ids.next("insp"),
            name=inspection["name"],
            outcome=inspection.get("outcome", "not-inspected"),
            inspected_at=inspection.get("inspected_at") or self._runtime.clock.iso(),
            inspected_by=inspection.get("inspected_by")
            or self._runtime.context.permissions.identity.id,
            checklist_id=inspection.get("checklist_id"),
            package_id=inspection.get("package_id"),
            elements=elements,
            signature_uri=inspection.get("signature_uri"),
        )
        self._stores.inspections.add(record)
        self._runtime.context.events.emit(
            PROCUREMENT_EVENTS.inspection_completed, {"record": record}
        )
        return ok(record)

    async def fail(
        self, inspection_id: Id, findings: Sequence[Mapping[str, Any]]
    ) -> Result[tuple[Id, ...], KernelError]:
        """Fail an inspection, raising one issue per finding.

        Routed through the command bus, so a deployment without markup reports that it cannot
        raise the issue rather than recording a failure nobody is told about.
        """
        inspection = self._stores.inspections.get(inspection_id)
        if inspection is None:
            return err(_not_found("inspection", inspection_id))
        if not findings:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "A failed inspection with no findings records nothing actionable.",
                    {"inspectionId": inspection_id},
                )
            )
        if not self._runtime.context.commands.has("markup.issue.create"):
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No markup capability is installed, so a defect cannot be raised as an issue.",
                    {"inspectionId": inspection_id},
                )
            )

        issue_ids: list[Id] = []
        for finding in findings:
            created = await self._runtime.context.commands.execute(
                "markup.issue.create",
                {
                    "title": f"{inspection.name}: {finding['note']}",
                    "priority": finding.get("priority", "high"),
                    "responsibility": finding.get("responsibility"),
                    "labels": ("defect", "inspection"),
                },
            )
            if not created.ok:
                return err(created.error)
            issue_ids.append(created.value.id)

            element = finding.get("element")
            if element is not None:
                # A failed element goes back to rework, so earned value drops with the defect.
                reference = (
                    element if isinstance(element, ElementRef) else ElementRef(**dict(element))
                )
                self._stores.statuses.add(
                    FieldStatusRecord(
                        id=self._runtime.ids.next("field"),
                        element=reference,
                        state="rework",
                        observed_at=self._runtime.clock.iso(),
                        observed_by=inspection.inspected_by,
                        package_id=inspection.package_id,
                        notes=finding["note"],
                    )
                )

        self._stores.inspections.update(
            inspection_id, {"outcome": "fail", "issue_ids": tuple(issue_ids)}
        )
        return ok(tuple(issue_ids))

    def list(self, **filter: Any) -> tuple[InspectionRecord, ...]:
        package_id = filter.get("package_id")
        outcome = filter.get("outcome")
        return self._stores.inspections.query(
            lambda i: (package_id is None or i.package_id == package_id)
            and (outcome is None or i.outcome == outcome)
        )


class InstallProgressServiceImpl:
    __slots__ = ("_runtime", "_stores", "_field", "_packages")

    def __init__(
        self,
        runtime: ProcurementRuntime,
        stores: ProcurementStores,
        field_status: FieldStatusServiceImpl,
        packages: PackageServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._field = field_status
        self._packages = packages

    async def compute(
        self, package_id: Id, data_date: IsoTimestamp
    ) -> Result[InstallProgressRecord, KernelError]:
        package = self._stores.packages.get(package_id)
        if package is None:
            return err(_not_found("package", package_id))
        if not package.elements:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Package "{package.code}" has no elements, so there is nothing to measure '
                    "progress against. A percentage without a denominator is an opinion.",
                    {"packageId": package_id},
                )
            )

        total = len(package.elements)
        installed = 0
        quantity_installed = 0.0
        unit: str | None = None

        for element in package.elements:
            current = self._field.current(element)
            if current is None or current.state not in EARNING_STATES:
                continue
            installed += 1
            if current.quantity_installed is not None:
                quantity_installed += current.quantity_installed
                unit = unit or current.unit

        record = InstallProgressRecord(
            id=self._runtime.ids.next("prog"),
            package_id=package_id,
            data_date=data_date,
            quantity_installed=quantity_installed if unit else float(installed),
            quantity_total=float(total),
            unit=unit or "each",
            percent_complete=installed / total,
            earned_value=(
                Money(
                    round(package.awarded_value.amount_minor * installed / total),
                    package.awarded_value.currency,
                )
                if package.awarded_value is not None
                else None
            ),
        )
        self._stores.progress.remove_where(
            lambda p: p.package_id == package_id and p.data_date == data_date
        )
        self._stores.progress.add(record)
        self._runtime.context.events.emit(
            PROCUREMENT_EVENTS.progress_computed, {"record": record}
        )
        return ok(record)

    def history(self, package_id: Id) -> tuple[InstallProgressRecord, ...]:
        return tuple(
            sorted(
                self._stores.progress.query(lambda p: p.package_id == package_id),
                key=lambda p: p.data_date,
            )
        )

    async def earned_value(
        self, package_id: Id, data_date: IsoTimestamp
    ) -> Result[Money, KernelError]:
        computed = await self.compute(package_id, data_date)
        if not computed.ok:
            return err(computed.error)
        if computed.value.earned_value is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "Earned value needs an awarded value; an unawarded package has no rate to "
                    "earn against.",
                    {"packageId": package_id},
                )
            )
        return ok(computed.value.earned_value)
