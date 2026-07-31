"""``massingviser.plugins.estimating`` -- 5D: quantities, classification, rates, bills, money.

The through-line is **provenance**. Every quantity records how it was measured and every rate
records where it came from, because the expensive failure in estimating is not arithmetic -- it is
a number nobody can trace when the model is re-issued and half the bill silently fails to move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    BoqLineRecord,
    BoqRecord,
    CashflowForecastRecord,
    ChangeImpactRecord,
    ClassificationMappingRecord,
    ClassificationSystemRecord,
    CostAssemblyRecord,
    ElementRef,
    EstimateRecord,
    Id,
    Money,
    QuantityRecord,
    ResourceRecord,
    TakeoffRuleRecord,
)


@dataclass(frozen=True)
class TakeoffElement:
    """One element as the estimator sees it.

    Properties are a flat scalar map on purpose: takeoff expressions address them by name, and a
    nested structure would need a path grammar that every rule author then has to learn.
    """

    global_id: str
    ifc_class: str
    properties: Mapping[str, float] = field(default_factory=dict)
    level_global_id: str | None = None
    classification_code: str | None = None


@runtime_checkable
class ModelElementSource(Protocol):
    """Where elements come from.

    Estimating never imports a viewer or a geometry kernel. Whatever holds the model -- a
    fragments viewer, an IFC parser, or the massing plugin -- provides this instead.
    """

    def elements(self, model_id: Id) -> Sequence[TakeoffElement]: ...
    def model_ids(self) -> Sequence[Id]: ...
    #: Revision the elements came from. Required for a re-run to be comparable.
    def model_version(self, model_id: Id) -> str | None: ...


ModelElementSourceToken: CapabilityToken[ModelElementSource] = create_capability_token(
    "estimating.element-source"
)


@dataclass(frozen=True)
class SchedulePeriod:
    start: str
    end: str
    #: 0..1 share of the project's work planned in this period.
    weight: float


@runtime_checkable
class ScheduleBasisSource(Protocol):
    def periods(self, unit: Literal["week", "month", "quarter"]) -> Sequence[SchedulePeriod]: ...


ScheduleBasisToken: CapabilityToken[ScheduleBasisSource] = create_capability_token(
    "estimating.schedule-basis"
)


@dataclass(frozen=True)
class TakeoffSummary:
    quantities: tuple[QuantityRecord, ...]
    rules_run: int
    elements_measured: int
    #: Rules that matched no element. Reported rather than swallowed: a rule matching nothing is
    #: almost always a filter typo, and the resulting bill looks complete while missing a trade.
    empty_rules: tuple[Id, ...] = ()
    #: Elements a rule matched but could not measure, with the reason.
    failures: tuple[tuple[str, str], ...] = ()


@runtime_checkable
class QuantityTakeoffService(Protocol):
    def rules(self) -> tuple[TakeoffRuleRecord, ...]: ...
    async def add_rule(self, **rule: Any) -> Result[TakeoffRuleRecord, KernelError]: ...
    def set_rule_enabled(self, rule_id: Id, enabled: bool) -> None: ...
    async def run(
        self, *, model_ids: Sequence[Id] | None = None, rule_ids: Sequence[Id] | None = None
    ) -> Result[TakeoffSummary, KernelError]: ...
    def quantities(self, **filter: Any) -> tuple[QuantityRecord, ...]: ...
    def elements_for(self, quantity_id: Id) -> tuple[ElementRef, ...]: ...


QuantityTakeoffToken: CapabilityToken[QuantityTakeoffService] = create_capability_token(
    "estimating.takeoff"
)


@runtime_checkable
class ClassificationMappingService(Protocol):
    def systems(self) -> tuple[ClassificationSystemRecord, ...]: ...
    async def add_system(
        self, name: str, version: str | None = None
    ) -> Result[ClassificationSystemRecord, KernelError]: ...
    def mappings(self, system_id: Id | None = None) -> tuple[ClassificationMappingRecord, ...]: ...
    async def set_mapping(self, **mapping: Any) -> Result[ClassificationMappingRecord, KernelError]: ...
    async def classify(
        self, system_id: Id, quantity_ids: Sequence[Id] | None = None
    ) -> Result[Mapping[str, Any], KernelError]: ...


ClassificationMappingToken: CapabilityToken[ClassificationMappingService] = create_capability_token(
    "estimating.classification"
)


@runtime_checkable
class CostAssemblyService(Protocol):
    def resources(self) -> tuple[ResourceRecord, ...]: ...
    async def upsert_resource(self, **resource: Any) -> Result[ResourceRecord, KernelError]: ...
    def assemblies(self, **filter: Any) -> tuple[CostAssemblyRecord, ...]: ...
    async def upsert_assembly(self, **assembly: Any) -> Result[CostAssemblyRecord, KernelError]: ...
    #: The composite unit rate: resources, waste, overhead and profit, computed once.
    def unit_rate(self, assembly_id: Id) -> Result[Money, KernelError]: ...


CostAssemblyToken: CapabilityToken[CostAssemblyService] = create_capability_token(
    "estimating.assemblies"
)


@runtime_checkable
class BoqService(Protocol):
    async def create(self, name: str, currency: str) -> Result[BoqRecord, KernelError]: ...
    async def generate(self, boq_id: Id, **options: Any) -> Result[BoqRecord, KernelError]: ...
    def lines(self, boq_id: Id) -> tuple[BoqLineRecord, ...]: ...
    async def upsert_line(self, **line: Any) -> Result[BoqLineRecord, KernelError]: ...
    async def remove_line(self, line_id: Id) -> Result[None, KernelError]: ...
    def total(self, boq_id: Id) -> Result[Money, KernelError]: ...


BoqToken: CapabilityToken[BoqService] = create_capability_token("estimating.boq")


@runtime_checkable
class EstimateService(Protocol):
    async def create(
        self, name: str, boq_id: Id, *, contingency_percent: float = 0.0
    ) -> Result[EstimateRecord, KernelError]: ...
    async def recalculate(self, estimate_id: Id) -> Result[EstimateRecord, KernelError]: ...
    #: Freezes the bill. An issued estimate must not change under somebody who has acted on it.
    async def issue(self, estimate_id: Id) -> Result[EstimateRecord, KernelError]: ...
    async def revise(self, estimate_id: Id) -> Result[EstimateRecord, KernelError]: ...
    def get(self, estimate_id: Id) -> EstimateRecord | None: ...
    def list(self) -> tuple[EstimateRecord, ...]: ...
    async def compare(self, a: Id, b: Id) -> Result[Mapping[str, Any], KernelError]: ...


EstimateToken: CapabilityToken[EstimateService] = create_capability_token("estimating.estimate")


@runtime_checkable
class CashflowForecastService(Protocol):
    async def generate(
        self, estimate_id: Id, *, unit: Literal["week", "month", "quarter"] = "month"
    ) -> Result[CashflowForecastRecord, KernelError]: ...
    async def record_actual(
        self, forecast_id: Id, period_start: str, actual: Money
    ) -> Result[CashflowForecastRecord, KernelError]: ...
    def get(self, forecast_id: Id) -> CashflowForecastRecord | None: ...


CashflowForecastToken: CapabilityToken[CashflowForecastService] = create_capability_token(
    "estimating.cashflow"
)


class ESTIMATING_COMMANDS:
    add_rule = "estimating.takeoff.add-rule"
    run_takeoff = "estimating.takeoff.run"
    add_resource = "estimating.assembly.add-resource"
    add_assembly = "estimating.assembly.add"
    create_boq = "estimating.boq.create"
    generate_boq = "estimating.boq.generate"
    create_estimate = "estimating.estimate.create"
    recalculate_estimate = "estimating.estimate.recalculate"
    issue_estimate = "estimating.estimate.issue"
    generate_cashflow = "estimating.cashflow.generate"


class ESTIMATING_PERMISSIONS:
    measure = "estimating.measure"
    price = "estimating.price"
    issue = "estimating.issue"


class ESTIMATING_EVENTS:
    takeoff_completed = "estimating.takeoff.completed"
    boq_generated = "estimating.boq.generated"
    estimate_changed = "estimating.estimate.changed"
    estimate_issued = "estimating.estimate.issued"
    cashflow_generated = "estimating.cashflow.generated"
