from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .common import ElementRef, Id, IsoTimestamp, Money, Provenance, UnitizedValue


@dataclass(frozen=True)
class QuantitySource:
    """Where a number came from.

    Carried rather than implied. A quantity read off the model and a quantity somebody typed in are
    both "1,240 m2" on screen, and the difference only becomes visible -- expensively -- when the
    model is re-issued and one of them silently fails to move.
    """

    kind: Literal["model-takeoff", "manual", "imported", "assumed"]
    #: Takeoff rule and its version, when the quantity was measured.
    rule_id: Id | None = None
    rule_version: int | None = None
    #: Model revision measured against -- what makes a re-run comparable.
    model_version: str | None = None
    entered_by: Id | None = None
    note: str | None = None


@dataclass(frozen=True)
class RateSource:
    """Where a rate came from. Same reasoning as ``QuantitySource``, applied to money."""

    kind: Literal["library", "assembly", "quotation", "manual", "benchmark", "assumed"]
    library_id: Id | None = None
    library_version: str | None = None
    assembly_id: Id | None = None
    vendor_id: Id | None = None
    quoted_at: IsoTimestamp | None = None
    entered_by: Id | None = None
    note: str | None = None


@dataclass(frozen=True)
class QuantityRecord:
    """A measured quantity taken from the model.

    ``elements`` is retained rather than just the total. An estimator's first question about any
    number is "which elements is that?", and a takeoff that cannot answer it is not auditable and
    will not be trusted.
    """

    id: Id
    model_id: Id
    #: What was measured, e.g. ``NetVolume``, ``GrossArea``, ``Length``, ``Count``.
    metric: str
    quantity: UnitizedValue
    #: Required: a quantity whose origin is unknown cannot be defended or re-measured.
    source: QuantitySource
    taken_at: IsoTimestamp
    elements: tuple[ElementRef, ...] = ()
    classification_code: str | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class TakeoffRuleRecord:
    """A rule that turns model elements into quantities.

    Versioned, because rates depend on it.
    """

    id: Id
    name: str
    version: int
    metric: str
    unit: str
    #: Element filter -- IFC class, property values, classification.
    filter: Mapping[str, Any] = field(default_factory=dict)
    #: Expression evaluated per element, e.g. ``Width * Height``.
    expression: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class ClassificationSystemRecord:
    id: Id
    #: e.g. ``Uniclass 2015``, ``MasterFormat``, ``NRM1``, ``OmniClass``.
    name: str
    version: str | None = None


@dataclass(frozen=True)
class ClassificationMappingRecord:
    """Maps model elements or takeoff output into an estimator's coding structure."""

    id: Id
    system_id: Id
    code: str
    title: str | None = None
    filter: Mapping[str, Any] | None = None
    quantity_ids: tuple[Id, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class ResourceRecord:
    id: Id
    name: str
    type: Literal["labour", "material", "plant", "subcontract", "other"]
    unit: str
    rate: Money
    effective_from: IsoTimestamp | None = None
    supplier: str | None = None


@dataclass(frozen=True)
class AssemblyComponent:
    resource_id: Id
    #: Resource units consumed per one unit of the assembly.
    factor: float
    waste_percent: float = 0.0


@dataclass(frozen=True)
class CostAssemblyRecord:
    """A composite unit rate: what it costs to do one unit of work, broken into resources."""

    id: Id
    code: str
    name: str
    unit: str
    components: tuple[AssemblyComponent, ...] = ()
    overhead_percent: float = 0.0
    profit_percent: float = 0.0
    library_id: Id | None = None
    version: int | None = None
    rate_source: RateSource | None = None


@dataclass(frozen=True)
class BoqLineRecord:
    id: Id
    boq_id: Id
    item_number: str
    description: str
    quantity: UnitizedValue
    classification_code: str | None = None
    assembly_id: Id | None = None
    rate: Money | None = None
    #: Present whenever ``rate`` is. A priced line with no rate origin is not reviewable.
    rate_source: RateSource | None = None
    total: Money | None = None
    #: Quantities this line was measured from -- the audit trail back to the model.
    quantity_ids: tuple[Id, ...] = ()
    #: Mirrors the quantities' origin so a line can be triaged without dereferencing each one.
    quantity_source: QuantitySource | None = None
    parent_id: Id | None = None


@dataclass(frozen=True)
class BoqRecord:
    id: Id
    name: str
    currency: str
    revision: str
    created_at: IsoTimestamp
    created_by: Id
    line_ids: tuple[Id, ...] = ()


EstimateStatus = Literal["draft", "issued", "superseded", "awarded"]


@dataclass(frozen=True)
class ModelVersionRef:
    model_id: Id
    version: str


@dataclass(frozen=True)
class EstimateRecord:
    id: Id
    name: str
    #: The bill this estimate reports. Once issued, this points at a **frozen copy** --
    #: regenerating the working bill afterwards must not rewrite a document somebody has already
    #: acted on.
    boq_id: Id
    status: EstimateStatus
    currency: str
    subtotal: Money
    total: Money
    created_at: IsoTimestamp
    created_by: Id
    #: The live bill it was generated from. Revisions re-price against this, not the frozen copy.
    working_boq_id: Id | None = None
    contingency_percent: float = 0.0
    #: Model revisions this estimate was measured against. Required for change control.
    basis_model_versions: tuple[ModelVersionRef, ...] = ()
    supersedes_id: Id | None = None


@dataclass(frozen=True)
class CashflowPeriodRecord:
    period_start: IsoTimestamp
    period_end: IsoTimestamp
    planned_spend: Money
    actual_spend: Money | None = None
    cumulative_planned: Money | None = None
    cumulative_actual: Money | None = None


@dataclass(frozen=True)
class CashflowForecastRecord:
    id: Id
    estimate_id: Id
    currency: str
    generated_at: IsoTimestamp
    #: Forecast is driven by the schedule; without it a cashflow is just a total in a bar chart.
    schedule_basis: Id | None = None
    periods: tuple[CashflowPeriodRecord, ...] = ()


@dataclass(frozen=True)
class QuantityDelta:
    metric: str
    delta: UnitizedValue


@dataclass(frozen=True)
class ChangeImpactRecord:
    """Cost consequence of a model change, linking a revision diff to money."""

    id: Id
    diff_id: Id
    estimate_id: Id
    delta_cost: Money
    status: Literal["identified", "estimated", "approved", "rejected"]
    identified_at: IsoTimestamp
    delta_quantities: tuple[QuantityDelta, ...] = ()
    #: Changed elements no priced line could be found for.
    #:
    #: Newly added elements have not been measured yet, so nothing prices them. Reporting them
    #: beats a delta that silently covers only the part of the change that happened to be
    #: measurable.
    unpriced_elements: tuple[ElementRef, ...] = ()
    schedule_impact_days: float | None = None
