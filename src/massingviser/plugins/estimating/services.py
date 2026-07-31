from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import (
    AssemblyComponent,
    BoqLineRecord,
    BoqRecord,
    CashflowForecastRecord,
    CashflowPeriodRecord,
    ClassificationMappingRecord,
    ClassificationSystemRecord,
    CostAssemblyRecord,
    ElementRef,
    EstimateRecord,
    Id,
    Money,
    QuantityRecord,
    QuantitySource,
    RateSource,
    ResourceRecord,
    TakeoffRuleRecord,
    UnitizedValue,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    ESTIMATING_EVENTS,
    ModelElementSourceToken,
    ScheduleBasisToken,
    TakeoffElement,
    TakeoffSummary,
)
from .math import (
    add_money,
    evaluate_expression,
    money,
    multiply_money,
    percent_of,
    sum_money,
)


@dataclass(frozen=True)
class EstimatingStores:
    rules: RecordStore[TakeoffRuleRecord]
    quantities: RecordStore[QuantityRecord]
    systems: RecordStore[ClassificationSystemRecord]
    mappings: RecordStore[ClassificationMappingRecord]
    resources: RecordStore[ResourceRecord]
    assemblies: RecordStore[CostAssemblyRecord]
    boqs: RecordStore[BoqRecord]
    lines: RecordStore[BoqLineRecord]
    estimates: RecordStore[EstimateRecord]
    cashflows: RecordStore[CashflowForecastRecord]


@dataclass(frozen=True)
class EstimatingRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_estimating_stores(context: PluginContext) -> EstimatingStores:
    return EstimatingStores(
        rules=create_record_store(context.state, "rules"),
        quantities=create_record_store(context.state, "quantities"),
        systems=create_record_store(context.state, "systems"),
        mappings=create_record_store(context.state, "mappings"),
        resources=create_record_store(context.state, "resources"),
        assemblies=create_record_store(context.state, "assemblies"),
        boqs=create_record_store(context.state, "boqs"),
        lines=create_record_store(context.state, "lines"),
        estimates=create_record_store(context.state, "estimates"),
        cashflows=create_record_store(context.state, "cashflows"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


def _matches(element: TakeoffElement, filter: Mapping[str, Any]) -> bool:
    """Element filter.

    ``ifc_class`` accepts a string or a list; property predicates compare exactly. Deliberately
    small -- a filter grammar that can express everything is a filter grammar nobody can debug when
    a rule quietly matches nothing.
    """
    wanted = filter.get("ifc_class")
    if wanted is not None:
        if isinstance(wanted, str):
            if element.ifc_class != wanted:
                return False
        elif element.ifc_class not in wanted:
            return False
    level = filter.get("level_global_id")
    if level is not None and element.level_global_id != level:
        return False
    code = filter.get("classification_code")
    if code is not None and element.classification_code != code:
        return False
    for key, value in (filter.get("properties") or {}).items():
        if element.properties.get(key) != value:
            return False
    return True


# ---------------------------------------------------------------------------------------------
# Takeoff
# ---------------------------------------------------------------------------------------------


class QuantityTakeoffServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: EstimatingRuntime, stores: EstimatingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def rules(self) -> tuple[TakeoffRuleRecord, ...]:
        return self._stores.rules.all()

    async def add_rule(self, **rule: Any) -> Result[TakeoffRuleRecord, KernelError]:
        rule_id = rule.get("id")
        if rule_id and self._stores.rules.has(rule_id):
            # Upsert. Re-running a setup path must not silently accumulate duplicate rules, which
            # would double-count every quantity they measure.
            updated = self._stores.rules.update(
                rule_id, {k: v for k, v in rule.items() if k != "id"}
            )
            return ok(updated) if updated else err(_not_found("takeoff rule", rule_id))

        record = TakeoffRuleRecord(
            id=rule_id or self._runtime.ids.next("rule"),
            name=rule["name"],
            version=rule.get("version", 1),
            metric=rule["metric"],
            unit=rule["unit"],
            filter=dict(rule.get("filter", {})),
            expression=rule.get("expression"),
            enabled=rule.get("enabled", True),
        )
        if record.expression:
            # Validated at registration, not at run time. A rule with a broken expression that
            # only fails halfway through a takeoff leaves a half-measured bill behind.
            probe = evaluate_expression(record.expression, {})
            if not probe.ok and "Unknown property" not in probe.error.message:
                return err(probe.error)
        self._stores.rules.add(record)
        return ok(record)

    def set_rule_enabled(self, rule_id: Id, enabled: bool) -> None:
        self._stores.rules.update(rule_id, {"enabled": enabled})

    async def run(
        self,
        *,
        model_ids: Sequence[Id] | None = None,
        rule_ids: Sequence[Id] | None = None,
    ) -> Result[TakeoffSummary, KernelError]:
        source = self._runtime.context.capabilities.get(ModelElementSourceToken)
        if source is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No model element source is installed, so there is nothing to measure. "
                    "A plugin must provide ModelElementSourceToken.",
                    {},
                )
            )

        rules = [
            rule
            for rule in self._stores.rules.all()
            if rule.enabled and (rule_ids is None or rule.id in rule_ids)
        ]
        models = list(model_ids) if model_ids is not None else list(source.model_ids())

        produced: list[QuantityRecord] = []
        empty_rules: list[Id] = []
        failures: list[tuple[str, str]] = []
        measured = 0
        taken_at = self._runtime.clock.iso()

        for rule in rules:
            matched_any = False
            for model_id in models:
                elements = [
                    element
                    for element in source.elements(model_id)
                    if _matches(element, rule.filter)
                ]
                if not elements:
                    continue
                matched_any = True

                total = 0.0
                refs: list[ElementRef] = []
                codes: set[str | None] = set()
                for element in elements:
                    if rule.expression:
                        evaluated = evaluate_expression(rule.expression, element.properties)
                        if not evaluated.ok:
                            # One unmeasurable element must not abort the trade. It is reported
                            # instead, because a quantity that silently skipped elements is worse
                            # than one that names them.
                            failures.append((element.global_id, evaluated.error.message))
                            continue
                        value = evaluated.value
                    else:
                        value = 1.0  # a rule with no expression is a count
                    total += value
                    refs.append(ElementRef(model_id=model_id, global_id=element.global_id))
                    codes.add(element.classification_code)
                    measured += 1

                if not refs:
                    continue

                # A quantity inherits its elements' classification only when they all agree.
                # Picking one of several would attach the whole measured volume to one cost code
                # and quietly misprice the rest; leaving it unset sends it to UNCLASSIFIED, where
                # it stays visible in the bill and can be mapped deliberately.
                inherited = codes.pop() if len(codes) == 1 else None
                codes.clear()

                produced.append(
                    QuantityRecord(
                        id=self._runtime.ids.next("qty"),
                        model_id=model_id,
                        metric=rule.metric,
                        quantity=UnitizedValue(total, rule.unit),
                        # Required, and populated from the rule that produced it -- this is the
                        # audit trail that makes the number defensible six months later.
                        source=QuantitySource(
                            kind="model-takeoff",
                            rule_id=rule.id,
                            rule_version=rule.version,
                            model_version=source.model_version(model_id),
                        ),
                        taken_at=taken_at,
                        elements=tuple(refs),
                        classification_code=inherited,
                    )
                )
            if not matched_any:
                empty_rules.append(rule.id)

        # A re-run supersedes the previous measurement for the rules that ran, rather than piling
        # duplicates on top of it.
        ran = {rule.id for rule in rules}
        self._stores.quantities.remove_where(
            lambda quantity: (
                quantity.source.rule_id in ran
                and (model_ids is None or quantity.model_id in models)
            )
        )
        self._stores.quantities.add_many(produced)

        summary = TakeoffSummary(
            quantities=tuple(produced),
            rules_run=len(rules),
            elements_measured=measured,
            empty_rules=tuple(empty_rules),
            failures=tuple(failures),
        )
        self._runtime.context.events.emit(ESTIMATING_EVENTS.takeoff_completed, {"summary": summary})
        return ok(summary)

    def quantities(self, **filter: Any) -> tuple[QuantityRecord, ...]:
        results = self._stores.quantities.all()
        metric = filter.get("metric")
        model_id = filter.get("model_id")
        code = filter.get("classification_code")
        return tuple(
            quantity
            for quantity in results
            if (metric is None or quantity.metric == metric)
            and (model_id is None or quantity.model_id == model_id)
            and (code is None or quantity.classification_code == code)
        )

    def elements_for(self, quantity_id: Id) -> tuple[ElementRef, ...]:
        quantity = self._stores.quantities.get(quantity_id)
        return quantity.elements if quantity else ()


# ---------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------


class ClassificationMappingServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: EstimatingRuntime, stores: EstimatingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def systems(self) -> tuple[ClassificationSystemRecord, ...]:
        return self._stores.systems.all()

    async def add_system(
        self, name: str, version: str | None = None
    ) -> Result[ClassificationSystemRecord, KernelError]:
        record = ClassificationSystemRecord(
            id=self._runtime.ids.next("clsys"), name=name, version=version
        )
        self._stores.systems.add(record)
        return ok(record)

    def mappings(self, system_id: Id | None = None) -> tuple[ClassificationMappingRecord, ...]:
        if system_id is None:
            return self._stores.mappings.all()
        return self._stores.mappings.query(lambda m: m.system_id == system_id)

    async def set_mapping(self, **mapping: Any) -> Result[ClassificationMappingRecord, KernelError]:
        mapping_id = mapping.get("id")
        if mapping_id and self._stores.mappings.has(mapping_id):
            updated = self._stores.mappings.update(
                mapping_id, {k: v for k, v in mapping.items() if k != "id"}
            )
            return ok(updated) if updated else err(_not_found("mapping", mapping_id))
        record = ClassificationMappingRecord(
            id=mapping_id or self._runtime.ids.next("clmap"),
            system_id=mapping["system_id"],
            code=mapping["code"],
            title=mapping.get("title"),
            filter=mapping.get("filter"),
            quantity_ids=tuple(mapping.get("quantity_ids", ())),
            confidence=mapping.get("confidence"),
        )
        self._stores.mappings.add(record)
        return ok(record)

    async def classify(
        self, system_id: Id, quantity_ids: Sequence[Id] | None = None
    ) -> Result[Mapping[str, Any], KernelError]:
        if not self._stores.systems.has(system_id):
            return err(_not_found("classification system", system_id))

        candidates = (
            self._stores.quantities.all()
            if quantity_ids is None
            else tuple(q for q in self._stores.quantities.all() if q.id in set(quantity_ids))
        )
        mappings = self.mappings(system_id)

        classified = 0
        unmatched: list[Id] = []
        for quantity in candidates:
            code = next(
                (
                    mapping.code
                    for mapping in mappings
                    if quantity.id in mapping.quantity_ids
                    or (mapping.filter or {}).get("metric") == quantity.metric
                ),
                None,
            )
            if code is None:
                # Reported, not guessed. An unclassified quantity is a visible gap; a
                # wrongly-classified one is a wrong bill that reconciles.
                unmatched.append(quantity.id)
                continue
            self._stores.quantities.update(quantity.id, {"classification_code": code})
            classified += 1
        return ok({"classified": classified, "unmatched": tuple(unmatched)})


# ---------------------------------------------------------------------------------------------
# Resources and assemblies
# ---------------------------------------------------------------------------------------------


class CostAssemblyServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: EstimatingRuntime, stores: EstimatingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def resources(self) -> tuple[ResourceRecord, ...]:
        return self._stores.resources.all()

    async def upsert_resource(self, **resource: Any) -> Result[ResourceRecord, KernelError]:
        resource_id = resource.get("id")
        if resource_id and self._stores.resources.has(resource_id):
            updated = self._stores.resources.update(
                resource_id, {k: v for k, v in resource.items() if k != "id"}
            )
            return ok(updated) if updated else err(_not_found("resource", resource_id))
        record = ResourceRecord(
            id=resource_id or self._runtime.ids.next("res"),
            name=resource["name"],
            type=resource.get("type", "material"),
            unit=resource["unit"],
            rate=resource["rate"],
            effective_from=resource.get("effective_from"),
            supplier=resource.get("supplier"),
        )
        self._stores.resources.add(record)
        return ok(record)

    def assemblies(self, **filter: Any) -> tuple[CostAssemblyRecord, ...]:
        library_id = filter.get("library_id")
        code = filter.get("code")
        return tuple(
            assembly
            for assembly in self._stores.assemblies.all()
            if (library_id is None or assembly.library_id == library_id)
            and (code is None or assembly.code == code)
        )

    async def upsert_assembly(self, **assembly: Any) -> Result[CostAssemblyRecord, KernelError]:
        components = tuple(
            component
            if isinstance(component, AssemblyComponent)
            else AssemblyComponent(**component)
            for component in assembly.get("components", ())
        )
        for component in components:
            if not self._stores.resources.has(component.resource_id):
                return err(_not_found("resource", component.resource_id))

        assembly_id = assembly.get("id")
        if assembly_id and self._stores.assemblies.has(assembly_id):
            changes = {k: v for k, v in assembly.items() if k != "id"}
            changes["components"] = components
            updated = self._stores.assemblies.update(assembly_id, changes)
            return ok(updated) if updated else err(_not_found("assembly", assembly_id))

        record = CostAssemblyRecord(
            id=assembly_id or self._runtime.ids.next("asm"),
            code=assembly["code"],
            name=assembly["name"],
            unit=assembly["unit"],
            components=components,
            overhead_percent=assembly.get("overhead_percent", 0.0),
            profit_percent=assembly.get("profit_percent", 0.0),
            library_id=assembly.get("library_id"),
            version=assembly.get("version"),
            rate_source=assembly.get("rate_source") or RateSource(kind="assembly"),
        )
        self._stores.assemblies.add(record)
        return ok(record)

    def unit_rate(self, assembly_id: Id) -> Result[Money, KernelError]:
        """The composite unit rate: resources, waste, overhead and profit.

        Every component is accumulated in minor units and rounded exactly once, at the end.
        Rounding each component before summing is how a rate ends up a penny out per line and a
        project's worth of lines ends up materially wrong.
        """
        assembly = self._stores.assemblies.get(assembly_id)
        if assembly is None:
            return err(_not_found("assembly", assembly_id))
        if not assembly.components:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Assembly "{assembly.code}" has no components, so it has no rate.',
                    {"assemblyId": assembly_id},
                )
            )

        currency: str | None = None
        subtotal = 0.0
        for component in assembly.components:
            resource = self._stores.resources.get(component.resource_id)
            if resource is None:
                return err(_not_found("resource", component.resource_id))
            if currency is None:
                currency = resource.rate.currency
            elif currency != resource.rate.currency:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'Assembly "{assembly.code}" mixes {currency} with '
                        f"{resource.rate.currency}.",
                        {"assemblyId": assembly_id},
                    )
                )
            subtotal += (
                resource.rate.amount_minor
                * component.factor
                * (1.0 + component.waste_percent / 100.0)
            )

        assert currency is not None
        with_overhead = subtotal * (1.0 + assembly.overhead_percent / 100.0)
        with_profit = with_overhead * (1.0 + assembly.profit_percent / 100.0)
        return ok(money(with_profit, currency))


# ---------------------------------------------------------------------------------------------
# Bills of quantities
# ---------------------------------------------------------------------------------------------


class BoqServiceImpl:
    __slots__ = ("_runtime", "_stores", "_assemblies")

    def __init__(
        self,
        runtime: EstimatingRuntime,
        stores: EstimatingStores,
        assemblies: CostAssemblyServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._assemblies = assemblies

    async def create(self, name: str, currency: str) -> Result[BoqRecord, KernelError]:
        record = BoqRecord(
            id=self._runtime.ids.next("boq"),
            name=name,
            currency=currency,
            revision="A",
            created_at=self._runtime.clock.iso(),
            created_by=self._runtime.context.permissions.identity.id,
        )
        self._stores.boqs.add(record)
        return ok(record)

    def lines(self, boq_id: Id) -> tuple[BoqLineRecord, ...]:
        return tuple(
            sorted(
                self._stores.lines.query(lambda line: line.boq_id == boq_id),
                key=lambda line: line.item_number,
            )
        )

    async def generate(self, boq_id: Id, **options: Any) -> Result[BoqRecord, KernelError]:
        """Build priced lines from measured quantities.

        One line per (classification code, metric) pair. Quantities with no classification are
        still billed, under an ``UNCLASSIFIED`` code -- dropping them would make the bill total
        quietly disagree with the model.
        """
        boq = self._stores.boqs.get(boq_id)
        if boq is None:
            return err(_not_found("boq", boq_id))

        quantities = self._stores.quantities.all()
        if not quantities:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "Nothing has been measured yet; run a takeoff before generating a bill.",
                    {"boqId": boq_id},
                )
            )

        rate_for_code: Mapping[str, Id] = options.get("assembly_by_code", {})

        grouped: dict[tuple[str, str, str], list[QuantityRecord]] = {}
        for quantity in quantities:
            key = (
                quantity.classification_code or "UNCLASSIFIED",
                quantity.metric,
                quantity.quantity.unit,
            )
            grouped.setdefault(key, []).append(quantity)

        self._stores.lines.remove_where(lambda line: line.boq_id == boq_id)

        lines: list[BoqLineRecord] = []
        for index, ((code, metric, unit), group) in enumerate(sorted(grouped.items()), start=1):
            total_quantity = sum(quantity.quantity.value for quantity in group)
            assembly_id = rate_for_code.get(code)
            rate: Money | None = None
            rate_source: RateSource | None = None
            line_total: Money | None = None

            if assembly_id:
                unit_rate = self._assemblies.unit_rate(assembly_id)
                if not unit_rate.ok:
                    return err(unit_rate.error)
                if unit_rate.value.currency != boq.currency:
                    return err(
                        KernelError(
                            "COMMAND_FAILED",
                            f"Assembly rate is {unit_rate.value.currency} but the bill is "
                            f"{boq.currency}.",
                            {"boqId": boq_id},
                        )
                    )
                rate = unit_rate.value
                rate_source = RateSource(kind="assembly", assembly_id=assembly_id)
                line_total = multiply_money(rate, total_quantity)

            # Mirrors the quantities' origin so a line can be triaged without dereferencing each
            # one. They all came from the same takeoff, so the first is representative.
            lines.append(
                BoqLineRecord(
                    id=self._runtime.ids.next("line"),
                    boq_id=boq_id,
                    item_number=f"{index:04d}",
                    description=f"{code} - {metric}",
                    classification_code=code,
                    quantity=UnitizedValue(total_quantity, unit),
                    assembly_id=assembly_id,
                    rate=rate,
                    rate_source=rate_source,
                    total=line_total,
                    quantity_ids=tuple(quantity.id for quantity in group),
                    quantity_source=group[0].source,
                )
            )

        self._stores.lines.add_many(lines)
        updated = self._stores.boqs.update(boq_id, {"line_ids": tuple(line.id for line in lines)})
        # Unpriced lines are counted and published. A bill whose total quietly covers only the
        # lines that happened to find a rate is the most expensive kind of wrong: it reconciles
        # against itself and understates the project.
        unpriced = tuple(line.classification_code for line in lines if line.total is None)
        self._runtime.context.events.emit(
            ESTIMATING_EVENTS.boq_generated,
            {"boqId": boq_id, "lines": len(lines), "unpricedCodes": unpriced},
        )
        return ok(updated or boq)

    async def upsert_line(self, **line: Any) -> Result[BoqLineRecord, KernelError]:
        line_id = line.get("id")
        if line.get("rate") is not None and line.get("rate_source") is None:
            # A priced line with no rate origin is not reviewable, so it is not accepted.
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "A line with a rate must record where that rate came from (rate_source).",
                    {"lineId": line_id},
                )
            )
        if line_id and self._stores.lines.has(line_id):
            updated = self._stores.lines.update(
                line_id, {k: v for k, v in line.items() if k != "id"}
            )
            return ok(updated) if updated else err(_not_found("boq line", line_id))
        record = BoqLineRecord(
            id=line_id or self._runtime.ids.next("line"),
            boq_id=line["boq_id"],
            item_number=line["item_number"],
            description=line["description"],
            quantity=line["quantity"],
            classification_code=line.get("classification_code"),
            assembly_id=line.get("assembly_id"),
            rate=line.get("rate"),
            rate_source=line.get("rate_source"),
            total=line.get("total"),
            quantity_ids=tuple(line.get("quantity_ids", ())),
            quantity_source=line.get("quantity_source"),
            parent_id=line.get("parent_id"),
        )
        self._stores.lines.add(record)
        return ok(record)

    async def remove_line(self, line_id: Id) -> Result[None, KernelError]:
        return (
            ok(None) if self._stores.lines.remove(line_id) else err(_not_found("boq line", line_id))
        )

    def total(self, boq_id: Id) -> Result[Money, KernelError]:
        boq = self._stores.boqs.get(boq_id)
        if boq is None:
            return err(_not_found("boq", boq_id))
        totals = [line.total for line in self.lines(boq_id) if line.total is not None]
        return sum_money(totals, boq.currency)


# ---------------------------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------------------------


class EstimateServiceImpl:
    __slots__ = ("_runtime", "_stores", "_boqs")

    def __init__(
        self, runtime: EstimatingRuntime, stores: EstimatingStores, boqs: BoqServiceImpl
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._boqs = boqs

    def _basis_versions(self) -> tuple[Any, ...]:
        from ...schema.cost import ModelVersionRef

        seen: dict[str, str] = {}
        for quantity in self._stores.quantities.all():
            version = quantity.source.model_version
            if version is not None:
                seen[quantity.model_id] = version
        return tuple(
            ModelVersionRef(model_id=model_id, version=version)
            for model_id, version in sorted(seen.items())
        )

    async def create(
        self, name: str, boq_id: Id, *, contingency_percent: float = 0.0
    ) -> Result[EstimateRecord, KernelError]:
        boq = self._stores.boqs.get(boq_id)
        if boq is None:
            return err(_not_found("boq", boq_id))
        subtotal = self._boqs.total(boq_id)
        if not subtotal.ok:
            return err(subtotal.error)

        total = add_money(subtotal.value, percent_of(subtotal.value, contingency_percent))
        if not total.ok:
            return err(total.error)

        record = EstimateRecord(
            id=self._runtime.ids.next("est"),
            name=name,
            boq_id=boq_id,
            working_boq_id=boq_id,
            status="draft",
            currency=boq.currency,
            subtotal=subtotal.value,
            total=total.value,
            created_at=self._runtime.clock.iso(),
            created_by=self._runtime.context.permissions.identity.id,
            contingency_percent=contingency_percent,
            # Required for change control: without the revisions it was measured against, an
            # estimate cannot be compared to the next one.
            basis_model_versions=self._basis_versions(),
        )
        self._stores.estimates.add(record)
        self._runtime.context.events.emit(ESTIMATING_EVENTS.estimate_changed, {"record": record})
        return ok(record)

    async def recalculate(self, estimate_id: Id) -> Result[EstimateRecord, KernelError]:
        estimate = self._stores.estimates.get(estimate_id)
        if estimate is None:
            return err(_not_found("estimate", estimate_id))
        if estimate.status in ("issued", "awarded"):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Estimate "{estimate.name}" is {estimate.status} and is frozen. '
                    "Revise it instead.",
                    {"estimateId": estimate_id, "status": estimate.status},
                )
            )

        subtotal = self._boqs.total(estimate.working_boq_id or estimate.boq_id)
        if not subtotal.ok:
            return err(subtotal.error)
        total = add_money(subtotal.value, percent_of(subtotal.value, estimate.contingency_percent))
        if not total.ok:
            return err(total.error)

        updated = self._stores.estimates.update(
            estimate_id,
            {
                "subtotal": subtotal.value,
                "total": total.value,
                "basis_model_versions": self._basis_versions(),
            },
        )
        self._runtime.context.events.emit(ESTIMATING_EVENTS.estimate_changed, {"record": updated})
        return ok(updated) if updated else err(_not_found("estimate", estimate_id))

    async def issue(self, estimate_id: Id) -> Result[EstimateRecord, KernelError]:
        """Freeze the bill behind an estimate.

        The estimate is repointed at a *copy* of the bill. Regenerating the working bill afterwards
        must not rewrite a document somebody has already tendered against.
        """
        estimate = self._stores.estimates.get(estimate_id)
        if estimate is None:
            return err(_not_found("estimate", estimate_id))
        if estimate.status != "draft":
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Only a draft can be issued; this one is {estimate.status}.",
                    {"estimateId": estimate_id, "status": estimate.status},
                )
            )

        working_id = estimate.working_boq_id or estimate.boq_id
        working = self._stores.boqs.get(working_id)
        if working is None:
            return err(_not_found("boq", working_id))

        frozen_id = self._runtime.ids.next("boq")
        frozen_lines = [
            replace(line, id=self._runtime.ids.next("line"), boq_id=frozen_id)
            for line in self._boqs.lines(working_id)
        ]
        self._stores.boqs.add(
            replace(
                working,
                id=frozen_id,
                name=f"{working.name} (issued)",
                revision=f"{working.revision}-issued",
                line_ids=tuple(line.id for line in frozen_lines),
            )
        )
        self._stores.lines.add_many(frozen_lines)

        updated = self._stores.estimates.update(
            estimate_id, {"status": "issued", "boq_id": frozen_id, "working_boq_id": working_id}
        )
        self._runtime.context.events.emit(ESTIMATING_EVENTS.estimate_issued, {"record": updated})
        return ok(updated) if updated else err(_not_found("estimate", estimate_id))

    async def revise(self, estimate_id: Id) -> Result[EstimateRecord, KernelError]:
        """Supersede an issued estimate with a fresh draft priced off the live bill."""
        estimate = self._stores.estimates.get(estimate_id)
        if estimate is None:
            return err(_not_found("estimate", estimate_id))

        working_id = estimate.working_boq_id or estimate.boq_id
        subtotal = self._boqs.total(working_id)
        if not subtotal.ok:
            return err(subtotal.error)
        total = add_money(subtotal.value, percent_of(subtotal.value, estimate.contingency_percent))
        if not total.ok:
            return err(total.error)

        revision = replace(
            estimate,
            id=self._runtime.ids.next("est"),
            status="draft",
            boq_id=working_id,
            working_boq_id=working_id,
            subtotal=subtotal.value,
            total=total.value,
            created_at=self._runtime.clock.iso(),
            basis_model_versions=self._basis_versions(),
            supersedes_id=estimate.id,
        )
        self._stores.estimates.add(revision)
        self._stores.estimates.update(estimate_id, {"status": "superseded"})
        self._runtime.context.events.emit(ESTIMATING_EVENTS.estimate_changed, {"record": revision})
        return ok(revision)

    def get(self, estimate_id: Id) -> EstimateRecord | None:
        return self._stores.estimates.get(estimate_id)

    def list(self) -> tuple[EstimateRecord, ...]:
        return self._stores.estimates.all()

    async def compare(self, a: Id, b: Id) -> Result[Mapping[str, Any], KernelError]:
        left = self._stores.estimates.get(a)
        right = self._stores.estimates.get(b)
        if left is None:
            return err(_not_found("estimate", a))
        if right is None:
            return err(_not_found("estimate", b))
        if left.currency != right.currency:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Cannot compare {left.currency} with {right.currency}.",
                    {},
                )
            )
        return ok(
            {
                "delta": Money(right.total.amount_minor - left.total.amount_minor, left.currency),
                "from": left,
                "to": right,
            }
        )


# ---------------------------------------------------------------------------------------------
# Cashflow
# ---------------------------------------------------------------------------------------------


class CashflowForecastServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: EstimatingRuntime, stores: EstimatingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def generate(
        self, estimate_id: Id, *, unit: str = "month"
    ) -> Result[CashflowForecastRecord, KernelError]:
        estimate = self._stores.estimates.get(estimate_id)
        if estimate is None:
            return err(_not_found("estimate", estimate_id))

        basis = self._runtime.context.capabilities.get(ScheduleBasisToken)
        if basis is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "A cashflow needs a schedule to spread the money over. Install a plugin "
                    "providing ScheduleBasisToken (4D planning) first.",
                    {"estimateId": estimate_id},
                )
            )
        periods = list(basis.periods(unit))
        if not periods:
            return err(
                KernelError(
                    "COMMAND_FAILED", "The schedule basis produced no periods.", {"unit": unit}
                )
            )

        weight_total = sum(period.weight for period in periods)
        if weight_total <= 0:
            return err(KernelError("COMMAND_FAILED", "Schedule period weights sum to zero.", {}))

        rows: list[CashflowPeriodRecord] = []
        cumulative = 0
        # The last period absorbs the rounding remainder, so the cashflow sums to the estimate
        # exactly rather than being a few pence out at the bottom of the page.
        allocated = 0
        for index, period in enumerate(periods):
            if index == len(periods) - 1:
                amount = estimate.total.amount_minor - allocated
            else:
                amount = multiply_money(estimate.total, period.weight / weight_total).amount_minor
            allocated += amount
            cumulative += amount
            rows.append(
                CashflowPeriodRecord(
                    period_start=period.start,
                    period_end=period.end,
                    planned_spend=Money(amount, estimate.currency),
                    cumulative_planned=Money(cumulative, estimate.currency),
                )
            )

        record = CashflowForecastRecord(
            id=self._runtime.ids.next("cf"),
            estimate_id=estimate_id,
            currency=estimate.currency,
            generated_at=self._runtime.clock.iso(),
            periods=tuple(rows),
        )
        self._stores.cashflows.add(record)
        self._runtime.context.events.emit(ESTIMATING_EVENTS.cashflow_generated, {"record": record})
        return ok(record)

    async def record_actual(
        self, forecast_id: Id, period_start: str, actual: Money
    ) -> Result[CashflowForecastRecord, KernelError]:
        forecast = self._stores.cashflows.get(forecast_id)
        if forecast is None:
            return err(_not_found("cashflow forecast", forecast_id))
        if actual.currency != forecast.currency:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Actual is {actual.currency} but the forecast is {forecast.currency}.",
                    {},
                )
            )

        cumulative = 0
        rows: list[CashflowPeriodRecord] = []
        matched = False
        for period in forecast.periods:
            value = actual if period.period_start == period_start else period.actual_spend
            if period.period_start == period_start:
                matched = True
            if value is not None:
                cumulative += value.amount_minor
            rows.append(
                replace(
                    period,
                    actual_spend=value,
                    cumulative_actual=(
                        Money(cumulative, forecast.currency) if value is not None else None
                    ),
                )
            )
        if not matched:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'No period starting "{period_start}" in this forecast.',
                    {"forecastId": forecast_id},
                )
            )

        updated = self._stores.cashflows.update(forecast_id, {"periods": tuple(rows)})
        return ok(updated) if updated else err(_not_found("cashflow forecast", forecast_id))

    def get(self, forecast_id: Id) -> CashflowForecastRecord | None:
        return self._stores.cashflows.get(forecast_id)
