"""Markup and estimating, and the composition between them."""

from __future__ import annotations

import pytest

from massingviser.plugins.estimating import (
    ESTIMATING_COMMANDS,
    BoqToken,
    EstimateToken,
    ModelElementSourceToken,
    QuantityTakeoffToken,
    ScheduleBasisToken,
    SchedulePeriod,
    TakeoffElement,
    estimating_plugin,
    evaluate_expression,
    multiply_money,
    percent_of,
    sum_money,
)
from massingviser.plugins.markup import (
    MARKUP_COMMANDS,
    AnchorToken,
    ElementResolverToken,
    IssueToken,
    markup_plugin,
)
from massingviser.schema import Money

# ---------------------------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------------------------


def test_money_is_integer_minor_units():
    with pytest.raises(TypeError):
        Money(12.5, "GBP")  # a float here is how a bill ends up a penny out


def test_money_rounds_half_away_from_zero_not_to_even():
    # Python's round() is banker's rounding: round(2.5) == 2. An estimator expects 3.
    assert multiply_money(Money(1, "GBP"), 2.5).amount_minor == 3
    assert multiply_money(Money(1, "GBP"), -2.5).amount_minor == -3


def test_mixing_currencies_is_refused():
    assert not sum_money([Money(1, "GBP"), Money(1, "USD")], "GBP").ok


def test_percent_of_is_exact():
    assert percent_of(Money(10_000, "GBP"), 7.5) == Money(750, "GBP")


# ---------------------------------------------------------------------------------------------
# The expression evaluator
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "variables", "expected"),
    [
        ("Width * Height", {"Width": 3, "Height": 4}, 12),
        ("2 + 3 * 4", {}, 14),
        ("(2 + 3) * 4", {}, 20),
        ("2 ^ 3 ^ 2", {}, 512),  # right-associative, as in maths
        ("-Width + 10", {"Width": 4}, 6),
        ("max(a, b) * 2", {"a": 3, "b": 7}, 14),
        ("round(Area, 1)", {"Area": 3.14159}, 3.1),
    ],
)
def test_expressions_evaluate(expression, variables, expected):
    result = evaluate_expression(expression, variables)
    assert result.ok and result.value == pytest.approx(expected)


@pytest.mark.parametrize(
    "hostile",
    [
        '__import__("os").system("id")',
        "Width.__class__.__mro__",
        "open('/etc/passwd')",
        "(lambda: 1)()",
        "eval('1+1')",
    ],
)
def test_the_evaluator_refuses_code(hostile):
    """A takeoff expression is *data* from a cost library or a saved project.

    Handing it to ``eval`` would give whoever wrote that file arbitrary code execution.
    """
    assert not evaluate_expression(hostile, {"Width": 1}).ok


def test_an_unknown_property_fails_loudly_rather_than_measuring_zero():
    result = evaluate_expression("Thickness * 2", {"Width": 1})
    assert not result.ok
    assert "Unknown property" in result.error.message


def test_division_by_zero_is_reported():
    assert not evaluate_expression("a / b", {"a": 1, "b": 0}).ok


# ---------------------------------------------------------------------------------------------
# Markup
# ---------------------------------------------------------------------------------------------


class _Resolver:
    def __init__(self, present: set[str]) -> None:
        self.present = present

    def exists(self, _model_id: str, global_id: str) -> bool:
        return global_id in self.present

    def global_ids(self, _model_id: str):
        return tuple(self.present)


async def test_a_markup_anchored_to_a_transient_id_is_refused(harness):
    await harness.load(markup_plugin)
    result = await harness.execute(MARKUP_COMMANDS.create, {"element_ids": [1234]})
    assert not result.ok
    assert "GlobalId" in result.error.message


async def test_a_deleted_element_orphans_its_markup_rather_than_moving_it(harness):
    await harness.load(markup_plugin)
    resolver = _Resolver({"WALL-1"})
    harness.kernel.capabilities.provide(ElementResolverToken, resolver)

    markup = (
        await harness.execute(
            MARKUP_COMMANDS.create, {"model_id": "m1", "element_ids": ["WALL-1"], "text": "check"}
        )
    ).value
    await harness.execute(
        MARKUP_COMMANDS.anchor,
        {"markup_id": markup.id, "element": {"model_id": "m1", "global_id": "WALL-1"}},
    )

    anchors = harness.capability(AnchorToken)
    assert (await anchors.reanchor("m1")).value.orphaned == ()

    resolver.present.clear()  # the next model revision deletes the wall
    report = (await anchors.reanchor("m1")).value
    assert report.orphaned == (markup.id,)
    assert [a.markup_id for a in anchors.orphaned()] == [markup.id]
    # The anchor keeps its GlobalId; it is marked unresolved, not relocated to the origin.
    assert anchors.resolve(markup.id).global_id == "WALL-1"


async def test_reanchoring_without_a_resolver_refuses_rather_than_guessing(harness):
    await harness.load(markup_plugin)
    result = await harness.execute(MARKUP_COMMANDS.reanchor, {"model_id": "m1"})
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"


@pytest.mark.parametrize(
    ("start", "target", "legal"),
    [
        ("open", "resolved", True),
        ("open", "in-review", True),
        ("open", "closed", False),  # skipping verification is the point of the state machine
        ("resolved", "closed", True),
        ("closed", "open", False),
        ("closed", "in-review", True),
    ],
)
async def test_issue_state_machine(harness, start, target, legal):
    await harness.load(markup_plugin)
    issues = harness.capability(IssueToken)
    issue = (await issues.create(title="I")).value

    # Walk to the starting state through legal moves.
    path = {
        "open": [],
        "in-review": ["in-review"],
        "resolved": ["resolved"],
        "closed": ["resolved", "closed"],
    }
    for step in path[start]:
        assert (await issues.transition(issue.id, step)).ok

    result = await issues.transition(issue.id, target)
    assert result.ok is legal
    if not legal:
        assert "cannot go from" in result.error.message


async def test_closing_an_issue_stamps_when(harness):
    await harness.load(markup_plugin)
    issues = harness.capability(IssueToken)
    issue = (await issues.create(title="I")).value
    await issues.transition(issue.id, "resolved")
    closed = (await issues.transition(issue.id, "closed", note="verified on site")).value
    assert closed.closed_at is not None


# ---------------------------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------------------------


class _Source:
    def __init__(self, count: int = 3) -> None:
        self.count = count

    def elements(self, _model_id: str):
        return [
            TakeoffElement(
                global_id=f"E{i}",
                ifc_class="IfcWall",
                properties={"Width": 0.2, "Height": 3.0, "Length": 5.0},
                classification_code="SUB-STRUCTURE",
            )
            for i in range(self.count)
        ]

    def model_ids(self):
        return ("m1",)

    def model_version(self, _model_id: str):
        return "rev-A"


class _Schedule:
    def periods(self, _unit: str = "month"):
        return (
            SchedulePeriod("2026-01-01", "2026-02-01", 0.3),
            SchedulePeriod("2026-02-01", "2026-03-01", 0.4),
            SchedulePeriod("2026-03-01", "2026-04-01", 0.3),
        )


async def _priced(harness, rate_minor: int = 42_000, contingency: float = 10.0):
    await harness.load(estimating_plugin)
    harness.kernel.capabilities.provide(ModelElementSourceToken, _Source())
    harness.kernel.capabilities.provide(ScheduleBasisToken, _Schedule())

    await harness.execute(
        ESTIMATING_COMMANDS.add_rule,
        {
            "id": "r1",
            "name": "Wall volume",
            "metric": "NetVolume",
            "unit": "m3",
            "filter": {"ifc_class": "IfcWall"},
            "expression": "Width * Height * Length",
        },
    )
    await harness.execute(
        ESTIMATING_COMMANDS.add_resource,
        {
            "id": "res",
            "name": "Concrete",
            "type": "material",
            "unit": "m3",
            "rate": Money(rate_minor, "GBP"),
        },
    )
    await harness.execute(
        ESTIMATING_COMMANDS.add_assembly,
        {
            "id": "asm",
            "code": "SUB-STRUCTURE",
            "name": "In-situ concrete",
            "unit": "m3",
            "components": [{"resource_id": "res", "factor": 1.0, "waste_percent": 5.0}],
            "overhead_percent": 10.0,
            "profit_percent": 5.0,
        },
    )
    await harness.execute(ESTIMATING_COMMANDS.run_takeoff, {})
    boq = (
        await harness.execute(ESTIMATING_COMMANDS.create_boq, {"name": "B", "currency": "GBP"})
    ).value
    await harness.execute(
        ESTIMATING_COMMANDS.generate_boq,
        {"boq_id": boq.id, "assembly_by_code": {"SUB-STRUCTURE": "asm"}},
    )
    estimate = (
        await harness.execute(
            ESTIMATING_COMMANDS.create_estimate,
            {"name": "E", "boq_id": boq.id, "contingency_percent": contingency},
        )
    ).value
    return boq, estimate


async def test_takeoff_without_a_source_says_so(harness):
    await harness.load(estimating_plugin)
    result = await harness.execute(ESTIMATING_COMMANDS.run_takeoff, {})
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"


async def test_every_quantity_records_its_provenance(harness):
    await _priced(harness)
    quantities = harness.capability(QuantityTakeoffToken).quantities()
    assert quantities
    for quantity in quantities:
        assert quantity.source.kind == "model-takeoff"
        assert quantity.source.rule_id == "r1"
        assert quantity.source.model_version == "rev-A"
        # The audit trail back to the model: which elements is that number?
        assert len(quantity.elements) == 3


async def test_a_rule_matching_nothing_is_reported_not_swallowed(harness):
    await harness.load(estimating_plugin)
    harness.kernel.capabilities.provide(ModelElementSourceToken, _Source())
    await harness.execute(
        ESTIMATING_COMMANDS.add_rule,
        {
            "id": "r-empty",
            "name": "Steel",
            "metric": "Mass",
            "unit": "kg",
            "filter": {"ifc_class": "IfcBeam"},
        },
    )
    summary = (await harness.execute(ESTIMATING_COMMANDS.run_takeoff, {})).value
    assert summary.empty_rules == ("r-empty",)


async def test_rerunning_a_takeoff_supersedes_rather_than_duplicates(harness):
    await _priced(harness)
    before = harness.capability(QuantityTakeoffToken).quantities()
    await harness.execute(ESTIMATING_COMMANDS.run_takeoff, {})
    after = harness.capability(QuantityTakeoffToken).quantities()
    assert len(before) == len(after) == 1


async def test_the_composite_unit_rate_compounds_waste_overhead_and_profit(harness):
    boq, _ = await _priced(harness)
    line = harness.capability(BoqToken).lines(boq.id)[0]
    # 420.00 x 1.05 waste x 1.10 overhead x 1.05 profit = 509.355 -> 509.36
    assert line.rate == Money(50_936, "GBP")
    assert line.rate_source.kind == "assembly"


async def test_a_priced_line_must_record_where_its_rate_came_from(harness):
    await harness.load(estimating_plugin)
    from massingviser.schema import UnitizedValue

    boq = (
        await harness.execute(ESTIMATING_COMMANDS.create_boq, {"name": "B", "currency": "GBP"})
    ).value
    result = await harness.capability(BoqToken).upsert_line(
        boq_id=boq.id,
        item_number="0001",
        description="unattributed",
        quantity=UnitizedValue(1.0, "m3"),
        rate=Money(1000, "GBP"),  # no rate_source
    )
    assert not result.ok and "where that rate came from" in result.error.message


async def test_an_issued_estimate_is_frozen_and_revising_supersedes(harness):
    _, estimate = await _priced(harness)
    assert estimate.status == "draft"

    issued = (
        await harness.execute(ESTIMATING_COMMANDS.issue_estimate, {"estimate_id": estimate.id})
    ).value
    assert issued.status == "issued"
    # The bill it reports is now a frozen copy, not the live one.
    assert issued.boq_id != issued.working_boq_id

    refused = await harness.execute(
        ESTIMATING_COMMANDS.recalculate_estimate, {"estimate_id": estimate.id}
    )
    assert not refused.ok and "frozen" in refused.error.message

    revision = (await harness.capability(EstimateToken).revise(estimate.id)).value
    assert revision.status == "draft" and revision.supersedes_id == estimate.id
    assert harness.capability(EstimateToken).get(estimate.id).status == "superseded"


async def test_an_estimate_records_the_model_revisions_it_was_measured_against(harness):
    _, estimate = await _priced(harness)
    assert [(v.model_id, v.version) for v in estimate.basis_model_versions] == [("m1", "rev-A")]


async def test_cashflow_sums_exactly_to_the_estimate(harness):
    _, estimate = await _priced(harness)
    forecast = (
        await harness.execute(ESTIMATING_COMMANDS.generate_cashflow, {"estimate_id": estimate.id})
    ).value
    # The last period absorbs the rounding remainder, so the page adds up.
    assert forecast.periods[-1].cumulative_planned == estimate.total
    assert (
        sum(p.planned_spend.amount_minor for p in forecast.periods) == estimate.total.amount_minor
    )


async def test_cashflow_without_a_schedule_refuses_rather_than_inventing_one(harness):
    await harness.load(estimating_plugin)
    harness.kernel.capabilities.provide(ModelElementSourceToken, _Source())
    boq = (
        await harness.execute(ESTIMATING_COMMANDS.create_boq, {"name": "B", "currency": "GBP"})
    ).value
    from massingviser.schema import UnitizedValue
    from massingviser.schema.cost import RateSource

    await harness.capability(BoqToken).upsert_line(
        boq_id=boq.id,
        item_number="0001",
        description="x",
        quantity=UnitizedValue(1.0, "m3"),
        rate=Money(1000, "GBP"),
        rate_source=RateSource(kind="manual"),
        total=Money(1000, "GBP"),
    )
    estimate = (
        await harness.execute(ESTIMATING_COMMANDS.create_estimate, {"name": "E", "boq_id": boq.id})
    ).value
    result = await harness.execute(
        ESTIMATING_COMMANDS.generate_cashflow, {"estimate_id": estimate.id}
    )
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"


# ---------------------------------------------------------------------------------------------
# The power operator
#
# `float(left ** right)` caught OverflowError and ValueError. Python does not raise for a negative
# base with a fractional exponent -- it returns a *complex* number, and `float()` of that raises
# TypeError, which escaped a function whose entire contract is to return a Result.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    ["(-8)^(1/3)", "(-2)^0.5", "(-1)^0.5", "0^-1", "0^-2.5"],
)
def test_a_power_with_no_real_value_is_reported_not_raised(expression):
    """`Depth ^ 0.5` on a negative quantity is a modelling mistake, not a crash."""
    result = evaluate_expression(expression, {})
    assert not result.ok
    assert result.error.code == "COMMAND_FAILED"


def test_a_negative_quantity_raised_to_a_half_names_the_values():
    result = evaluate_expression("Depth ^ 0.5", {"Depth": -4.0})
    assert not result.ok
    assert "no real value" in result.error.message
    assert result.error.details["base"] == -4.0


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(-8)^2", 64.0),  # an even integer power of a negative base is real
        ("(-8)^3", -512.0),  # and so is an odd one
        ("8^(1/3)", 2.0),
        ("2^-1", 0.5),
        ("0^0", 1.0),
        ("2^3^2", 512.0),  # right-associative
    ],
)
def test_powers_that_do_have_a_real_value_still_work(expression, expected):
    """The guard must not swallow the cases that were always fine."""
    result = evaluate_expression(expression, {})
    assert result.ok
    assert result.value == pytest.approx(expected)


def test_unary_minus_binds_tighter_than_the_power_operator():
    """Pinned because it is a real fork, and a silent disagreement about a sign is unfindable.

    Spreadsheets read `-2^2` as `(-2)^2 = 4`; Python and most mathematical writing read it as
    `-(2^2) = -4`. Estimators write these formulas in Excel all day, so the spreadsheet reading is
    the less surprising one here -- but only if it stays put.
    """
    assert evaluate_expression("-2^2", {}).value == pytest.approx(4.0)
    assert evaluate_expression("-(2^2)", {}).value == pytest.approx(-4.0)
    assert evaluate_expression("0-2^2", {}).value == pytest.approx(-4.0)
