"""The delivery-side families: coordination, 4D planning, procurement and field.

Each test names the property that makes the family worth having. Clash detection whose re-run
discards triage is worse than none; a 4D link that dies on re-issue means nobody re-issues; an
earned-value claim that ignores rework is a claim nobody can defend.
"""

from __future__ import annotations

import pytest

from massingviser.plugins.coordination import (
    ClashEngineToken,
    ClashToken,
    IssueRoutingToken,
    ModelSnapshotToken,
    RawClash,
    ResponsibilityToken,
    RevisionDiffToken,
    SnapshotElement,
    ValidationFinding,
    ValidationRuleToken,
    ValidationToken,
    clash_signature,
    coordination_plugin,
)
from massingviser.plugins.markup import IssueToken, markup_plugin
from massingviser.plugins.planning import (
    ElementFilterSourceToken,
    PlannedActualToken,
    ScheduleImportToken,
    TaskModelLinkToken,
    TimelinePlaybackToken,
    parse_timestamp,
    planning_plugin,
)
from massingviser.plugins.procurement import (
    BoqLineSourceToken,
    FieldStatusToken,
    InspectionToken,
    InstallProgressToken,
    PackageBoqLine,
    PackageToken,
    VendorScopeToken,
    procurement_plugin,
)
from massingviser.schema import ElementRef, Money, ValidationRuleRecord

# ---------------------------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------------------------


class _Engine:
    def __init__(self, pairs=(("W1", "P1"), ("W2", "P2"), ("W3", "P3"))) -> None:
        self.pairs = list(pairs)

    def intersect(self, a, b, kind, tolerance):
        return [
            RawClash(a=ElementRef("m1", x), b=ElementRef("m1", y), distance=0.05)
            for x, y in self.pairs
        ]


def test_a_clash_signature_is_independent_of_which_selection_each_element_landed_in():
    """The property the whole re-run cycle rests on."""
    left = ElementRef("m1", "WALL-1")
    right = ElementRef("m2", "PIPE-9")
    assert clash_signature("t1", left, right) == clash_signature("t1", right, left)
    # Different test, different clash -- the same pair in another test is a separate decision.
    assert clash_signature("t1", left, right) != clash_signature("t2", left, right)


async def test_a_clearance_test_with_no_tolerance_is_refused(harness):
    await harness.load(coordination_plugin)
    result = await harness.capability(ClashToken).define_test(
        name="Clearance", kind="clearance", tolerance=0.0
    )
    assert not result.ok and "wearing the wrong name" in result.error.message


async def test_running_without_an_engine_says_which_capability_is_missing(harness):
    await harness.load(coordination_plugin)
    clashes = harness.capability(ClashToken)
    test = (await clashes.define_test(name="T", kind="hard")).value
    result = await clashes.run(test.id)
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"


async def test_a_rerun_carries_triage_forward_and_ages_out_what_is_gone(harness):
    await harness.load(coordination_plugin)
    engine = _Engine()
    harness.kernel.capabilities.provide(ClashEngineToken, engine)
    clashes = harness.capability(ClashToken)

    test = (await clashes.define_test(name="MEP vs Structure", kind="hard")).value
    first = (await clashes.run(test.id)).value
    assert (first.created, first.persisted, first.resolved) == (3, 0, 0)

    found = {clash.a.global_id: clash for clash in clashes.results(test.id)}
    await clashes.set_status(found["W1"].id, "approved")
    await clashes.set_status(found["W2"].id, "reviewed")

    engine.pairs = [("W1", "P1"), ("W2", "P2"), ("W4", "P4")]  # W3 fixed, W4 new
    second = (await clashes.run(test.id)).value
    assert (second.created, second.persisted, second.resolved) == (1, 2, 1)

    states = {clash.a.global_id: clash.status for clash in clashes.results(test.id)}
    # A week of triage survives the re-run. This is the whole point.
    assert states == {"W1": "approved", "W2": "reviewed", "W3": "resolved", "W4": "new"}


async def test_a_fixed_clash_is_resolved_not_deleted(harness):
    await harness.load(coordination_plugin)
    engine = _Engine([("W1", "P1")])
    harness.kernel.capabilities.provide(ClashEngineToken, engine)
    clashes = harness.capability(ClashToken)
    test = (await clashes.define_test(name="T", kind="hard")).value
    await clashes.run(test.id)

    engine.pairs = []
    await clashes.run(test.id)
    # The record that it existed, and what was decided about it, is the audit trail.
    assert [c.status for c in clashes.results(test.id)] == ["resolved"]


async def test_promoting_a_clash_raises_one_issue_however_often_it_is_called(harness):
    await harness.load(markup_plugin, coordination_plugin)
    harness.kernel.capabilities.provide(ClashEngineToken, _Engine([("W1", "P1")]))
    clashes = harness.capability(ClashToken)
    test = (await clashes.define_test(name="T", kind="hard")).value
    await clashes.run(test.id)
    clash = clashes.results(test.id)[0]

    first = (await clashes.promote_to_issue(clash.id)).value
    second = (await clashes.promote_to_issue(clash.id)).value
    assert first == second
    assert len(harness.capability(IssueToken).query()) == 1


async def test_promoting_without_markup_reports_rather_than_crashing(harness):
    await harness.load(coordination_plugin)
    harness.kernel.capabilities.provide(ClashEngineToken, _Engine([("W1", "P1")]))
    clashes = harness.capability(ClashToken)
    test = (await clashes.define_test(name="T", kind="hard")).value
    await clashes.run(test.id)
    result = await clashes.promote_to_issue(clashes.results(test.id)[0].id)
    assert not result.ok and result.error.code == "CAPABILITY_NOT_FOUND"


class _Snapshots:
    def __init__(self) -> None:
        self.data = {
            "A": (
                SnapshotElement("E1", "IfcWall", {"Fire": "60"}, (0, 0, 0), {"Volume": 10.0}),
                SnapshotElement("E2", "IfcWall", {}, (5, 0, 0), {"Volume": 4.0}),
                SnapshotElement("E3", "IfcSlab", {}, (0, 0, 0), {"Volume": 20.0}),
            ),
            "B": (
                SnapshotElement("E1", "IfcWall", {"Fire": "120"}, (0, 0, 0), {"Volume": 12.0}),
                SnapshotElement("E2", "IfcWall", {}, (9, 0, 0), {"Volume": 4.0}),
                SnapshotElement("E4", "IfcWall", {}, (0, 3, 0), {"Volume": 7.0}),
            ),
        }

    def snapshot(self, model_id, version):
        return self.data.get(version) if model_id == "m1" else None

    def model_ids(self):
        return ("m1",)

    def versions(self, model_id):
        return ("A", "B")


async def test_a_revision_diff_separates_moved_from_modified(harness):
    await harness.load(coordination_plugin)
    harness.kernel.capabilities.provide(ModelSnapshotToken, _Snapshots())
    diff = (await harness.capability(RevisionDiffToken).compare("m1", "A", "B")).value

    kinds = {entry.element.global_id: entry.kind for entry in diff.entries}
    assert kinds == {"E1": "modified", "E2": "moved", "E3": "removed", "E4": "added"}

    modified = next(e for e in diff.entries if e.element.global_id == "E1")
    assert modified.changed_properties == ("Fire",)
    assert modified.quantity_delta["Volume"] == pytest.approx(2.0)

    # A removal carries a negative delta so the 5D hand-off can just add them up.
    removed = next(e for e in diff.entries if e.element.global_id == "E3")
    assert removed.quantity_delta["Volume"] == pytest.approx(-20.0)


async def test_diffing_a_missing_revision_names_it(harness):
    await harness.load(coordination_plugin)
    harness.kernel.capabilities.provide(ModelSnapshotToken, _Snapshots())
    result = await harness.capability(RevisionDiffToken).compare("m1", "A", "Z")
    assert not result.ok and "Z" in result.error.message


class _Rule:
    descriptor = ValidationRuleRecord(
        id="rule.fire", name="Walls declare fire rating", severity="error"
    )

    def check(self, model_id, elements):
        return [
            ValidationFinding("no fire rating", ElementRef(model_id, element.global_id))
            for element in elements
            if element.ifc_class == "IfcWall" and "Fire" not in element.properties
        ]


class _BrokenRule:
    descriptor = ValidationRuleRecord(id="rule.broken", name="Explodes", severity="warning")

    def check(self, model_id, elements):
        raise RuntimeError("rule is broken")


async def test_validation_aggregates_contributed_rules(harness):
    await harness.load(coordination_plugin)
    harness.kernel.capabilities.provide(ModelSnapshotToken, _Snapshots())
    harness.kernel.capabilities.provide(ValidationRuleToken, _Rule())

    summary = (await harness.capability(ValidationToken).run()).value
    assert summary.rules_run == 1
    assert summary.errors == 2  # E2 and E4 are walls with no fire rating in revision B


async def test_one_broken_rule_does_not_stop_the_others(harness):
    await harness.load(coordination_plugin)
    harness.kernel.capabilities.provide(ModelSnapshotToken, _Snapshots())
    harness.kernel.capabilities.provide(ValidationRuleToken, _Rule())
    harness.kernel.capabilities.provide(ValidationRuleToken, _BrokenRule())

    summary = (await harness.capability(ValidationToken).run()).value
    assert summary.errors == 2  # the good rule still ran
    assert [rule_id for rule_id, _ in summary.failed_rules] == ["rule.broken"]


async def test_a_disabled_rule_is_skipped(harness):
    await harness.load(coordination_plugin)
    harness.kernel.capabilities.provide(ModelSnapshotToken, _Snapshots())
    validation = harness.capability(ValidationToken)
    harness.kernel.capabilities.provide(ValidationRuleToken, _Rule())

    validation.set_enabled("rule.fire", False)
    assert (await validation.run()).value.rules_run == 0


async def test_issues_route_by_discipline_and_unrouted_ones_are_named(harness):
    await harness.load(markup_plugin, coordination_plugin)
    issues = harness.capability(IssueToken)
    routing = harness.capability(IssueRoutingToken)
    matrix = harness.capability(ResponsibilityToken)

    await matrix.upsert(scope="Services", discipline="MEP", responsible_id="alice")
    await routing.add_rule(discipline="MEP")

    mine = (await issues.create(title="Duct clash", responsibility="MEP")).value
    theirs = (await issues.create(title="Cladding", responsibility="Facade")).value

    summary = (await routing.route()).value
    assert summary.routed == 1
    assert summary.unrouted == (theirs.id,)
    assert issues.get(mine.id).assignee == "alice"


async def test_a_routing_rule_matching_nothing_is_refused(harness):
    await harness.load(markup_plugin, coordination_plugin)
    result = await harness.capability(IssueRoutingToken).add_rule()
    assert not result.ok and "capture every issue" in result.error.message


# ---------------------------------------------------------------------------------------------
# 4D planning
# ---------------------------------------------------------------------------------------------

SCHEDULE = """id,name,planned_start,planned_finish,percent_complete,predecessor
T1,Substructure,2026-01-01,2026-02-01,1.0,
T2,Frame,2026-02-01,2026-05-01,0.4,T1
T3,Envelope,2026-05-01,2026-08-01,0.0,T2
"""


class _Filter:
    def __init__(self) -> None:
        self.elements = [("W1", "IfcWall"), ("W2", "IfcWall"), ("S1", "IfcSlab")]

    def match(self, model_id, filter):
        wanted = (filter or {}).get("ifc_class")
        return [
            ElementRef(model_id, global_id)
            for global_id, ifc_class in self.elements
            if wanted is None or ifc_class == wanted
        ]

    def model_ids(self):
        return ("m1",)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2026-01-01", True), ("2026-01-01T09:00:00Z", True), ("not a date", False), ("", False)],
)
def test_timestamp_parsing_tolerates_real_exports(value, expected):
    assert (parse_timestamp(value) is not None) is expected


async def test_unreadable_rows_are_named_not_dropped(harness):
    await harness.load(planning_plugin)
    payload = SCHEDULE + "BAD,,2026-01-01,2026-02-01,,\nWORSE,Backwards,2026-06-01,2026-01-01,,\n"
    summary = (await harness.capability(ScheduleImportToken).import_schedule(payload, "csv")).value
    assert summary.tasks == 3
    reasons = dict(summary.rejected)
    assert "no name" in reasons["BAD"]
    # A task that finishes before it starts poisons every duration computed from it.
    assert "finishes before it starts" in reasons["WORSE"]


async def test_a_csv_missing_a_required_column_is_refused(harness):
    await harness.load(planning_plugin)
    result = await harness.capability(ScheduleImportToken).import_schedule("id,name\nT1,x\n", "csv")
    assert not result.ok and "planned_start" in result.error.message


async def test_reimporting_keeps_the_model_links(harness):
    """The operation that decides whether 4D survives contact with a project."""
    await harness.load(planning_plugin)
    harness.kernel.capabilities.provide(ElementFilterSourceToken, _Filter())
    schedule = harness.capability(ScheduleImportToken)
    links = harness.capability(TaskModelLinkToken)

    await schedule.import_schedule(SCHEDULE, "csv")
    await links.link_by_rule("T2", "m1", {"ifc_class": "IfcWall"}, "construct")
    assert len(links.links("T2")) == 1

    summary = (await schedule.reimport(SCHEDULE, "csv")).value
    assert (summary.added, summary.removed) == (0, 0)
    assert summary.orphaned_links == ()
    assert len(links.links("T2")) == 1  # the link survived a re-issue


async def test_a_link_whose_task_vanished_is_reported_not_deleted(harness):
    await harness.load(planning_plugin)
    harness.kernel.capabilities.provide(ElementFilterSourceToken, _Filter())
    schedule = harness.capability(ScheduleImportToken)
    links = harness.capability(TaskModelLinkToken)

    await schedule.import_schedule(SCHEDULE, "csv")
    link = (await links.link_by_rule("T3", "m1", {"ifc_class": "IfcSlab"}, "construct")).value

    shortened = "\n".join(SCHEDULE.strip().splitlines()[:3]) + "\n"  # drop T3
    summary = (await schedule.reimport(shortened, "csv")).value
    assert summary.removed == 1
    assert summary.orphaned_links == (link.id,)


async def test_a_rule_link_reresolves_against_the_new_revision(harness):
    await harness.load(planning_plugin)
    filter_source = _Filter()
    harness.kernel.capabilities.provide(ElementFilterSourceToken, filter_source)
    await harness.capability(ScheduleImportToken).import_schedule(SCHEDULE, "csv")
    links = harness.capability(TaskModelLinkToken)

    link = (await links.link_by_rule("T2", "m1", {"ifc_class": "IfcWall"}, "construct")).value
    assert len(link.elements) == 2

    filter_source.elements.append(("W3", "IfcWall"))  # the next revision adds a wall
    assert (await links.reresolve()).value.resolved == 1
    assert len(links.links("T2")[0].elements) == 3


async def test_a_rule_that_now_matches_nothing_keeps_its_link_and_reports(harness):
    await harness.load(planning_plugin)
    filter_source = _Filter()
    harness.kernel.capabilities.provide(ElementFilterSourceToken, filter_source)
    await harness.capability(ScheduleImportToken).import_schedule(SCHEDULE, "csv")
    links = harness.capability(TaskModelLinkToken)
    link = (await links.link_by_rule("T2", "m1", {"ifc_class": "IfcWall"}, "construct")).value

    filter_source.elements = [("S1", "IfcSlab")]
    summary = (await links.reresolve()).value
    assert summary.unmatched == (link.id,)
    assert len(links.links("T2")) == 1  # kept, so the plan's coverage is not silently dropped


async def test_unlinked_elements_are_the_4d_equivalent_of_an_unpriced_line(harness):
    await harness.load(planning_plugin)
    harness.kernel.capabilities.provide(ElementFilterSourceToken, _Filter())
    await harness.capability(ScheduleImportToken).import_schedule(SCHEDULE, "csv")
    links = harness.capability(TaskModelLinkToken)
    await links.link_by_rule("T2", "m1", {"ifc_class": "IfcWall"}, "construct")

    unlinked = (await links.unlinked_elements("m1")).value
    assert [element.global_id for element in unlinked] == ["S1"]


async def test_playback_groups_elements_by_what_the_task_does_to_them(harness):
    await harness.load(planning_plugin)
    harness.kernel.capabilities.provide(ElementFilterSourceToken, _Filter())
    await harness.capability(ScheduleImportToken).import_schedule(SCHEDULE, "csv")
    links = harness.capability(TaskModelLinkToken)
    playback = harness.capability(TimelinePlaybackToken)

    await links.link_by_rule("T2", "m1", {"ifc_class": "IfcWall"}, "construct")
    await links.link_by_rule("T1", "m1", {"ifc_class": "IfcSlab"}, "temporary")

    # T2 (Frame) finishes 2026-05-01; T1 (Substructure) runs Jan-Feb.
    mid = (await playback.state_at("2026-03-01")).value
    assert len(mid["construct"]) == 0 and len(mid["temporary"]) == 0

    during = (await playback.state_at("2026-01-15")).value
    assert len(during["temporary"]) == 1  # present only while its task runs

    later = (await playback.state_at("2026-06-01")).value
    assert len(later["construct"]) == 2


async def test_seeking_to_an_unreadable_date_is_refused(harness):
    await harness.load(planning_plugin)
    result = await harness.capability(TimelinePlaybackToken).seek("whenever")
    assert not result.ok


async def test_progress_variance_is_expressed_against_the_task_duration(harness):
    await harness.load(planning_plugin)
    await harness.capability(ScheduleImportToken).import_schedule(SCHEDULE, "csv")
    progress = harness.capability(PlannedActualToken)

    records = {r.task_id: r for r in (await progress.compare("2026-04-01")).value}
    assert records["T1"].planned_percent == pytest.approx(1.0)
    assert records["T3"].planned_percent == pytest.approx(0.0)
    # T2 is 2/3 through its window but only 40% done, so it is behind.
    assert records["T2"].planned_percent == pytest.approx(0.663, abs=1e-3)
    assert records["T2"].schedule_variance_days < 0

    behind = (await progress.behind_schedule("2026-04-01", threshold_days=5)).value
    assert [r.task_id for r in behind] == ["T2"]


# ---------------------------------------------------------------------------------------------
# Procurement and field
# ---------------------------------------------------------------------------------------------


class _Bill:
    LINES = (
        PackageBoqLine("L1", "Frame", Money(1_000_000, "GBP")),
        PackageBoqLine("L2", "Cladding", Money(500_000, "GBP")),
        PackageBoqLine("L3", "Roof", Money(200_000, "GBP")),
    )

    def lines(self, line_ids=None):
        return [line for line in self.LINES if line_ids is None or line.id in set(line_ids)]


@pytest.fixture()
async def procurement(harness):
    await harness.load(markup_plugin, procurement_plugin)
    harness.kernel.capabilities.provide(BoqLineSourceToken, _Bill())
    return harness


async def test_a_package_takes_its_budget_from_the_lines_it_was_built_from(harness):
    await harness.load(markup_plugin, procurement_plugin)
    harness.kernel.capabilities.provide(BoqLineSourceToken, _Bill())
    packages = harness.capability(PackageToken)

    package = (await packages.from_boq_lines(["L1", "L2"], "Superstructure", "PKG-01")).value
    # Not a number somebody typed -- which is what makes the eventual award comparable.
    assert package.budget == Money(1_500_000, "GBP")
    assert (await packages.uncovered_scope()).value == ("L3",)


async def test_two_packages_cannot_share_a_code(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    await packages.create(code="PKG-01", name="First")
    result = await packages.create(code="PKG-01", name="Second")
    assert not result.ok and "already exists" in result.error.message


async def test_a_package_cannot_skip_its_lifecycle(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    package = (await packages.create(code="P", name="n")).value
    result = await packages.set_status(package.id, "complete")
    assert not result.ok and "cannot go from" in result.error.message


async def test_a_bid_excluding_more_is_flagged_rather_than_silently_winning(harness):
    await harness.load(markup_plugin, procurement_plugin)
    harness.kernel.capabilities.provide(BoqLineSourceToken, _Bill())
    packages = harness.capability(PackageToken)
    vendors = harness.capability(VendorScopeToken)

    package = (await packages.from_boq_lines(["L1", "L2"], "Super", "PKG-01")).value
    for name, quote, exclusions in (
        ("Alpha", 1_400_000, ()),
        ("Beta", 1_300_000, ("temporary works", "craneage")),
        ("Gamma", 1_550_000, ()),
    ):
        vendor = (await vendors.upsert_vendor(name=name)).value
        await vendors.submit_scope(
            package_id=package.id,
            vendor_id=vendor.id,
            quoted_value=Money(quote, "GBP"),
            exclusions=exclusions,
        )

    ranked = (await vendors.compare(package.id)).value
    assert [c.vendor_name for c in ranked] == ["Beta", "Alpha", "Gamma"]
    # Cheapest, and not comparable -- said out loud rather than reordered behind the user's back.
    assert "not comparable on price alone" in ranked[0].caveat
    assert ranked[1].caveat is None
    assert ranked[2].variance == Money(50_000, "GBP")


async def test_resubmitting_replaces_a_bid_rather_than_stacking(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    vendors = harness.capability(VendorScopeToken)
    package = (await packages.create(code="P", name="n")).value
    vendor = (await vendors.upsert_vendor(name="Alpha")).value

    await vendors.submit_scope(
        package_id=package.id, vendor_id=vendor.id, quoted_value=Money(100, "GBP")
    )
    await vendors.submit_scope(
        package_id=package.id, vendor_id=vendor.id, quoted_value=Money(90, "GBP")
    )
    scopes = vendors.scopes(package.id)
    assert len(scopes) == 1 and scopes[0].quoted_value == Money(90, "GBP")


async def test_awarding_twice_is_refused(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    vendors = harness.capability(VendorScopeToken)
    package = (await packages.create(code="P", name="n", budget=Money(1000, "GBP"))).value
    alpha = (await vendors.upsert_vendor(name="Alpha")).value
    beta = (await vendors.upsert_vendor(name="Beta")).value

    assert (await vendors.award(package.id, alpha.id, Money(900, "GBP"))).value.status == "awarded"
    second = await vendors.award(package.id, beta.id, Money(800, "GBP"))
    # Awarding commits money. Re-awarding is a new package, not an edit.
    assert not second.ok and "already awarded" in second.error.message


async def test_an_award_in_the_wrong_currency_is_refused(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    vendors = harness.capability(VendorScopeToken)
    package = (await packages.create(code="P", name="n", budget=Money(1000, "GBP"))).value
    vendor = (await vendors.upsert_vendor(name="Alpha")).value
    result = await vendors.award(package.id, vendor.id, Money(900, "USD"))
    assert not result.ok and "USD" in result.error.message


async def test_field_status_is_append_only_and_the_latest_wins(harness):
    await harness.load(markup_plugin, procurement_plugin)
    field = harness.capability(FieldStatusToken)
    element = ElementRef("m1", "E1")

    await field.record(element=element, state="in-progress")
    await field.record(element=element, state="installed")
    # Both observations are kept -- a progress claim has to be re-checkable against what was seen.
    assert len(field.query(latest_only=False)) == 2
    assert field.current(element).state == "installed"


async def test_progress_without_elements_refuses_to_invent_a_percentage(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    package = (await packages.create(code="P", name="n")).value
    result = await harness.capability(InstallProgressToken).compute(package.id, "2026-06-01")
    assert not result.ok and "without a denominator" in result.error.message


async def _awarded_package(harness, elements: int = 10):
    packages = harness.capability(PackageToken)
    vendors = harness.capability(VendorScopeToken)
    package = (await packages.create(code="P", name="Super", budget=Money(1_000_000, "GBP"))).value
    vendor = (await vendors.upsert_vendor(name="Alpha")).value
    await vendors.award(package.id, vendor.id, Money(1_000_000, "GBP"))
    refs = tuple(ElementRef("m1", f"E{i}") for i in range(elements))
    await packages.update(package.id, {"elements": refs})
    return package, refs


async def test_earned_value_follows_elements_in_place(harness):
    await harness.load(markup_plugin, procurement_plugin)
    package, refs = await _awarded_package(harness)
    field = harness.capability(FieldStatusToken)
    for element in refs[:6]:
        await field.record(element=element, state="installed", package_id=package.id)

    record = (
        await harness.capability(InstallProgressToken).compute(package.id, "2026-06-01")
    ).value
    assert record.percent_complete == pytest.approx(0.6)
    assert record.earned_value == Money(600_000, "GBP")


async def test_a_failed_inspection_sends_work_back_and_earned_value_falls(harness):
    """Work that has to be redone has not been earned, however complete it looked."""
    await harness.load(markup_plugin, procurement_plugin)
    package, refs = await _awarded_package(harness)
    field = harness.capability(FieldStatusToken)
    progress = harness.capability(InstallProgressToken)
    for element in refs[:6]:
        await field.record(element=element, state="installed", package_id=package.id)
    assert (await progress.compute(package.id, "2026-06-01")).value.earned_value == Money(
        600_000, "GBP"
    )

    inspections = harness.capability(InspectionToken)
    inspection = (await inspections.create(name="Weld check", package_id=package.id)).value
    issue_ids = (
        await inspections.fail(inspection.id, [{"element": refs[0], "note": "porosity"}])
    ).value

    assert harness.capability(IssueToken).get(issue_ids[0]).title.endswith("porosity")
    assert field.current(refs[0]).state == "rework"
    after = (await progress.compute(package.id, "2026-06-02")).value
    assert after.percent_complete == pytest.approx(0.5)
    assert after.earned_value == Money(500_000, "GBP")


async def test_failing_an_inspection_with_no_findings_is_refused(harness):
    await harness.load(markup_plugin, procurement_plugin)
    inspections = harness.capability(InspectionToken)
    inspection = (await inspections.create(name="Check")).value
    result = await inspections.fail(inspection.id, [])
    assert not result.ok and "nothing actionable" in result.error.message


async def test_earned_value_needs_an_award_to_earn_against(harness):
    await harness.load(markup_plugin, procurement_plugin)
    packages = harness.capability(PackageToken)
    package = (await packages.create(code="P", name="n")).value
    await packages.update(package.id, {"elements": (ElementRef("m1", "E1"),)})
    result = await harness.capability(InstallProgressToken).earned_value(package.id, "2026-06-01")
    assert not result.ok and "no rate to earn against" in result.error.message
