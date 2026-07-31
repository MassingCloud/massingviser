"""Stable schema identifiers.

These strings are written into every persisted document and are therefore permanent: renaming one
orphans existing data. New capability families add entries; they do not repurpose them.

The identifiers keep the ``massingifc.`` prefix on purpose. They name a *wire format*, not a
package, and documents written by the TypeScript implementation must open here unchanged --
renaming them would fork the format for no gain.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final


class SCHEMA:
    project: Final = "massingifc.project"
    session: Final = "massingifc.session"
    model: Final = "massingifc.model"

    markup: Final = "massingifc.markup"
    issue: Final = "massingifc.issue"
    comment_thread: Final = "massingifc.comment-thread"
    review_snapshot: Final = "massingifc.review-snapshot"
    review_session: Final = "massingifc.review-session"
    viewpoint: Final = "massingifc.viewpoint"

    massing_object: Final = "massingifc.massing.object"
    massing_profile: Final = "massingifc.massing.profile"
    massing_story: Final = "massingifc.massing.story"
    massing_option_set: Final = "massingifc.massing.option-set"
    level: Final = "massingifc.level"
    grid: Final = "massingifc.grid"
    site_boundary: Final = "massingifc.site-boundary"

    family_repository: Final = "massingifc.family.repository"
    family_package: Final = "massingifc.family.package"
    family_instance: Final = "massingifc.family.instance"

    twin_object: Final = "massingifc.twin.object"
    twin_alignment: Final = "massingifc.twin.alignment"
    twin_observation: Final = "massingifc.twin.observation"
    twin_promotion: Final = "massingifc.twin.promotion"

    clash: Final = "massingifc.coordination.clash"
    clash_test: Final = "massingifc.coordination.clash-test"
    validation_rule: Final = "massingifc.coordination.validation-rule"
    validation_result: Final = "massingifc.coordination.validation-result"
    revision_diff: Final = "massingifc.coordination.revision-diff"
    responsibility: Final = "massingifc.coordination.responsibility"

    schedule_task: Final = "massingifc.planning.task"
    task_dependency: Final = "massingifc.planning.dependency"
    task_model_link: Final = "massingifc.planning.model-link"
    progress_comparison: Final = "massingifc.planning.progress"

    quantity: Final = "massingifc.cost.quantity"
    takeoff_rule: Final = "massingifc.cost.takeoff-rule"
    classification_system: Final = "massingifc.cost.classification-system"
    classification_mapping: Final = "massingifc.cost.classification-mapping"
    resource: Final = "massingifc.cost.resource"
    cost_assembly: Final = "massingifc.cost.assembly"
    boq: Final = "massingifc.cost.boq"
    boq_line: Final = "massingifc.cost.boq-line"
    estimate: Final = "massingifc.cost.estimate"
    cashflow: Final = "massingifc.cost.cashflow"
    change_impact: Final = "massingifc.cost.change-impact"

    procurement_package: Final = "massingifc.procurement.package"
    vendor: Final = "massingifc.procurement.vendor"
    vendor_scope: Final = "massingifc.procurement.vendor-scope"
    field_status: Final = "massingifc.field.status"
    inspection: Final = "massingifc.field.inspection"
    install_progress: Final = "massingifc.field.install-progress"


ALL_SCHEMAS: tuple[str, ...] = tuple(
    value
    for name, value in vars(SCHEMA).items()
    if not name.startswith("_") and isinstance(value, str)
)

#: Current version of every schema the platform ships.
#:
#: Everything starts at 1. The value of declaring them now -- rather than when the first migration
#: is written -- is that the forward-incompatibility guard works from the first release: a document
#: written by a future build is refused instead of being silently misread.
CURRENT_VERSION: Mapping[str, int] = MappingProxyType({schema: 1 for schema in ALL_SCHEMAS})
