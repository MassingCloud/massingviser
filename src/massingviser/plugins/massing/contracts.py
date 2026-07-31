"""``massingviser.plugins.massing`` -- the first authoring vertical.

Massing is where conceptual design actually happens: a sketched footprint, a story count, and a
fast read on area and volume. The contract is built around that loop rather than around generic
solid modelling, because story-awareness is what separates a massing tool from an extruder -- a
mass whose stories are implicit cannot answer "what is the GFA if I add two floors?", which is the
question the tool exists to answer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    GridLineRecord,
    Id,
    LevelRecord,
    MassingMetrics,
    MassingObjectRecord,
    MassingStoryRecord,
    OptionSetRecord,
    ProfileRecord,
    SiteBoundaryRecord,
)


@dataclass(frozen=True)
class CreateMassingInput:
    name: str
    profile_id: Id
    story_count: int
    #: Uniform height applied to every story when ``story_heights`` is not supplied.
    story_height: float | None = None
    story_heights: Sequence[float] | None = None
    color: str | None = None
    opacity: float | None = None
    option_set_id: Id | None = None
    family_template_id: Id | None = None


@runtime_checkable
class ProfileService(Protocol):
    async def create(
        self,
        points: Sequence[Sequence[float]],
        *,
        name: str | None = None,
        base_elevation: float = 0.0,
    ) -> Result[ProfileRecord, KernelError]: ...
    async def update(
        self, profile_id: Id, points: Sequence[Sequence[float]]
    ) -> Result[ProfileRecord, KernelError]: ...
    async def add_hole(
        self, profile_id: Id, points: Sequence[Sequence[float]]
    ) -> Result[ProfileRecord, KernelError]: ...
    #: Rejects self-intersecting or degenerate outlines before they reach geometry generation.
    def validate(self, points: Sequence[Sequence[float]]) -> Result[None, KernelError]: ...
    def get(self, profile_id: Id) -> ProfileRecord | None: ...
    def list(self) -> tuple[ProfileRecord, ...]: ...


ProfileToken: CapabilityToken[ProfileService] = create_capability_token("massing.profiles")


@runtime_checkable
class MassingService(Protocol):
    async def create(
        self, input: CreateMassingInput
    ) -> Result[MassingObjectRecord, KernelError]: ...
    async def update(
        self, id: Id, changes: Mapping[str, Any]
    ) -> Result[MassingObjectRecord, KernelError]: ...
    async def remove(self, id: Id) -> Result[None, KernelError]: ...
    def get(self, id: Id) -> MassingObjectRecord | None: ...
    def list(self) -> tuple[MassingObjectRecord, ...]: ...
    async def duplicate(
        self, id: Id, *, name: str | None = None, option_set_id: Id | None = None
    ) -> Result[MassingObjectRecord, KernelError]: ...


MassingToken: CapabilityToken[MassingService] = create_capability_token("massing.service")


@runtime_checkable
class StoryService(Protocol):
    """Story-level editing.

    ``edit_stories`` takes a predicate because the real workflows are bulk ones -- "make every
    floor above 10 residential", "set levels 2-6 to 3.6 m". Exposing only per-story setters would
    push that loop into every caller and lose the single-undo-step property.
    """

    def stories(self, massing_object_id: Id) -> tuple[MassingStoryRecord, ...]: ...
    async def set_story_count(
        self, massing_object_id: Id, count: int
    ) -> Result[MassingObjectRecord, KernelError]: ...
    async def set_story_height(
        self, massing_object_id: Id, story_index: int, height: float
    ) -> Result[MassingStoryRecord, KernelError]: ...
    async def edit_stories(
        self,
        massing_object_id: Id,
        predicate: Callable[[MassingStoryRecord], bool],
        changes: Mapping[str, Any],
    ) -> Result[tuple[MassingStoryRecord, ...], KernelError]: ...
    async def insert_story(
        self, massing_object_id: Id, at_index: int, height: float
    ) -> Result[MassingObjectRecord, KernelError]: ...
    async def remove_story(
        self, massing_object_id: Id, at_index: int
    ) -> Result[MassingObjectRecord, KernelError]: ...


StoryToken: CapabilityToken[StoryService] = create_capability_token("massing.stories")


@runtime_checkable
class AppearanceService(Protocol):
    async def set_color(self, massing_object_id: Id, color: str) -> Result[None, KernelError]: ...
    async def set_opacity(
        self, massing_object_id: Id, opacity: float
    ) -> Result[None, KernelError]: ...
    #: Applies a consistent palette across an option set, for side-by-side review.
    async def apply_option_styling(
        self, option_set_id: Id, palette: Sequence[str] | None = None
    ) -> Result[None, KernelError]: ...


AppearanceToken: CapabilityToken[AppearanceService] = create_capability_token("massing.appearance")


@dataclass(frozen=True)
class OptionSummary:
    gross_floor_area: float
    volume: float
    footprint_area: float
    floor_area_ratio: float | None = None
    within_limits: bool | None = None


@runtime_checkable
class MetricsService(Protocol):
    async def compute(self, massing_object_id: Id) -> Result[MassingMetrics, KernelError]: ...
    async def compute_all(
        self, option_set_id: Id | None = None
    ) -> Result[tuple[MassingMetrics, ...], KernelError]: ...
    #: Totals for an option, plus planning compliance when a site boundary is set.
    async def summarise(self, option_set_id: Id) -> Result[OptionSummary, KernelError]: ...


MetricsToken: CapabilityToken[MetricsService] = create_capability_token("massing.metrics")


@runtime_checkable
class OptionService(Protocol):
    async def create(
        self, name: str, massing_object_ids: Sequence[Id] = ()
    ) -> Result[OptionSetRecord, KernelError]: ...
    async def set_active(self, option_set_id: Id) -> Result[None, KernelError]: ...
    async def compare(
        self, option_set_ids: Sequence[Id]
    ) -> Result[tuple[MassingMetrics, ...], KernelError]: ...
    def list(self) -> tuple[OptionSetRecord, ...]: ...


OptionToken: CapabilityToken[OptionService] = create_capability_token("massing.options")


@runtime_checkable
class ContextService(Protocol):
    def levels(self) -> tuple[LevelRecord, ...]: ...
    def grids(self) -> tuple[GridLineRecord, ...]: ...
    def site_boundary(self) -> SiteBoundaryRecord | None: ...
    #: Generates levels from a mass's stories so downstream authoring has real levels to host to.
    async def derive_levels(
        self, massing_object_id: Id
    ) -> Result[tuple[LevelRecord, ...], KernelError]: ...


ContextToken: CapabilityToken[ContextService] = create_capability_token("massing.context")


PromotionTarget = Literal["building-systems", "family", "working-model"]


@runtime_checkable
class PromotionService(Protocol):
    #: Turns a conceptual mass into something downstream: a core and facade, or reusable content.
    async def promote(
        self, massing_object_id: Id, target: PromotionTarget, **options: Any
    ) -> Result[Mapping[str, Any], KernelError]: ...


PromotionToken: CapabilityToken[PromotionService] = create_capability_token("massing.promotion")


@runtime_checkable
class MassPromotionHandler(Protocol):
    """Performs promotion for one target.

    Registered by whichever plugin owns the destination -- authoring for building systems,
    family-libraries for reusable content. Massing itself must not import those packages, so the
    work is delegated through this capability instead.
    """

    @property
    def target(self) -> PromotionTarget: ...
    async def promote(
        self, mass: MassingObjectRecord, options: Mapping[str, Any] | None = None
    ) -> Result[Mapping[str, Any], KernelError]: ...


MassPromotionHandlerToken: CapabilityToken[MassPromotionHandler] = create_capability_token(
    "massing.promotion-handler"
)


class MASSING_COMMANDS:
    sketch_profile = "massing.profile.sketch"
    create_mass = "massing.create"
    remove_mass = "massing.remove"
    restore_mass = "massing.restore"
    duplicate_mass = "massing.duplicate"
    set_story_count = "massing.stories.set-count"
    edit_stories = "massing.stories.edit"
    set_color = "massing.appearance.set-color"
    set_opacity = "massing.appearance.set-opacity"
    compute_metrics = "massing.metrics.compute"
    create_option = "massing.option.create"
    activate_option = "massing.option.activate"
    compare_options = "massing.option.compare"
    derive_levels = "massing.context.derive-levels"
    set_site_boundary = "massing.context.set-site"
    promote = "massing.promote"


class MASSING_PERMISSIONS:
    edit = "massing.edit"
    promote = "massing.promote"


class MASSING_EVENTS:
    created = "massing.created"
    updated = "massing.updated"
    removed = "massing.removed"
    profile_created = "massing.profile.created"
    profile_updated = "massing.profile.updated"
    stories_changed = "massing.stories.changed"
    metrics_computed = "massing.metrics.computed"
    option_activated = "massing.option.activated"
    promoted = "massing.promoted"
    site_changed = "massing.site.changed"
