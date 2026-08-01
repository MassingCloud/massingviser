from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...kernel import KernelError, PluginContext, Result, err, ok
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
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    MASSING_EVENTS,
    CreateMassingInput,
    MassPromotionHandlerToken,
    OptionSummary,
    PromotionTarget,
)
from .geometry import (
    apply_planar_rigid,
    as_planar_rigid,
    compute_mass_metrics,
    floor_area_ratio,
    polygon_area,
    resolve_story_heights,
    to_xy,
    validate_profile,
)

#: Default storey height when a caller supplies none. Ordinary commercial floor-to-floor.
DEFAULT_STORY_HEIGHT = 3.5

#: Fallback palette for option styling -- distinguishable, and readable against a light scene.
OPTION_PALETTE = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")


@dataclass(frozen=True)
class MassingStores:
    profiles: RecordStore[ProfileRecord]
    masses: RecordStore[MassingObjectRecord]
    stories: RecordStore[MassingStoryRecord]
    options: RecordStore[OptionSetRecord]
    levels: RecordStore[LevelRecord]
    grids: RecordStore[GridLineRecord]
    site: RecordStore[SiteBoundaryRecord]


@dataclass(frozen=True)
class MassingRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_massing_stores(context: PluginContext) -> MassingStores:
    return MassingStores(
        profiles=create_record_store(context.state, "profiles"),
        masses=create_record_store(context.state, "objects"),
        stories=create_record_store(context.state, "stories"),
        options=create_record_store(context.state, "options"),
        levels=create_record_store(context.state, "levels"),
        grids=create_record_store(context.state, "grids"),
        site=create_record_store(context.state, "site"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


# ---------------------------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------------------------


class ProfileServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def _validate(
        self,
        points: Sequence[Sequence[float]],
        holes: Sequence[Sequence[Sequence[float]]] = (),
    ) -> Result[None, KernelError]:
        issues = validate_profile(to_xy(points), [to_xy(hole) for hole in holes])
        if not issues:
            return ok(None)
        return err(
            KernelError(
                "COMMAND_FAILED",
                " ".join(issue.message for issue in issues),
                {"issues": [issue.code for issue in issues]},
            )
        )

    def validate(self, points: Sequence[Sequence[float]]) -> Result[None, KernelError]:
        return self._validate(points)

    async def create(
        self,
        points: Sequence[Sequence[float]],
        *,
        name: str | None = None,
        base_elevation: float = 0.0,
    ) -> Result[ProfileRecord, KernelError]:
        validated = self._validate(points)
        if not validated.ok:
            return err(validated.error)

        record = ProfileRecord(
            id=self._runtime.ids.next("profile"),
            points=tuple(_as_vec3(p) for p in points),
            closed=True,
            name=name,
            base_elevation=base_elevation,
        )
        self._stores.profiles.add(record)
        self._runtime.context.events.emit(MASSING_EVENTS.profile_created, {"record": record})
        return ok(record)

    async def update(
        self, profile_id: Id, points: Sequence[Sequence[float]]
    ) -> Result[ProfileRecord, KernelError]:
        existing = self._stores.profiles.get(profile_id)
        if existing is None:
            return err(_not_found("profile", profile_id))

        validated = self._validate(points, existing.holes)
        if not validated.ok:
            return err(validated.error)

        updated = self._stores.profiles.update(
            profile_id, {"points": tuple(_as_vec3(p) for p in points)}
        )
        if updated is None:
            return err(_not_found("profile", profile_id))
        # Several masses may share one profile; every one of their metrics is now stale.
        self._runtime.context.events.emit(MASSING_EVENTS.profile_updated, {"record": updated})
        return ok(updated)

    async def add_hole(
        self, profile_id: Id, points: Sequence[Sequence[float]]
    ) -> Result[ProfileRecord, KernelError]:
        existing = self._stores.profiles.get(profile_id)
        if existing is None:
            return err(_not_found("profile", profile_id))

        holes = (*existing.holes, tuple(_as_vec3(p) for p in points))
        validated = self._validate(existing.points, holes)
        if not validated.ok:
            return err(validated.error)

        updated = self._stores.profiles.update(profile_id, {"holes": holes})
        if updated is None:
            return err(_not_found("profile", profile_id))
        self._runtime.context.events.emit(MASSING_EVENTS.profile_updated, {"record": updated})
        return ok(updated)

    def get(self, profile_id: Id) -> ProfileRecord | None:
        return self._stores.profiles.get(profile_id)

    def list(self) -> tuple[ProfileRecord, ...]:
        return self._stores.profiles.all()


def _as_vec3(point: Sequence[float]) -> tuple[float, float, float]:
    x = float(point[0])
    y = float(point[1])
    z = float(point[2]) if len(point) > 2 else 0.0
    return (x, y, z)


# ---------------------------------------------------------------------------------------------
# Stories
# ---------------------------------------------------------------------------------------------


def reconcile_stories(
    runtime: MassingRuntime, stores: MassingStores, mass: MassingObjectRecord
) -> tuple[MassingStoryRecord, ...]:
    """Bring the story records in line with a mass's ``story_count`` and ``story_heights``.

    Story records exist only to carry the per-story extras the flat ``MassingObjectRecord`` cannot
    -- programme, GFA exclusion, a setback outline. Reconciling rather than rebuilding preserves
    those extras when the count changes, which is the difference between adding a floor and losing
    the annotations on every floor below it.
    """
    heights = resolve_story_heights(mass.story_count, mass.story_heights, DEFAULT_STORY_HEIGHT)
    existing = sorted(
        stores.stories.query(lambda story: story.massing_object_id == mass.id),
        key=lambda story: story.index,
    )

    profile = stores.profiles.get(mass.profile_id)
    elevation = profile.base_elevation if profile else 0.0

    nxt: list[MassingStoryRecord] = []
    for index, height in enumerate(heights):
        previous = existing[index] if index < len(existing) else None
        nxt.append(
            MassingStoryRecord(
                id=previous.id if previous else runtime.ids.next("story"),
                massing_object_id=mass.id,
                index=index,
                elevation=elevation,
                height=height,
                name=previous.name if previous else None,
                profile_id=previous.profile_id if previous else None,
                programme=previous.programme if previous else None,
                excluded_from_gfa=previous.excluded_from_gfa if previous else False,
            )
        )
        elevation += height

    stores.stories.remove_where(lambda story: story.massing_object_id == mass.id)
    stores.stories.add_many(nxt)
    return tuple(nxt)


class StoryServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def _require_mass(self, id: Id) -> Result[MassingObjectRecord, KernelError]:
        mass = self._stores.masses.get(id)
        return ok(mass) if mass else err(_not_found("massing object", id))

    def _apply_heights(
        self, mass: MassingObjectRecord, heights: Sequence[float]
    ) -> Result[MassingObjectRecord, KernelError]:
        updated = self._stores.masses.update(
            mass.id,
            {
                "story_count": len(heights),
                "story_heights": tuple(heights),
                "total_height": sum(heights),
            },
        )
        if updated is None:
            return err(_not_found("massing object", mass.id))
        reconcile_stories(self._runtime, self._stores, updated)
        self._runtime.context.events.emit(
            MASSING_EVENTS.stories_changed,
            {"massingObjectId": updated.id, "storyCount": updated.story_count},
        )
        return ok(updated)

    def stories(self, massing_object_id: Id) -> tuple[MassingStoryRecord, ...]:
        return tuple(
            sorted(
                self._stores.stories.query(
                    lambda story: story.massing_object_id == massing_object_id
                ),
                key=lambda story: story.index,
            )
        )

    async def set_story_count(
        self, massing_object_id: Id, count: int
    ) -> Result[MassingObjectRecord, KernelError]:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "Story count must be a non-negative integer.",
                    {"count": count},
                )
            )
        mass = self._require_mass(massing_object_id)
        if not mass.ok:
            return err(mass.error)
        return self._apply_heights(
            mass.value,
            resolve_story_heights(count, mass.value.story_heights, DEFAULT_STORY_HEIGHT),
        )

    async def set_story_height(
        self, massing_object_id: Id, story_index: int, height: float
    ) -> Result[MassingStoryRecord, KernelError]:
        if height <= 0:
            return err(
                KernelError("COMMAND_FAILED", "Story height must be positive.", {"height": height})
            )
        mass = self._require_mass(massing_object_id)
        if not mass.ok:
            return err(mass.error)
        if story_index < 0 or story_index >= mass.value.story_count:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"No story at index {story_index}.",
                    {"storyIndex": story_index},
                )
            )

        heights = resolve_story_heights(
            mass.value.story_count, mass.value.story_heights, DEFAULT_STORY_HEIGHT
        )
        heights[story_index] = height
        applied = self._apply_heights(mass.value, heights)
        if not applied.ok:
            return err(applied.error)

        story = self._stores.stories.find(
            lambda candidate: (
                candidate.massing_object_id == massing_object_id and candidate.index == story_index
            )
        )
        return (
            ok(story) if story else err(_not_found("story", f"{massing_object_id}#{story_index}"))
        )

    async def edit_stories(
        self,
        massing_object_id: Id,
        predicate: Callable[[MassingStoryRecord], bool],
        changes: Mapping[str, Any],
    ) -> Result[tuple[MassingStoryRecord, ...], KernelError]:
        mass = self._require_mass(massing_object_id)
        if not mass.ok:
            return err(mass.error)

        targets = self._stores.stories.query(
            lambda story: story.massing_object_id == massing_object_id and predicate(story)
        )
        if not targets:
            return ok(())

        edited: list[MassingStoryRecord] = []
        for story in targets:
            updated = self._stores.stories.update(story.id, dict(changes))
            if updated is not None:
                edited.append(updated)

        # A height change in a bulk edit shifts every storey above it, so elevations and the mass's
        # own totals have to be rebuilt rather than patched in place.
        if changes.get("height") is not None:
            heights = [story.height for story in self.stories(massing_object_id)]
            applied = self._apply_heights(mass.value, heights)
            if not applied.ok:
                return err(applied.error)
            return ok(self.stories(massing_object_id))

        self._runtime.context.events.emit(
            MASSING_EVENTS.stories_changed,
            {"massingObjectId": massing_object_id, "storyCount": mass.value.story_count},
        )
        return ok(tuple(edited))

    async def insert_story(
        self, massing_object_id: Id, at_index: int, height: float
    ) -> Result[MassingObjectRecord, KernelError]:
        mass = self._require_mass(massing_object_id)
        if not mass.ok:
            return err(mass.error)
        if at_index < 0 or at_index > mass.value.story_count:
            return err(
                KernelError(
                    "COMMAND_FAILED", f"Cannot insert at {at_index}.", {"atIndex": at_index}
                )
            )
        heights = resolve_story_heights(
            mass.value.story_count, mass.value.story_heights, DEFAULT_STORY_HEIGHT
        )
        heights.insert(at_index, height)
        return self._apply_heights(mass.value, heights)

    async def remove_story(
        self, massing_object_id: Id, at_index: int
    ) -> Result[MassingObjectRecord, KernelError]:
        mass = self._require_mass(massing_object_id)
        if not mass.ok:
            return err(mass.error)
        if at_index < 0 or at_index >= mass.value.story_count:
            return err(
                KernelError(
                    "COMMAND_FAILED", f"No story at index {at_index}.", {"atIndex": at_index}
                )
            )
        heights = resolve_story_heights(
            mass.value.story_count, mass.value.story_heights, DEFAULT_STORY_HEIGHT
        )
        heights.pop(at_index)
        return self._apply_heights(mass.value, heights)


# ---------------------------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------------------------

#: Fields the mass service refuses to accept through `update`.
#:
#: They are derived from the story records; accepting them here would let a caller desynchronise
#: `story_count` from the stories that actually exist.
_DERIVED_FIELDS = frozenset({"id", "story_count", "story_heights", "total_height"})


class MassingServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def create(self, input: CreateMassingInput) -> Result[MassingObjectRecord, KernelError]:
        if not self._stores.profiles.has(input.profile_id):
            return err(_not_found("profile", input.profile_id))
        if (
            not isinstance(input.story_count, int)
            or isinstance(input.story_count, bool)
            or input.story_count < 0
        ):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "Story count must be a non-negative integer.",
                    {"storyCount": input.story_count},
                )
            )

        seed = input.story_heights
        if seed is None and input.story_height is not None:
            seed = [input.story_height]
        heights = resolve_story_heights(
            input.story_count, seed, input.story_height or DEFAULT_STORY_HEIGHT
        )

        record = MassingObjectRecord(
            id=self._runtime.ids.next("mass"),
            name=input.name,
            profile_id=input.profile_id,
            story_count=len(heights),
            story_heights=tuple(heights),
            total_height=sum(heights),
            editable=True,
            color=input.color,
            opacity=input.opacity,
            option_set_id=input.option_set_id,
            family_template_id=input.family_template_id,
        )

        self._stores.masses.add(record)
        reconcile_stories(self._runtime, self._stores, record)

        if input.option_set_id:
            option = self._stores.options.get(input.option_set_id)
            if option is not None:
                self._stores.options.update(
                    option.id,
                    {"massing_object_ids": (*option.massing_object_ids, record.id)},
                )

        self._runtime.context.events.emit(MASSING_EVENTS.created, {"record": record})
        return ok(record)

    async def transform(
        self, id: Id, matrix: Sequence[float], *, rejoin: Id | None = None
    ) -> Result[MassingObjectRecord, KernelError]:
        """Move or rotate one mass, by rewriting the footprint it is extruded from.

        A mass has no transform of its own -- it *is* its profile, extruded. So moving one means
        moving its profile, and that is where the hazard lies: profiles are deliberately shared, so
        that editing a footprint updates every option built on it. Moving a mass must not move its
        siblings, so a shared profile is forked here and only this mass repointed at the copy.

        Anything that is not a rotation about z plus a translation is refused by name. See
        ``as_planar_rigid``: a tilt has no representation as a mass, and quietly dropping it would
        report success for an edit that moved the building somewhere else.
        """
        existing = self._stores.masses.get(id)
        if existing is None:
            return err(_not_found("massing object", id))
        if not existing.editable:
            return err(
                KernelError("COMMAND_FAILED", f'Massing object "{id}" is locked.', {"id": id})
            )

        # Undo of a transform that forked a shared profile. Applying the inverse matrix instead
        # would put the mass back in the right place on the *wrong* profile -- geometrically
        # identical, but no longer moving when its sibling's footprint is edited, which is the one
        # thing sharing a profile is for.
        if rejoin is not None and rejoin != existing.profile_id:
            if not self._stores.profiles.has(rejoin):
                return err(_not_found("profile", rejoin))
            abandoned = existing.profile_id
            rejoined = self._stores.masses.update(id, {"profile_id": rejoin})
            if rejoined is None:
                return err(_not_found("massing object", id))
            if not any(other.profile_id == abandoned for other in self._stores.masses.all()):
                self._stores.profiles.remove(abandoned)
            reconcile_stories(self._runtime, self._stores, rejoined)
            self._runtime.context.events.emit(MASSING_EVENTS.updated, {"record": rejoined})
            return ok(rejoined)

        rigid = as_planar_rigid(matrix)
        if rigid is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "A mass is a vertical extrusion, so it can be rotated about z and moved, and "
                    "nothing else. This matrix is not that -- it tilts, scales, shears or "
                    "projects.",
                    {"id": id, "matrix": tuple(float(v) for v in matrix)},
                )
            )
        cos, sin, dx, dy, dz = rigid

        profile = self._stores.profiles.get(existing.profile_id)
        if profile is None:
            return err(_not_found("profile", existing.profile_id))

        points = tuple(apply_planar_rigid(profile.points, cos, sin, dx, dy))
        holes = tuple(tuple(apply_planar_rigid(hole, cos, sin, dx, dy)) for hole in profile.holes)
        elevation = profile.base_elevation + dz

        shared = any(
            other.id != id and other.profile_id == profile.id for other in self._stores.masses.all()
        )
        if shared:
            forked = ProfileRecord(
                id=self._runtime.ids.next("profile"),
                points=points,
                closed=profile.closed,
                name=profile.name,
                holes=holes,
                base_elevation=elevation,
            )
            self._stores.profiles.add(forked)
            moved = self._stores.masses.update(id, {"profile_id": forked.id})
            self._runtime.context.events.emit(MASSING_EVENTS.profile_created, {"record": forked})
        else:
            self._stores.profiles.update(
                profile.id,
                {"points": points, "holes": holes, "base_elevation": elevation},
            )
            moved = self._stores.masses.get(id)
            updated_profile = self._stores.profiles.get(profile.id)
            if updated_profile is not None:
                self._runtime.context.events.emit(
                    MASSING_EVENTS.profile_updated, {"record": updated_profile}
                )

        if moved is None:
            return err(_not_found("massing object", id))
        # The stories sit on the profile's plane, so a vertical move has to reach them too.
        if dz:
            reconcile_stories(self._runtime, self._stores, moved)
        self._runtime.context.events.emit(MASSING_EVENTS.updated, {"record": moved})
        return ok(moved)

    async def update(
        self, id: Id, changes: Mapping[str, Any]
    ) -> Result[MassingObjectRecord, KernelError]:
        existing = self._stores.masses.get(id)
        if existing is None:
            return err(_not_found("massing object", id))
        if not existing.editable:
            return err(
                KernelError("COMMAND_FAILED", f'Massing object "{id}" is locked.', {"id": id})
            )
        safe = {key: value for key, value in changes.items() if key not in _DERIVED_FIELDS}
        updated = self._stores.masses.update(id, safe)
        if updated is None:
            return err(_not_found("massing object", id))
        self._runtime.context.events.emit(MASSING_EVENTS.updated, {"record": updated})
        return ok(updated)

    async def remove(self, id: Id) -> Result[None, KernelError]:
        if not self._stores.masses.has(id):
            return err(_not_found("massing object", id))
        self._stores.masses.remove(id)
        self._stores.stories.remove_where(lambda story: story.massing_object_id == id)
        for option in self._stores.options.query(lambda o: id in o.massing_object_ids):
            self._stores.options.update(
                option.id,
                {
                    "massing_object_ids": tuple(
                        mass_id for mass_id in option.massing_object_ids if mass_id != id
                    )
                },
            )
        self._runtime.context.events.emit(MASSING_EVENTS.removed, {"id": id})
        return ok(None)

    def restore(self, record: MassingObjectRecord) -> Result[MassingObjectRecord, KernelError]:
        """Reinstate a removed mass verbatim. Exists so ``remove`` has an exact inverse for undo."""
        if self._stores.masses.has(record.id):
            return ok(record)
        self._stores.masses.add(record)
        reconcile_stories(self._runtime, self._stores, record)
        self._runtime.context.events.emit(MASSING_EVENTS.created, {"record": record})
        return ok(record)

    def get(self, id: Id) -> MassingObjectRecord | None:
        return self._stores.masses.get(id)

    def list(self) -> tuple[MassingObjectRecord, ...]:
        return self._stores.masses.all()

    async def duplicate(
        self, id: Id, *, name: str | None = None, option_set_id: Id | None = None
    ) -> Result[MassingObjectRecord, KernelError]:
        source = self._stores.masses.get(id)
        if source is None:
            return err(_not_found("massing object", id))

        copy = replace(
            source,
            id=self._runtime.ids.next("mass"),
            name=name or f"{source.name} copy",
            story_heights=tuple(source.story_heights),
            option_set_id=option_set_id if option_set_id is not None else source.option_set_id,
        )
        self._stores.masses.add(copy)

        # Copy the per-story extras too -- duplicating a scheme and losing its programme allocation
        # would make option studies useless.
        source_stories = sorted(
            self._stores.stories.query(lambda story: story.massing_object_id == id),
            key=lambda story: story.index,
        )
        self._stores.stories.add_many(
            [
                replace(story, id=self._runtime.ids.next("story"), massing_object_id=copy.id)
                for story in source_stories
            ]
        )

        self._runtime.context.events.emit(MASSING_EVENTS.created, {"record": copy})
        return ok(copy)


# ---------------------------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------------------------


class AppearanceServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def set_color(self, massing_object_id: Id, color: str) -> Result[None, KernelError]:
        updated = self._stores.masses.update(massing_object_id, {"color": color})
        if updated is None:
            return err(_not_found("massing object", massing_object_id))
        self._runtime.context.events.emit(MASSING_EVENTS.updated, {"record": updated})
        return ok(None)

    async def set_opacity(self, massing_object_id: Id, opacity: float) -> Result[None, KernelError]:
        if opacity < 0 or opacity > 1:
            return err(
                KernelError(
                    "COMMAND_FAILED", "Opacity must be between 0 and 1.", {"opacity": opacity}
                )
            )
        updated = self._stores.masses.update(massing_object_id, {"opacity": opacity})
        if updated is None:
            return err(_not_found("massing object", massing_object_id))
        self._runtime.context.events.emit(MASSING_EVENTS.updated, {"record": updated})
        return ok(None)

    async def apply_option_styling(
        self, option_set_id: Id, palette: Sequence[str] | None = None
    ) -> Result[None, KernelError]:
        option = self._stores.options.get(option_set_id)
        if option is None:
            return err(_not_found("option set", option_set_id))

        colours = tuple(palette) if palette else OPTION_PALETTE
        for index, mass_id in enumerate(option.massing_object_ids):
            self._stores.masses.update(mass_id, {"color": colours[index % len(colours)]})
        return ok(None)


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


class MetricsServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def _outline_for(
        self, profile_id: Id | None
    ) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]] | None:
        if not profile_id:
            return None
        profile = self._stores.profiles.get(profile_id)
        if profile is None:
            return None
        return to_xy(profile.points), [to_xy(hole) for hole in profile.holes]

    def _site_area(self) -> float | None:
        boundary = self._stores.site.all()
        if not boundary:
            return None
        first = boundary[0]
        if first.area is not None:
            return first.area
        area = polygon_area(to_xy(first.points))
        return area if area > 0 else None

    def compute_sync(self, massing_object_id: Id) -> Result[MassingMetrics, KernelError]:
        mass = self._stores.masses.get(massing_object_id)
        if mass is None:
            return err(_not_found("massing object", massing_object_id))

        base = self._outline_for(mass.profile_id)
        if base is None:
            return err(_not_found("profile", mass.profile_id))
        outer, holes = base

        stories = sorted(
            self._stores.stories.query(lambda story: story.massing_object_id == massing_object_id),
            key=lambda story: story.index,
        )

        story_outlines: dict[int, Sequence[tuple[float, float]]] = {}
        excluded: list[int] = []
        for story in stories:
            if story.excluded_from_gfa:
                excluded.append(story.index)
            override = self._outline_for(story.profile_id)
            if override is not None:
                story_outlines[story.index] = override[0]

        heights = (
            [story.height for story in stories]
            if stories
            else resolve_story_heights(mass.story_count, mass.story_heights, DEFAULT_STORY_HEIGHT)
        )

        profile = self._stores.profiles.get(mass.profile_id)
        result = compute_mass_metrics(
            outer=outer,
            holes=holes,
            story_heights=heights,
            base_elevation=profile.base_elevation if profile else 0.0,
            excluded_stories=excluded,
            story_outlines=story_outlines,
        )

        far = floor_area_ratio(result.gross_floor_area, self._site_area())
        metrics = MassingMetrics(
            massing_object_id=massing_object_id,
            footprint_area=result.footprint_area,
            gross_floor_area=result.gross_floor_area,
            volume=result.volume,
            envelope_area=result.envelope_area,
            story_count=result.story_count,
            height=result.height,
            computed_at=self._runtime.clock.iso(),
            floor_area_ratio=far,
        )

        # Cached back onto the record so the common read path does not recompute geometry.
        self._stores.masses.update(
            massing_object_id,
            {
                "area": result.footprint_area,
                "gross_floor_area": result.gross_floor_area,
                "volume": result.volume,
            },
        )
        self._runtime.context.events.emit(MASSING_EVENTS.metrics_computed, {"metrics": metrics})
        return ok(metrics)

    async def compute(self, massing_object_id: Id) -> Result[MassingMetrics, KernelError]:
        return self.compute_sync(massing_object_id)

    async def compute_all(
        self, option_set_id: Id | None = None
    ) -> Result[tuple[MassingMetrics, ...], KernelError]:
        if option_set_id:
            option = self._stores.options.get(option_set_id)
            ids = option.massing_object_ids if option else ()
        else:
            ids = tuple(mass.id for mass in self._stores.masses.all())

        results = []
        for id in ids:
            result = self.compute_sync(id)
            # One unbuildable mass must not hide the metrics for every other option.
            if result.ok:
                results.append(result.value)
        return ok(tuple(results))

    async def summarise(self, option_set_id: Id) -> Result[OptionSummary, KernelError]:
        option = self._stores.options.get(option_set_id)
        if option is None:
            return err(_not_found("option set", option_set_id))

        gross_floor_area = 0.0
        volume = 0.0
        footprint_area = 0.0
        for mass_id in option.massing_object_ids:
            result = self.compute_sync(mass_id)
            if not result.ok:
                continue
            gross_floor_area += result.value.gross_floor_area
            volume += result.value.volume
            footprint_area += result.value.footprint_area

        boundaries = self._stores.site.all()
        boundary = boundaries[0] if boundaries else None
        far = floor_area_ratio(gross_floor_area, self._site_area())
        max_far = boundary.max_floor_area_ratio if boundary else None
        max_height = boundary.max_height if boundary else None

        tallest = max(
            (
                mass.total_height
                for mass in (self._stores.masses.get(id) for id in option.massing_object_ids)
                if mass is not None
            ),
            default=0.0,
        )
        if max_far is None and max_height is None:
            within_limits = None
        else:
            within_limits = (max_far is None or (far or 0.0) <= max_far) and (
                max_height is None or tallest <= max_height
            )

        return ok(
            OptionSummary(
                gross_floor_area=gross_floor_area,
                volume=volume,
                footprint_area=footprint_area,
                floor_area_ratio=far,
                within_limits=within_limits,
            )
        )


# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------


class OptionServiceImpl:
    __slots__ = ("_runtime", "_stores", "_metrics")

    def __init__(
        self, runtime: MassingRuntime, stores: MassingStores, metrics: MetricsServiceImpl
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._metrics = metrics

    async def create(
        self, name: str, massing_object_ids: Sequence[Id] = ()
    ) -> Result[OptionSetRecord, KernelError]:
        record = OptionSetRecord(
            id=self._runtime.ids.next("option"),
            name=name,
            massing_object_ids=tuple(massing_object_ids),
            created_at=self._runtime.clock.iso(),
        )
        self._stores.options.add(record)
        for mass_id in massing_object_ids:
            self._stores.masses.update(mass_id, {"option_set_id": record.id})
        return ok(record)

    async def set_active(self, option_set_id: Id) -> Result[None, KernelError]:
        if not self._stores.options.has(option_set_id):
            return err(_not_found("option set", option_set_id))
        for option in self._stores.options.all():
            self._stores.options.update(option.id, {"active": option.id == option_set_id})
        self._runtime.context.events.emit(
            MASSING_EVENTS.option_activated, {"optionSetId": option_set_id}
        )
        return ok(None)

    async def compare(
        self, option_set_ids: Sequence[Id]
    ) -> Result[tuple[MassingMetrics, ...], KernelError]:
        results: list[MassingMetrics] = []
        for option_set_id in option_set_ids:
            computed = await self._metrics.compute_all(option_set_id)
            if computed.ok:
                results.extend(computed.value)
        return ok(tuple(results))

    def list(self) -> tuple[OptionSetRecord, ...]:
        return self._stores.options.all()


# ---------------------------------------------------------------------------------------------
# Context (levels, grids, site)
# ---------------------------------------------------------------------------------------------


class ContextServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def levels(self) -> tuple[LevelRecord, ...]:
        return self._stores.levels.all()

    def grids(self) -> tuple[GridLineRecord, ...]:
        return self._stores.grids.all()

    def site_boundary(self) -> SiteBoundaryRecord | None:
        boundaries = self._stores.site.all()
        return boundaries[0] if boundaries else None

    async def set_site_boundary(
        self,
        points: Sequence[Sequence[float]],
        *,
        name: str | None = None,
        max_floor_area_ratio: float | None = None,
        max_height: float | None = None,
    ) -> Result[SiteBoundaryRecord, KernelError]:
        issues = validate_profile(to_xy(points))
        if issues:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    " ".join(issue.message for issue in issues),
                    {"issues": [issue.code for issue in issues]},
                )
            )
        record = SiteBoundaryRecord(
            id=self._runtime.ids.next("site"),
            points=tuple(_as_vec3(p) for p in points),
            name=name,
            area=polygon_area(to_xy(points)),
            max_floor_area_ratio=max_floor_area_ratio,
            max_height=max_height,
        )
        # One site at a time -- a project with two boundaries has no defined plot ratio.
        self._stores.site.clear()
        self._stores.site.add(record)
        self._runtime.context.events.emit(MASSING_EVENTS.site_changed, {"record": record})
        return ok(record)

    async def derive_levels(
        self, massing_object_id: Id
    ) -> Result[tuple[LevelRecord, ...], KernelError]:
        mass = self._stores.masses.get(massing_object_id)
        if mass is None:
            return err(_not_found("massing object", massing_object_id))

        stories = sorted(
            self._stores.stories.query(lambda story: story.massing_object_id == massing_object_id),
            key=lambda story: story.index,
        )
        if not stories:
            return ok(())

        levels = [
            LevelRecord(
                id=self._runtime.ids.next("level"),
                name=story.name or f"Level {story.index + 1}",
                elevation=story.elevation,
            )
            for story in stories
        ]
        # A roof level: downstream authoring needs something to host the roof to.
        top = stories[-1]
        levels.append(
            LevelRecord(
                id=self._runtime.ids.next("level"),
                name="Roof",
                elevation=top.elevation + top.height,
            )
        )

        self._stores.levels.add_many(levels)
        return ok(tuple(levels))


# ---------------------------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------------------------


class PromotionServiceImpl:
    """Promotion delegates rather than implements.

    Turning a mass into building systems, a family package, or a working model means writing into
    another capability family's domain. Doing that directly would couple massing to authoring and
    family-libraries and defeat the plugin architecture, so massing looks for a registered handler
    and reports honestly when none is installed.
    """

    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: MassingRuntime, stores: MassingStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def promote(
        self, massing_object_id: Id, target: PromotionTarget, **options: Any
    ) -> Result[Mapping[str, Any], KernelError]:
        mass = self._stores.masses.get(massing_object_id)
        if mass is None:
            return err(_not_found("massing object", massing_object_id))

        handlers = self._runtime.context.capabilities.get_all(MassPromotionHandlerToken)
        handler = next(
            (provider.value for provider in handlers if provider.value.target == target), None
        )
        if handler is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    f'No plugin handles promotion to "{target}". '
                    "Install the capability that owns that destination.",
                    {
                        "target": target,
                        "available": [provider.value.target for provider in handlers],
                    },
                )
            )

        promoted = await handler.promote(mass, options)
        if not promoted.ok:
            return err(promoted.error)

        target_id = promoted.value.get("targetId")
        self._runtime.context.events.emit(
            MASSING_EVENTS.promoted,
            {
                "massingObjectId": massing_object_id,
                "target": target,
                "targetId": target_id,
            },
        )
        return ok({"targetId": target_id, "target": target})
