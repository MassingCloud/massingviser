from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ...kernel import CommandDefinition, CommandInvocation, PluginContext, UIContribution
from ...schema import Id, MassingObjectRecord, MassingStoryRecord
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    MASSING_COMMANDS,
    MASSING_PERMISSIONS,
    AppearanceToken,
    ContextToken,
    CreateMassingInput,
    MassingToken,
    MetricsToken,
    OptionToken,
    ProfileToken,
    PromotionToken,
    StoryToken,
)
from .geometry import invert_planar_rigid
from .services import (
    DEFAULT_STORY_HEIGHT,
    AppearanceServiceImpl,
    ContextServiceImpl,
    MassingRuntime,
    MassingServiceImpl,
    MetricsServiceImpl,
    OptionServiceImpl,
    ProfileServiceImpl,
    PromotionServiceImpl,
    StoryServiceImpl,
    create_massing_stores,
)

PLUGIN_ID = "massingviser.massing"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    """Raise on failure so the command bus records it as a failed command.

    Handlers raise rather than return ``Err`` because the bus already funnels every raise through
    ``attempt_async`` into a ``Result``. Returning one here would nest ``Result`` inside ``Result``
    and every caller would have to unwrap twice.
    """
    if not result.ok:
        raise result.error
    return result.value


def create_massing_plugin(
    *,
    clock: Clock | None = None,
    ids: IdFactory | None = None,
    #: Auto-recompute metrics when geometry changes. Defaults to true.
    auto_compute_metrics: bool = True,
) -> Any:
    """The massing capability, packaged as a plugin.

    Every mutation is exposed as a command with an inverse, so massing participates in the kernel's
    undo history rather than maintaining its own. That is what makes "sketch, extrude, add two
    floors, change the colour, undo four times" behave the way a designer expects -- including
    across a sequence that touches profiles, stories and appearance.
    """
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = MassingRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_massing_stores(context)

        profiles = ProfileServiceImpl(runtime, stores)
        masses = MassingServiceImpl(runtime, stores)
        stories = StoryServiceImpl(runtime, stores)
        appearance = AppearanceServiceImpl(runtime, stores)
        metrics = MetricsServiceImpl(runtime, stores)
        option_sets = OptionServiceImpl(runtime, stores, metrics)
        project_context = ContextServiceImpl(runtime, stores)
        promotion = PromotionServiceImpl(runtime, stores)

        context.capabilities.provide(ProfileToken, profiles, version=PLUGIN_VERSION)
        context.capabilities.provide(MassingToken, masses, version=PLUGIN_VERSION)
        context.capabilities.provide(StoryToken, stories, version=PLUGIN_VERSION)
        context.capabilities.provide(AppearanceToken, appearance, version=PLUGIN_VERSION)
        context.capabilities.provide(MetricsToken, metrics, version=PLUGIN_VERSION)
        context.capabilities.provide(OptionToken, option_sets, version=PLUGIN_VERSION)
        context.capabilities.provide(ContextToken, project_context, version=PLUGIN_VERSION)
        context.capabilities.provide(PromotionToken, promotion, version=PLUGIN_VERSION)

        def refresh(massing_object_id: Id) -> None:
            """Metrics are cheap, and always-fresh beats sometimes-stale for a feedback number."""
            if auto_compute_metrics:
                metrics.compute_sync(massing_object_id)

        # Captured before removal so the inverse can put back the exact record, including the
        # per-story extras that a fresh create would not reproduce.
        restore_buffer: dict[
            Id, tuple[MassingObjectRecord | None, tuple[MassingStoryRecord, ...]]
        ] = {}

        # -----------------------------------------------------------------------------------
        # Profiles
        # -----------------------------------------------------------------------------------

        async def sketch_profile(params: Mapping[str, Any], _ctx: Any) -> Id:
            created = _unwrap(
                await profiles.create(
                    params["points"],
                    name=params.get("name"),
                    base_elevation=params.get("base_elevation", 0.0),
                )
            )
            return created.id

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.sketch_profile,
                title="Sketch profile",
                permission=MASSING_PERMISSIONS.edit,
                handler=sketch_profile,
            )
        )

        # -----------------------------------------------------------------------------------
        # Masses
        # -----------------------------------------------------------------------------------

        async def create_mass(params: Mapping[str, Any], _ctx: Any) -> MassingObjectRecord:
            created = _unwrap(await masses.create(CreateMassingInput(**dict(params))))
            refresh(created.id)
            return created

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.create_mass,
                title="Create mass",
                permission=MASSING_PERMISSIONS.edit,
                handler=create_mass,
                create_inverse=lambda _params, record: CommandInvocation(
                    MASSING_COMMANDS.remove_mass, {"id": record.id}
                ),
            )
        )

        async def remove_mass(params: Mapping[str, Any], _ctx: Any) -> MassingObjectRecord | None:
            mass_id = params["id"]
            snapshot = masses.get(mass_id)
            story_snapshot = stories.stories(mass_id)
            _unwrap(await masses.remove(mass_id))
            restore_buffer[mass_id] = (snapshot, story_snapshot)
            return snapshot

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.remove_mass,
                title="Delete mass",
                permission=MASSING_PERMISSIONS.edit,
                handler=remove_mass,
                create_inverse=lambda params, _result: CommandInvocation(
                    MASSING_COMMANDS.restore_mass, {"id": params["id"]}
                ),
            )
        )

        def restore_mass(params: Mapping[str, Any], _ctx: Any) -> None:
            mass_id = params["id"]
            buffered = restore_buffer.get(mass_id)
            if buffered is None or buffered[0] is None:
                return
            mass, story_snapshot = buffered
            masses.restore(mass)
            for story in story_snapshot:
                current = stores.stories.find(
                    lambda candidate, s=story: (
                        candidate.massing_object_id == mass_id and candidate.index == s.index
                    )
                )
                if current is not None:
                    stores.stories.replace(replace(story, id=current.id))
            refresh(mass_id)

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.restore_mass,
                title="Restore mass",
                permission=MASSING_PERMISSIONS.edit,
                handler=restore_mass,
                create_inverse=lambda params, _result: CommandInvocation(
                    MASSING_COMMANDS.remove_mass, {"id": params["id"]}
                ),
            )
        )

        async def duplicate_mass(params: Mapping[str, Any], _ctx: Any) -> MassingObjectRecord:
            copy = _unwrap(
                await masses.duplicate(
                    params["id"],
                    name=params.get("name"),
                    option_set_id=params.get("option_set_id"),
                )
            )
            refresh(copy.id)
            return copy

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.duplicate_mass,
                title="Duplicate mass",
                permission=MASSING_PERMISSIONS.edit,
                handler=duplicate_mass,
                create_inverse=lambda _params, record: CommandInvocation(
                    MASSING_COMMANDS.remove_mass, {"id": record.id}
                ),
            )
        )

        async def transform_mass(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            mass_id = params["id"]
            before = masses.get(mass_id)
            moved = _unwrap(
                await masses.transform(mass_id, params["matrix"], rejoin=params.get("rejoin"))
            )
            refresh(mass_id)
            return {
                "id": moved.id,
                "matrix": tuple(float(v) for v in params["matrix"]),
                # What the mass was extruded from before the move. If the transform forked a
                # shared profile, undo has to put it back on the original rather than leave it on
                # a private copy that happens to sit in the same place.
                "profile_id": before.profile_id if before else moved.profile_id,
            }

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.transform_mass,
                title="Move or rotate mass",
                permission=MASSING_PERMISSIONS.edit,
                handler=transform_mass,
                create_inverse=lambda _params, result: CommandInvocation(
                    MASSING_COMMANDS.transform_mass,
                    {
                        "id": result["id"],
                        "matrix": invert_planar_rigid(result["matrix"]),
                        "rejoin": result["profile_id"],
                    },
                ),
            )
        )

        # -----------------------------------------------------------------------------------
        # Stories
        # -----------------------------------------------------------------------------------

        async def set_story_count(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            mass_id = params["id"]
            previous = masses.get(mass_id)
            previous_heights = list(previous.story_heights) if previous else []
            _unwrap(await stories.set_story_count(mass_id, params["count"]))
            refresh(mass_id)
            return {"id": mass_id, "previous_heights": previous_heights}

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.set_story_count,
                title="Set story count",
                permission=MASSING_PERMISSIONS.edit,
                handler=set_story_count,
                # Restoring the heights list restores the count too, and does it without losing a
                # taller ground floor the way replaying "set count" with a uniform height would.
                create_inverse=lambda _params, result: CommandInvocation(
                    "massing.stories.set-heights",
                    {"id": result["id"], "heights": result["previous_heights"]},
                ),
            )
        )

        async def set_story_heights(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            mass_id = params["id"]
            heights = list(params["heights"])
            previous = masses.get(mass_id)
            if previous is None:
                raise KeyError(f'No massing object with id "{mass_id}".')
            previous_heights = list(previous.story_heights)

            _unwrap(await stories.set_story_count(mass_id, len(heights)))
            for index, height in enumerate(heights):
                _unwrap(await stories.set_story_height(mass_id, index, height))
            refresh(mass_id)
            return {"id": mass_id, "previous_heights": previous_heights}

        context.commands.register(
            CommandDefinition(
                id="massing.stories.set-heights",
                title="Set story heights",
                permission=MASSING_PERMISSIONS.edit,
                handler=set_story_heights,
                create_inverse=lambda _params, result: CommandInvocation(
                    "massing.stories.set-heights",
                    {"id": result["id"], "heights": result["previous_heights"]},
                ),
            )
        )

        async def edit_stories(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            mass_id = params["id"]
            previous = stories.stories(mass_id)
            from_index = params.get("from_index", 0)
            to_index = params.get("to_index")
            upper = float("inf") if to_index is None else to_index

            _unwrap(
                await stories.edit_stories(
                    mass_id,
                    lambda story: from_index <= story.index <= upper,
                    params.get("changes", {}),
                )
            )
            refresh(mass_id)
            return {"id": mass_id, "previous": previous}

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.edit_stories,
                title="Edit stories",
                permission=MASSING_PERMISSIONS.edit,
                handler=edit_stories,
                create_inverse=lambda _params, result: CommandInvocation(
                    "massing.stories.restore",
                    {"id": result["id"], "stories": result["previous"]},
                ),
            )
        )

        async def restore_stories(params: Mapping[str, Any], _ctx: Any) -> None:
            mass_id = params["id"]
            snapshot: Sequence[MassingStoryRecord] = params["stories"]
            heights = [story.height for story in snapshot]
            if heights:
                _unwrap(await stories.set_story_count(mass_id, len(heights)))
            for story in snapshot:
                current = stores.stories.find(
                    lambda candidate, s=story: (
                        candidate.massing_object_id == mass_id and candidate.index == s.index
                    )
                )
                if current is not None:
                    stores.stories.replace(replace(story, id=current.id))
            refresh(mass_id)

        context.commands.register(
            CommandDefinition(
                id="massing.stories.restore",
                title="Restore stories",
                permission=MASSING_PERMISSIONS.edit,
                handler=restore_stories,
            )
        )

        # -----------------------------------------------------------------------------------
        # Appearance
        # -----------------------------------------------------------------------------------

        async def set_color(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            mass_id = params["id"]
            existing = masses.get(mass_id)
            previous = existing.color if existing else None
            _unwrap(await appearance.set_color(mass_id, params["color"]))
            return {"id": mass_id, "previous": previous}

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.set_color,
                title="Set colour",
                permission=MASSING_PERMISSIONS.edit,
                handler=set_color,
                # Nothing to restore to when there was no colour; leave the history clean rather
                # than inventing a value.
                create_inverse=lambda _params, result: (
                    None
                    if result["previous"] is None
                    else CommandInvocation(
                        MASSING_COMMANDS.set_color,
                        {"id": result["id"], "color": result["previous"]},
                    )
                ),
            )
        )

        async def set_opacity(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            mass_id = params["id"]
            existing = masses.get(mass_id)
            previous = existing.opacity if existing else None
            _unwrap(await appearance.set_opacity(mass_id, params["opacity"]))
            return {"id": mass_id, "previous": previous}

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.set_opacity,
                title="Set opacity",
                permission=MASSING_PERMISSIONS.edit,
                handler=set_opacity,
                create_inverse=lambda _params, result: (
                    None
                    if result["previous"] is None
                    else CommandInvocation(
                        MASSING_COMMANDS.set_opacity,
                        {"id": result["id"], "opacity": result["previous"]},
                    )
                ),
            )
        )

        # -----------------------------------------------------------------------------------
        # Metrics, options, context, promotion
        # -----------------------------------------------------------------------------------

        async def compute_metrics(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await metrics.compute(params["id"]))

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.compute_metrics,
                title="Compute metrics",
                handler=compute_metrics,
            )
        )

        async def create_option(params: Mapping[str, Any], _ctx: Any) -> Id:
            created = _unwrap(
                await option_sets.create(params["name"], params.get("massing_object_ids", ()))
            )
            return created.id

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.create_option,
                title="Create option",
                permission=MASSING_PERMISSIONS.edit,
                handler=create_option,
            )
        )

        async def activate_option(params: Mapping[str, Any], _ctx: Any) -> None:
            _unwrap(await option_sets.set_active(params["option_set_id"]))

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.activate_option,
                title="Activate option",
                permission=MASSING_PERMISSIONS.edit,
                handler=activate_option,
            )
        )

        async def compare_options(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await option_sets.compare(params["option_set_ids"]))

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.compare_options,
                title="Compare options",
                handler=compare_options,
            )
        )

        async def derive_levels(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await project_context.derive_levels(params["id"]))

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.derive_levels,
                title="Derive levels",
                permission=MASSING_PERMISSIONS.edit,
                handler=derive_levels,
            )
        )

        async def set_site(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await project_context.set_site_boundary(
                    params["points"],
                    name=params.get("name"),
                    max_floor_area_ratio=params.get("max_floor_area_ratio"),
                    max_height=params.get("max_height"),
                )
            )

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.set_site_boundary,
                title="Set site boundary",
                permission=MASSING_PERMISSIONS.edit,
                handler=set_site,
            )
        )

        async def promote(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await promotion.promote(params["id"], params["target"], **params.get("options", {}))
            )

        context.commands.register(
            CommandDefinition(
                id=MASSING_COMMANDS.promote,
                title="Promote mass",
                permission=MASSING_PERMISSIONS.promote,
                handler=promote,
            )
        )

        # -----------------------------------------------------------------------------------
        # UI contributions
        #
        # Descriptors only. The kernel never renders; the viser shell reads these and decides how
        # a "panel" or a "toolbar" looks in a browser.
        # -----------------------------------------------------------------------------------

        context.ui.register(
            UIContribution(
                id="massing.panel", point="panel", title="Massing", placement="left", order=20
            )
        )
        context.ui.register(
            UIContribution(
                id="massing.toolbar.sketch",
                point="toolbar",
                title="Sketch mass",
                group="authoring",
                order=10,
                command_id=MASSING_COMMANDS.sketch_profile,
            )
        )
        context.ui.register(
            UIContribution(
                id="massing.inspector.metrics",
                point="inspector",
                title="Massing metrics",
                order=10,
                when=lambda ctx: ctx.get("selectionKind") == "massing",
            )
        )

        context.logger.info(
            "Massing capability ready", {"defaultStoryHeight": DEFAULT_STORY_HEIGHT}
        )

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Massing",
        description="Sketch-based, story-aware conceptual massing.",
        permissions=[MASSING_PERMISSIONS.edit, MASSING_PERMISSIONS.promote],
        activate=activate,
    )


#: Ready-to-use instance for hosts that need no injection.
massing_plugin = create_massing_plugin()
