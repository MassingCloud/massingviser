"""Composition root.

Assembles a kernel, registers the capability plugins, and adds the one piece of glue that lets them
compose: a bridge that publishes massing geometry as *takeoff elements* and as an *element
resolver*.

That bridge is worth reading closely, because it is the architecture's whole claim in twenty lines.
Estimating does not import massing. Markup does not import massing. Neither knows the other exists.
They declare capability tokens for what they need -- "something that can list elements", "something
that can tell me whether an element still exists" -- and this plugin satisfies both from massing
records. Swap the bridge for one backed by an IFC file and the cost and markup plugins do not
change by a line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from .geometry import (
    MESH_ENCODING,
    SceneIndex,
    SpatialIndexToken,
    build_geometry_payloads,
)
from .kernel import Kernel, create_kernel
from .plugins.analytics import MetricProviderToken, MetricValue, analytics_plugin
from .plugins.authoring import authoring_plugin
from .plugins.coordination import (
    ClashEngineToken,
    ModelSnapshotToken,
    RawClash,
    SnapshotElement,
    coordination_plugin,
)
from .plugins.engine import (
    GeometryPayloadSourceToken,
    GeometryRef,
    PayloadRef,
    SceneNode,
    SceneNodeSourceToken,
    engine_plugin,
    payload_path,
)
from .plugins.estimating import (
    BoqToken,
    ModelElementSourceToken,
    ScheduleBasisToken,
    SchedulePeriod,
    TakeoffElement,
    estimating_plugin,
)
from .plugins.families import families_plugin
from .plugins.federation import federation_plugin
from .plugins.icdd import icdd_plugin
from .plugins.interop import interop_plugin
from .plugins.markup import ElementResolverToken, markup_plugin
from .plugins.massing import (
    MassingToken,
    MetricsToken,
    ProfileToken,
    StoryToken,
    extrude_stories,
    massing_plugin,
    to_xy,
)
from .plugins.planning import (
    ElementFilterSourceToken,
    ScheduleImportToken,
    planning_plugin,
)
from .plugins.procurement import BoqLineSourceToken, PackageBoqLine, procurement_plugin
from .plugins.shell import shell_plugin
from .plugins.twin import twin_plugin
from .schema import create_default_migration_registry
from .sdk import define_plugin


def _adapters_plugin() -> Any:
    """Imported lazily so a deployment with no extras never touches the adapter modules."""
    from .adapters.plugin import create_adapters_plugin

    return create_adapters_plugin()


#: The model id massing geometry is published under.
#:
#: Massing is a *model* as far as the rest of the platform is concerned -- it has elements with
#: stable identities that quantities and markup can reference -- so it needs a model id like any
#: other. Naming it here rather than inventing one per consumer is what keeps a quantity and a pin
#: talking about the same thing.
MASSING_MODEL_ID = "massing"

#: Bumped whenever the geometry changes meaning, not whenever it changes value. Quantities record
#: it so a re-run is comparable; a version that moved on every edit would make every estimate
#: incomparable with every other.
MASSING_MODEL_VERSION = "concept-1"


class _MassingElementSource:
    """Publishes each storey of each mass as one takeoff element.

    Per storey rather than per mass, because that is the level a cost plan works at: a rate applies
    to a floor plate, exclusions apply to a floor, and an estimator asked "which elements is that
    quantity?" expects floors back, not a single 40-storey object.
    """

    __slots__ = ("_kernel",)

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel

    def _elements(self) -> list[TakeoffElement]:
        masses = self._kernel.capabilities.get(MassingToken)
        stories = self._kernel.capabilities.get(StoryToken)
        metrics = self._kernel.capabilities.get(MetricsToken)
        if masses is None or stories is None or metrics is None:
            return []

        elements: list[TakeoffElement] = []
        for mass in masses.list():
            computed = metrics.compute_sync(mass.id)
            # A mass whose profile has been deleted cannot be measured. It is skipped, and the
            # takeoff's `empty_rules` reporting is what makes that visible rather than silent.
            if not computed.ok:
                continue
            footprint = computed.value.footprint_area
            for story in stories.stories(mass.id):
                area = story.area if story.area is not None else footprint
                elements.append(
                    TakeoffElement(
                        # Stable across sessions and rebuilds: it is derived from the mass id and
                        # the storey index, never from a render handle.
                        global_id=f"{mass.id}:{story.index:03d}",
                        ifc_class="IfcBuildingStorey",
                        properties={
                            "Area": area,
                            "Height": story.height,
                            "Volume": area * story.height,
                            "Elevation": story.elevation,
                        },
                        level_global_id=f"{mass.id}:{story.index:03d}",
                        classification_code=None if story.excluded_from_gfa else "SUPERSTRUCTURE",
                    )
                )
        return elements

    # -- ModelElementSource ------------------------------------------------------------------

    def elements(self, model_id: str) -> Sequence[TakeoffElement]:
        return self._elements() if model_id == MASSING_MODEL_ID else []

    def model_ids(self) -> Sequence[str]:
        return (MASSING_MODEL_ID,)

    def model_version(self, model_id: str) -> str | None:
        return MASSING_MODEL_VERSION if model_id == MASSING_MODEL_ID else None


class _MassingElementResolver:
    """Answers markup's "does this element still exist?".

    A mass id counts as an element here, so a pin dropped on a mass survives every edit to that
    mass and orphans the moment the mass is deleted -- which is the behaviour the anchoring
    contract promises, expressed against the only model this build actually has.
    """

    __slots__ = ("_kernel",)

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel

    def global_ids(self, model_id: str) -> Sequence[str]:
        masses = self._kernel.capabilities.get(MassingToken)
        stories = self._kernel.capabilities.get(StoryToken)
        if masses is None or model_id != MASSING_MODEL_ID:
            return ()
        ids: list[str] = []
        for mass in masses.list():
            ids.append(mass.id)
            if stories is not None:
                ids.extend(f"{mass.id}:{story.index:03d}" for story in stories.stories(mass.id))
        return tuple(ids)

    def exists(self, model_id: str, global_id: str) -> bool:
        return global_id in set(self.global_ids(model_id))


class _NominalScheduleBasis:
    """A placeholder S-curve so cashflow works before 4D planning exists.

    Explicitly nominal. A real schedule basis comes from the planning plugin; this exists so the
    cashflow path is exercised end to end rather than being dead code waiting on a package that is
    not in this build. It is registered at a low priority so a real one wins the moment it appears.
    """

    __slots__ = ()

    #: Rough construction S-curve: slow start, heavy middle, tail-off.
    WEIGHTS = (0.05, 0.10, 0.18, 0.22, 0.20, 0.15, 0.10)

    def periods(self, unit: str = "month") -> Sequence[SchedulePeriod]:
        return tuple(
            SchedulePeriod(
                start=f"2026-{index + 1:02d}-01",
                end=f"2026-{index + 2:02d}-01",
                weight=weight,
            )
            for index, weight in enumerate(self.WEIGHTS)
        )


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _period_start(moment: datetime, unit: str) -> datetime:
    if unit == "week":
        return (moment - timedelta(days=moment.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if unit == "quarter":
        return datetime(moment.year, 3 * ((moment.month - 1) // 3) + 1, 1)
    return datetime(moment.year, moment.month, 1)


def _next_period(start: datetime, unit: str) -> datetime:
    if unit == "week":
        return start + timedelta(days=7)
    months = 3 if unit == "quarter" else 1
    month = start.month - 1 + months
    return datetime(start.year + month // 12, month % 12 + 1, 1)


class _PlannedScheduleBasis:
    """The cashflow curve, derived from the programme that was actually imported.

    Weighted by **duration overlap**: each task spreads its work evenly across the days it spans,
    and every period collects the share that lands inside it. That is the standard way to get an
    S-curve out of a programme without pretending to resource loading the schedule does not carry.
    A milestone contributes nothing, because a zero-day task consumes no work -- counting it would
    put a spike of cost on a date where nothing is built.

    Falls back to the nominal curve while there is no programme, rather than reporting no periods
    at all. An estimate with no cashflow is less useful than one with a clearly labelled placeholder
    -- and `basis` says which of the two you are looking at.
    """

    __slots__ = ("_kernel", "_nominal")

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel
        self._nominal = _NominalScheduleBasis()

    def _tasks(self) -> Sequence[Any]:
        schedule = self._kernel.capabilities.get(ScheduleImportToken)
        return schedule.tasks() if schedule is not None else ()

    def basis(self) -> str:
        """Which curve the last call would return. Surfaced so a report can say so."""
        return "programme" if self._spans() else "nominal"

    def _spans(self) -> list[tuple[datetime, datetime]]:
        tasks = self._tasks()
        # A summary task spans its children and represents none of its own work. Counting both puts
        # the roll-up's duration on top of the detail underneath it, which loads the front of the
        # curve with cost that nothing builds. Only leaves carry work.
        parents = {task.parent_id for task in tasks if task.parent_id}

        spans = []
        for task in tasks:
            if task.id in parents:
                continue
            start = _parse_date(task.planned_start)
            finish = _parse_date(task.planned_finish)
            if start is None or finish is None or finish <= start:
                continue
            spans.append((start, finish))
        return spans

    def periods(self, unit: str = "month") -> Sequence[SchedulePeriod]:
        spans = self._spans()
        if not spans:
            return self._nominal.periods(unit)

        horizon = max(finish for _, finish in spans)
        cursor = _period_start(min(start for start, _ in spans), unit)
        buckets: list[tuple[datetime, datetime, float]] = []
        while cursor < horizon:
            end = _next_period(cursor, unit)
            share = 0.0
            for start, finish in spans:
                overlap = (min(finish, end) - max(start, cursor)).total_seconds()
                if overlap > 0:
                    # Share of *this task's* duration, so a six-month task and a one-week task
                    # each contribute one unit of work in total rather than in proportion to how
                    # long they run. Length is when the work happens, not how much there is.
                    share += overlap / (finish - start).total_seconds()
            buckets.append((cursor, end, share))
            cursor = end

        total = sum(share for _, _, share in buckets)
        if total <= 0:  # every task was a milestone
            return self._nominal.periods(unit)
        return tuple(
            SchedulePeriod(
                start=start.date().isoformat(),
                end=end.date().isoformat(),
                weight=share / total,
            )
            for start, end, share in buckets
        )


class _MassingElementFilter:
    """Resolves a 4D selection rule against massing storeys.

    The filter grammar is deliberately the same shape estimating's takeoff uses -- ``ifc_class``
    and a level -- so a rule written for a cost plan reads the same way in a programme.
    """

    __slots__ = ("_source",)

    def __init__(self, source: _MassingElementSource) -> None:
        self._source = source

    def match(self, model_id: str, filter: Any) -> Sequence[Any]:
        from .schema import ElementRef

        wanted_class = (filter or {}).get("ifc_class")
        level = (filter or {}).get("level_global_id")
        return tuple(
            ElementRef(model_id=model_id, global_id=element.global_id)
            for element in self._source.elements(model_id)
            if (wanted_class is None or element.ifc_class == wanted_class)
            and (level is None or element.level_global_id == level)
        )

    def model_ids(self) -> Sequence[str]:
        return self._source.model_ids()


class _MassingSnapshots:
    """Massing as a single-revision model.

    Honest about its limit: a concept model has one revision, so ``compare_to_previous`` refuses
    rather than inventing a second. A real deployment stores snapshots per issued revision, and the
    diff service does not change when it does.
    """

    __slots__ = ("_source",)

    def __init__(self, source: _MassingElementSource) -> None:
        self._source = source

    def snapshot(self, model_id: str, version: str) -> Sequence[SnapshotElement] | None:
        if model_id != MASSING_MODEL_ID or version != MASSING_MODEL_VERSION:
            return None
        return tuple(
            SnapshotElement(
                global_id=element.global_id,
                ifc_class=element.ifc_class,
                properties={"Elevation": element.properties.get("Elevation", 0.0)},
                quantities={
                    "Area": element.properties.get("Area", 0.0),
                    "Volume": element.properties.get("Volume", 0.0),
                },
            )
            for element in self._source.elements(model_id)
        )

    def model_ids(self) -> Sequence[str]:
        return (MASSING_MODEL_ID,)

    def versions(self, model_id: str) -> Sequence[str]:
        return (MASSING_MODEL_VERSION,) if model_id == MASSING_MODEL_ID else ()


class _MassingSpatialIndex:
    """Builds a real spatial index from massing geometry, and keeps it fresh.

    Rebuilt on demand rather than cached against an event, because a stale index answers "what did
    I click" with the wrong element -- and that is worse than rebuilding a few thousand boxes.
    """

    __slots__ = ("_kernel",)

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel

    def build(self) -> SceneIndex:
        masses = self._kernel.capabilities.get(MassingToken)
        stories = self._kernel.capabilities.get(StoryToken)
        profiles = self._kernel.capabilities.get(ProfileToken)
        if masses is None or stories is None or profiles is None:
            return SceneIndex({})

        elements = []
        for mass in masses.list():
            profile = profiles.get(mass.profile_id)
            if profile is None:
                continue
            footprint = [(point[0], point[1]) for point in profile.points]
            for story in stories.stories(mass.id):
                elements.append(
                    (
                        f"{mass.id}:{story.index:03d}",
                        # Each mass is its own group, so a clash test between two masses is the
                        # federated case in miniature.
                        mass.id,
                        footprint,
                        story.elevation,
                        story.height,
                    )
                )
        return SceneIndex.from_extrusions(elements)

    # -- SpatialIndex ------------------------------------------------------------------------

    def pick(self, origin, direction, **options):
        return self.build().pick(origin, direction, **options)

    def cull(self, view_projection):
        return self.build().cull(view_projection)

    def groups(self):
        masses = self._kernel.capabilities.get(MassingToken)
        return tuple(mass.id for mass in masses.list()) if masses else ()


class _BvhClashEngine:
    """Coordination's geometry port, backed by the real spatial index.

    Broad-phase, and honest about it: overlapping bounds are a *candidate*. Narrow-phase needs
    solids, which is a geometry-kernel job behind its own capability. What this replaces -- an
    elevation-band comparison -- was not even broad-phase; it was a placeholder.
    """

    __slots__ = ("_index",)

    def __init__(self, index: _MassingSpatialIndex) -> None:
        self._index = index

    def intersect(
        self, a: Sequence[Any], b: Sequence[Any], kind: str, tolerance: float
    ) -> Sequence[RawClash]:
        from .schema import ElementRef

        scene = self._index.build()
        groups = self._index.groups()
        if len(groups) < 2:
            return ()

        # A test with no explicit selections compares every pair of masses, which is what a
        # federated clash run means when nobody has narrowed it.
        wanted_left = {ref.global_id for ref in a} or set(groups)
        wanted_right = {ref.global_id for ref in b} or set(groups)

        found: list[RawClash] = []
        seen: set[tuple[str, str]] = set()
        for left in sorted(wanted_left & set(groups)):
            for right in sorted(wanted_right & set(groups)):
                if left >= right:
                    continue
                for candidate in scene.clash(left, right, tolerance=tolerance):
                    key = (candidate.a, candidate.b)
                    if key in seen:
                        continue
                    seen.add(key)
                    box = scene.box_of(candidate.a)
                    found.append(
                        RawClash(
                            a=ElementRef(MASSING_MODEL_ID, candidate.a),
                            b=ElementRef(MASSING_MODEL_ID, candidate.b),
                            point=box.centre if box else None,
                            distance=round(candidate.penetration, 6),
                        )
                    )
        return tuple(found)


class _BoqLineBridge:
    """Publishes the estimating plugin's bill lines in the shape procurement asks for."""

    __slots__ = ("_kernel",)

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel

    def lines(self, line_ids: Sequence[str] | None = None) -> Sequence[PackageBoqLine]:
        boqs = self._kernel.capabilities.get(BoqToken)
        if boqs is None:
            return ()
        wanted = set(line_ids) if line_ids is not None else None
        out: list[PackageBoqLine] = []
        for boq in (
            self._kernel.state.get_slice("massingviser.estimating-5d/boqs").get()
            if self._kernel.state.has_slice("massingviser.estimating-5d/boqs")
            else ()
        ):
            for line in boqs.lines(boq.id):
                if wanted is not None and line.id not in wanted:
                    continue
                out.append(
                    PackageBoqLine(
                        id=line.id,
                        description=line.description,
                        total=line.total,
                        classification_code=line.classification_code,
                        quantity=line.quantity.value,
                        unit=line.quantity.unit,
                    )
                )
        return tuple(out)


class _MassingMetrics:
    """Publishes the scheme's headline numbers to analytics.

    Analytics owns no numbers of its own; it aggregates. Computing GFA here rather than reading it
    from massing would be a second copy of the calculation, free to diverge.
    """

    namespace = "massing"
    __slots__ = ("_kernel",)

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel

    def collect(self) -> Sequence[MetricValue]:
        masses = self._kernel.capabilities.get(MassingToken)
        metrics = self._kernel.capabilities.get(MetricsToken)
        if masses is None or metrics is None:
            return ()

        gfa = volume = footprint = 0.0
        count = 0
        for mass in masses.list():
            computed = metrics.compute_sync(mass.id)
            if not computed.ok:
                continue
            count += 1
            gfa += computed.value.gross_floor_area
            volume += computed.value.volume
            footprint += computed.value.footprint_area
        return (
            MetricValue("masses", float(count), label="Masses"),
            MetricValue("gross_floor_area", gfa, unit="m2", label="Gross floor area"),
            MetricValue("volume", volume, unit="m3", label="Volume"),
            MetricValue("footprint_area", footprint, unit="m2", label="Footprint"),
        )


class _MassingGeometrySource:
    """Tessellates massing storeys and publishes them as content-addressed geometry payloads.

    The same ``extrude_stories`` call the viewer draws with, so what an engine receives and what
    the browser shows cannot drift into two different buildings.

    Rebuilt when the geometry *signature* changes -- mass ids, profile outlines and storey heights.
    Keying on the inputs rather than on an event means an edit that does not move a vertex does not
    invalidate a single payload, which is the behaviour that makes caching by content id worth
    anything. Encoding a tower is milliseconds; doing it on every camera move would not be.
    """

    __slots__ = ("_kernel", "_signature", "_set", "_refs")

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel
        self._signature: Any = None
        self._set: Any = None
        self._refs: tuple[Any, ...] = ()

    def _inputs(self) -> tuple[Any, dict[str, tuple[Any, Any]]]:
        masses = self._kernel.capabilities.get(MassingToken)
        stories = self._kernel.capabilities.get(StoryToken)
        profiles = self._kernel.capabilities.get(ProfileToken)
        if masses is None or stories is None or profiles is None:
            return (), {}

        signature: list[Any] = []
        meshes: dict[str, tuple[Any, Any]] = {}
        for mass in masses.list():
            profile = profiles.get(mass.profile_id)
            if profile is None:
                continue
            outer = to_xy(profile.points)
            holes = [to_xy(hole) for hole in profile.holes]
            story_records = stories.stories(mass.id)
            heights = [story.height for story in story_records] or list(mass.story_heights)
            if not heights:
                continue

            overrides: dict[int, Sequence[tuple[float, float]]] = {}
            for story in story_records:
                if story.profile_id:
                    override = profiles.get(story.profile_id)
                    if override is not None:
                        overrides[story.index] = to_xy(override.points)

            signature.append(
                (
                    mass.id,
                    tuple(outer),
                    tuple(tuple(hole) for hole in holes),
                    tuple(heights),
                    profile.base_elevation,
                    tuple(sorted((k, tuple(v)) for k, v in overrides.items())),
                )
            )
            for story_mesh in extrude_stories(
                outer,
                holes,
                heights,
                base_elevation=profile.base_elevation,
                story_outlines=overrides or None,
            ):
                if story_mesh.mesh.is_empty:
                    continue
                # The same id `_MassingElementSource` costs and `_MassingElementResolver` anchors.
                # Geometry that used its own numbering would be geometry nothing else could name.
                meshes[f"{mass.id}:{story_mesh.index:03d}"] = (
                    story_mesh.mesh.vertices,
                    story_mesh.mesh.faces,
                )
        return tuple(signature), meshes

    def _payloads(self) -> Any:
        signature, meshes = self._inputs()
        if signature != self._signature or self._set is None:
            self._set = build_geometry_payloads(meshes)
            self._signature = signature
            self._refs = tuple(
                PayloadRef(
                    id=payload.id,
                    role="geometry",
                    path=payload_path(payload.id, "bin"),
                    encoding=MESH_ENCODING,
                    byte_length=payload.byte_length,
                    lod=payload.lod,
                    mesh_count=payload.mesh_count,
                )
                for payload in self._set.payloads
            )
        return self._set

    # -- GeometryPayloadSource ---------------------------------------------------------------

    def payload_refs(self) -> Sequence[Any]:
        self._payloads()
        return self._refs

    def geometry(self) -> Mapping[str, Sequence[GeometryRef]]:
        built = self._payloads()
        return {
            global_id: tuple(
                GeometryRef(
                    payload_id=placement.payload_id,
                    geometry_index=placement.geometry_index,
                    lod=placement.lod,
                    face_count=placement.face_count,
                )
                for placement in ladder
            )
            for global_id, ladder in built.placements.items()
        }

    def read(self, payload_id: str) -> bytes | None:
        found = self._payloads().by_id(payload_id)
        return found.data if found is not None else None


class _MassingSceneSource:
    """Publishes massing storeys as engine scene nodes.

    The semantic half. Geometry arrives separately, through ``_MassingGeometrySource``, and the
    engine service joins the two on GlobalId -- so a deployment that installs only this one still
    exports a package that selects, filters and inspects.
    """

    __slots__ = ("_source",)

    def __init__(self, source: _MassingElementSource) -> None:
        self._source = source

    def nodes(self) -> Sequence[SceneNode]:
        return tuple(
            SceneNode(
                global_id=element.global_id,
                ifc_class=element.ifc_class,
                parent_global_id=element.global_id.split(":")[0],
                level_global_id=element.level_global_id,
                property_sets={
                    "Pset_QuantityTakeOff": {
                        "Area": element.properties.get("Area"),
                        "Height": element.properties.get("Height"),
                        "Volume": element.properties.get("Volume"),
                    }
                },
            )
            for element in self._source.elements(MASSING_MODEL_ID)
        )

    def payloads(self) -> Sequence[Any]:
        return ()

    def reality_layers(self) -> Sequence[Any]:
        return ()

    def source_units(self) -> str:
        return "m"

    def crs(self) -> str | None:
        return None


def create_bridge_plugin(kernel: Kernel[Any]) -> Any:
    """Wire the capability families together through tokens alone.

    Every provider here is satisfied from another family's *capability*, never from an import
    between families. Replace this one plugin with an IFC-backed equivalent and coordination,
    planning, procurement, cost and markup all keep working unchanged.
    """

    def activate(context: Any) -> None:
        source = _MassingElementSource(kernel)

        context.capabilities.provide(ModelElementSourceToken, source, version="0.1.0")
        context.capabilities.provide(
            ElementResolverToken, _MassingElementResolver(kernel), version="0.1.0"
        )
        context.capabilities.provide(
            ScheduleBasisToken, _PlannedScheduleBasis(kernel), version="0.1.0"
        )
        context.capabilities.provide(
            ElementFilterSourceToken, _MassingElementFilter(source), version="0.1.0"
        )
        context.capabilities.provide(ModelSnapshotToken, _MassingSnapshots(source), version="0.1.0")
        spatial = _MassingSpatialIndex(kernel)
        context.capabilities.provide(SpatialIndexToken, spatial, version="0.1.0")
        context.capabilities.provide(ClashEngineToken, _BvhClashEngine(spatial), version="0.1.0")
        context.capabilities.provide(BoqLineSourceToken, _BoqLineBridge(kernel), version="0.1.0")
        context.capabilities.provide(MetricProviderToken, _MassingMetrics(kernel), version="0.1.0")
        context.capabilities.provide(
            SceneNodeSourceToken, _MassingSceneSource(source), version="0.1.0"
        )
        context.capabilities.provide(
            GeometryPayloadSourceToken, _MassingGeometrySource(kernel), version="0.1.0"
        )
        context.logger.info("Massing published as a measurable, anchorable model")

    return define_plugin(
        id="massingviser.bridge",
        version="0.1.0",
        name="Massing bridge",
        description="Publishes massing geometry as takeoff elements and markup anchors.",
        activate=activate,
    )


def filesystem_storage(path: Any) -> Any:
    """A durable adapter that knows how to write this platform's records.

    The wiring lives here, not in ``massingviser.storage``. The adapters are deliberately generic --
    keys, values, bytes -- and the schema owns how a record becomes JSON. The composition root is
    the only layer entitled to know about both, so it is the only place the two are joined.
    """
    from .schema.codec import record_default, record_object_hook
    from .storage import FileSystemStorageAdapter

    return FileSystemStorageAdapter(path, default=record_default, object_hook=record_object_hook)


def sqlite_storage(path: Any = ":memory:") -> Any:
    """As :func:`filesystem_storage`, backed by a single SQLite file.

    Prefer this when a project holds many small records: one file per key costs an inode and a
    syscall each, and listing becomes a directory scan.
    """
    from .schema.codec import record_default, record_object_hook
    from .storage import SqliteStorageAdapter

    return SqliteStorageAdapter(path, default=record_default, object_hook=record_object_hook)


#: Registration order does not matter -- the plugin host sorts by declared dependency.
DEFAULT_PLUGINS = (
    "massing",
    "markup",
    "estimating",
    "coordination",
    "planning",
    "procurement",
    "families",
    "authoring",
    "twin",
    "federation",
    "interop",
    "analytics",
    "shell",
    "engine",
    "icdd",
    "bridge",
    # Optional. Activates on any machine; registers only the extras that are installed, and logs
    # which are not. See `massingviser.adapters`.
    "adapters",
)


def build_kernel(
    *,
    plugins: Sequence[str] = DEFAULT_PLUGINS,
    telemetry: Any = None,
    **kernel_options: Any,
) -> Kernel[Any]:
    """A kernel with the shipped capability plugins registered but not yet started.

    The migration registry is installed up front with every shipped schema declared at v1, so the
    forward-incompatibility guard works from the first save rather than from the first migration.
    """
    kernel_options.setdefault("migrator", create_default_migration_registry())
    if telemetry is not None:
        kernel_options["telemetry"] = telemetry

    kernel = create_kernel(**kernel_options)

    available = {
        "massing": lambda: massing_plugin,
        "markup": lambda: markup_plugin,
        "estimating": lambda: estimating_plugin,
        "coordination": lambda: coordination_plugin,
        "planning": lambda: planning_plugin,
        "procurement": lambda: procurement_plugin,
        "families": lambda: families_plugin,
        "authoring": lambda: authoring_plugin,
        "twin": lambda: twin_plugin,
        "federation": lambda: federation_plugin,
        "interop": lambda: interop_plugin,
        "analytics": lambda: analytics_plugin,
        "shell": lambda: shell_plugin,
        "engine": lambda: engine_plugin,
        "icdd": lambda: icdd_plugin,
        "adapters": _adapters_plugin,
        "bridge": lambda: create_bridge_plugin(kernel),
    }
    for name in plugins:
        factory = available.get(name)
        if factory is None:
            raise KeyError(f"Unknown plugin {name!r}. Available: {', '.join(sorted(available))}.")
        registered = kernel.use(factory())
        if not registered.ok:
            raise registered.error
    return kernel
