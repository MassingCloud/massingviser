from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import (
    ElementRef,
    Id,
    IsoTimestamp,
    TwinAlignmentRecord,
    TwinObjectRecord,
    TwinObservationRecord,
    TwinPromotionRecord,
    TwinTimelineRecord,
    measurability_reason,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .alignment import PlanarFit, fit_planar
from .contracts import TWIN_EVENTS, PointPair, TwinObjectFactoryToken

#: Targets that turn evidence into something the platform treats as authored geometry.
#:
#: ``asset`` is deliberately absent: cataloguing a splat is fine, measuring against one is not.
MEASURABLE_TARGETS = ("authoring", "family")


@dataclass(frozen=True)
class TwinStores:
    objects: RecordStore[TwinObjectRecord]
    alignments: RecordStore[TwinAlignmentRecord]
    observations: RecordStore[TwinObservationRecord]
    timelines: RecordStore[TwinTimelineRecord]
    promotions: RecordStore[TwinPromotionRecord]


@dataclass(frozen=True)
class TwinRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_twin_stores(context: PluginContext) -> TwinStores:
    return TwinStores(
        objects=create_record_store(context.state, "objects"),
        alignments=create_record_store(context.state, "alignments"),
        observations=create_record_store(context.state, "observations"),
        timelines=create_record_store(context.state, "timelines"),
        promotions=create_record_store(context.state, "promotions"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


class TwinRegistryServiceImpl:
    __slots__ = ("_runtime", "_stores", "_materialised")

    def __init__(self, runtime: TwinRuntime, stores: TwinStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._materialised: dict[Id, Any] = {}

    async def register(self, record: TwinObjectRecord) -> Result[TwinObjectRecord, KernelError]:
        if self._stores.objects.has(record.id):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Twin object "{record.id}" is already registered.',
                    {"twinObjectId": record.id},
                )
            )
        # Captured reality with no georeference cannot be checked against a survey or combined
        # with anything else, so it is refused rather than accepted and quietly mistrusted later.
        if record.kind in ("point-cloud", "gaussian-splat", "mesh-scan") and record.geo_reference is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'"{record.name}" is captured reality with no georeference. A transform alone '
                    "places it relative to a project origin nobody outside the project knows.",
                    {"twinObjectId": record.id, "kind": record.kind},
                )
            )
        self._stores.objects.add(record)
        self._runtime.context.events.emit(TWIN_EVENTS.registered, {"record": record})
        return ok(record)

    async def unregister(self, twin_object_id: Id) -> Result[None, KernelError]:
        if not self._stores.objects.remove(twin_object_id):
            return err(_not_found("twin object", twin_object_id))
        runtime_object = self._materialised.pop(twin_object_id, None)
        if runtime_object is not None:
            for provider in self._runtime.context.capabilities.get_all(TwinObjectFactoryToken):
                try:
                    provider.value.dispose(runtime_object)
                except Exception:  # noqa: BLE001
                    pass  # a factory that fails on teardown must not block deregistration
        self._stores.observations.remove_where(lambda o: o.twin_object_id == twin_object_id)
        return ok(None)

    async def materialise(self, twin_object_id: Id) -> Result[Any, KernelError]:
        record = self._stores.objects.get(twin_object_id)
        if record is None:
            return err(_not_found("twin object", twin_object_id))
        if twin_object_id in self._materialised:
            return ok(self._materialised[twin_object_id])

        factory = next(
            (
                provider.value
                for provider in self._runtime.context.capabilities.get_all(TwinObjectFactoryToken)
                if provider.value.kind == record.kind
            ),
            None,
        )
        if factory is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    f'No factory can materialise a "{record.kind}".',
                    {"kind": record.kind},
                )
            )
        created = await factory.create(record)
        if not created.ok:
            return err(created.error)
        self._materialised[twin_object_id] = created.value
        return ok(created.value)

    def set_visible(self, twin_object_id: Id, visible: bool) -> None:
        self._stores.objects.update(twin_object_id, {"visible": visible})

    def get(self, twin_object_id: Id) -> TwinObjectRecord | None:
        return self._stores.objects.get(twin_object_id)

    def list(self, **filter: Any) -> tuple[TwinObjectRecord, ...]:
        kind = filter.get("kind")
        aligned = filter.get("aligned")
        return self._stores.objects.query(
            lambda o: (kind is None or o.kind == kind) and (aligned is None or o.aligned == aligned)
        )

    async def link(
        self, twin_object_id: Id, elements: Sequence[ElementRef]
    ) -> Result[TwinObjectRecord, KernelError]:
        updated = self._stores.objects.update(
            twin_object_id, {"linked_elements": tuple(elements)}
        )
        return ok(updated) if updated else err(_not_found("twin object", twin_object_id))


class TwinAlignmentServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: TwinRuntime, stores: TwinStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def _record(
        self, twin_object_id: Id, method: str, transform: Sequence[float], rms: float | None,
        control_points: Sequence[Any] = (),
    ) -> TwinAlignmentRecord:
        record = TwinAlignmentRecord(
            id=self._runtime.ids.next("align"),
            twin_object_id=twin_object_id,
            method=method,  # type: ignore[arg-type]
            transform=tuple(transform),
            applied_at=self._runtime.clock.iso(),
            applied_by=self._runtime.context.permissions.identity.id,
            rms_error=rms,
            control_points=tuple(control_points),
        )
        self._stores.alignments.add(record)
        return record

    async def set_transform(
        self, twin_object_id: Id, transform: Sequence[float]
    ) -> Result[TwinAlignmentRecord, KernelError]:
        if not self._stores.objects.has(twin_object_id):
            return err(_not_found("twin object", twin_object_id))
        record = self._record(twin_object_id, "manual", transform, None)
        # A hand-placed transform sets `aligned`, but with no residual -- so a later reader can
        # tell "somebody dragged it into place" from "it was registered against control".
        self._stores.objects.update(
            twin_object_id,
            {"transform": tuple(transform), "aligned": True, "alignment_confidence": None},
        )
        self._runtime.context.events.emit(TWIN_EVENTS.aligned, {"record": record})
        return ok(record)

    async def align_by_points(
        self, twin_object_id: Id, pairs: Sequence[PointPair], *, allow_scale: bool = False
    ) -> Result[TwinAlignmentRecord, KernelError]:
        if not self._stores.objects.has(twin_object_id):
            return err(_not_found("twin object", twin_object_id))
        if len(pairs) < 2:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "A planar fit needs at least two control points; one determines a translation "
                    "and nothing else.",
                    {"twinObjectId": twin_object_id, "points": len(pairs)},
                )
            )

        fit = fit_planar(
            [pair.source for pair in pairs],
            [pair.target for pair in pairs],
            allow_scale=allow_scale,
        )
        if fit is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "The control points do not determine a transform -- they are coincident or "
                    "collinear to within tolerance.",
                    {"twinObjectId": twin_object_id},
                )
            )

        record = self._record(
            twin_object_id,
            "three-point" if len(pairs) >= 3 else "manual",
            fit.as_matrix(),
            round(fit.rms_error, 6),
            [(pair.source, pair.target) for pair in pairs],
        )
        # Confidence falls off with residual rather than being asserted. A fit is as trustworthy as
        # the control it was solved from, and saying so is the whole reason `rms_error` is stored.
        confidence = 1.0 / (1.0 + fit.rms_error)
        self._stores.objects.update(
            twin_object_id,
            {
                "transform": fit.as_matrix(),
                "aligned": True,
                "alignment_confidence": round(confidence, 4),
            },
        )
        self._runtime.context.events.emit(TWIN_EVENTS.aligned, {"record": record})
        return ok(record)

    def history(self, twin_object_id: Id) -> tuple[TwinAlignmentRecord, ...]:
        return self._stores.alignments.query(lambda a: a.twin_object_id == twin_object_id)

    async def revert(self, alignment_id: Id) -> Result[TwinAlignmentRecord, KernelError]:
        target = self._stores.alignments.get(alignment_id)
        if target is None:
            return err(_not_found("alignment", alignment_id))
        self._stores.objects.update(
            target.twin_object_id,
            {"transform": target.transform, "aligned": True},
        )
        return ok(target)


class TwinObservationServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: TwinRuntime, stores: TwinStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def record(self, **observation: Any) -> Result[TwinObservationRecord, KernelError]:
        twin_object_id = observation["twin_object_id"]
        if not self._stores.objects.has(twin_object_id):
            return err(_not_found("twin object", twin_object_id))
        record = TwinObservationRecord(
            id=self._runtime.ids.next("obs"),
            twin_object_id=twin_object_id,
            metric=observation["metric"],
            value=observation["value"],
            observed_at=observation.get("observed_at") or self._runtime.clock.iso(),
            unit=observation.get("unit"),
            provenance=observation.get("provenance"),
            quality=observation.get("quality"),
        )
        self._stores.observations.add(record)
        self._runtime.context.events.emit(TWIN_EVENTS.observed, {"record": record})
        return ok(record)

    async def record_many(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> Result[int, KernelError]:
        count = 0
        for observation in observations:
            result = await self.record(**dict(observation))
            if not result.ok:
                return err(result.error)
            count += 1
        return ok(count)

    def latest(self, twin_object_id: Id, metric: str) -> TwinObservationRecord | None:
        matching = [
            (record.observed_at, index, record)
            for index, record in enumerate(self._stores.observations.all())
            if record.twin_object_id == twin_object_id and record.metric == metric
        ]
        if not matching:
            return None
        # Insertion order breaks ties, as elsewhere: clock resolution is coarse enough that two
        # readings in one burst share a timestamp.
        return max(matching, key=lambda entry: (entry[0], entry[1]))[2]

    def query(self, **filter: Any) -> tuple[TwinObservationRecord, ...]:
        twin_object_id = filter.get("twin_object_id")
        metric = filter.get("metric")
        since = filter.get("since")
        until = filter.get("until")
        quality = filter.get("quality")
        return tuple(
            sorted(
                self._stores.observations.query(
                    lambda o: (twin_object_id is None or o.twin_object_id == twin_object_id)
                    and (metric is None or o.metric == metric)
                    and (since is None or o.observed_at >= since)
                    and (until is None or o.observed_at <= until)
                    and (quality is None or o.quality == quality)
                ),
                key=lambda o: o.observed_at,
            )
        )


class TwinTimelineServiceImpl:
    __slots__ = ("_runtime", "_stores", "_observations")

    def __init__(
        self,
        runtime: TwinRuntime,
        stores: TwinStores,
        observations: TwinObservationServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._observations = observations

    async def build(
        self, twin_object_id: Id, metric: str, from_time: IsoTimestamp, to_time: IsoTimestamp
    ) -> Result[TwinTimelineRecord, KernelError]:
        if not self._stores.objects.has(twin_object_id):
            return err(_not_found("twin object", twin_object_id))
        if to_time < from_time:
            return err(
                KernelError("COMMAND_FAILED", "The timeline ends before it starts.", {})
            )
        series = self._observations.query(
            twin_object_id=twin_object_id, metric=metric, since=from_time, until=to_time
        )
        record = TwinTimelineRecord(
            id=self._runtime.ids.next("tl"),
            twin_object_id=twin_object_id,
            metric=metric,
            from_time=from_time,
            to_time=to_time,
            observation_ids=tuple(observation.id for observation in series),
        )
        self._stores.timelines.add(record)
        return ok(record)

    def value_at(self, timeline_id: Id, at: IsoTimestamp) -> TwinObservationRecord | None:
        """The reading in force at an instant -- the last one at or before it.

        Not the nearest: a sensor reading taken after the moment you are asking about has not
        happened yet, and interpolating across it would invent data.
        """
        timeline = self._stores.timelines.get(timeline_id)
        if timeline is None:
            return None
        candidates = [
            self._stores.observations.get(observation_id)
            for observation_id in timeline.observation_ids
        ]
        eligible = [
            observation
            for observation in candidates
            if observation is not None and observation.observed_at <= at
        ]
        return max(eligible, key=lambda o: o.observed_at) if eligible else None


class TwinPromotionServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: TwinRuntime, stores: TwinStores) -> None:
        self._runtime = runtime
        self._stores = stores

    async def promote(
        self, twin_object_id: Id, target: str, **options: Any
    ) -> Result[TwinPromotionRecord, KernelError]:
        """Turn evidence into authored content, if the evidence supports it.

        The gate is measurability, and it lives on the schema so every consumer agrees about it. A
        bare Gaussian splat renders convincingly and has no surface, so promoting one to geometry
        would produce a wall whose dimensions mean nothing -- while cataloguing it as an ``asset``
        stays allowed, because that claims nothing about measurement.
        """
        record = self._stores.objects.get(twin_object_id)
        if record is None:
            return err(_not_found("twin object", twin_object_id))

        if target in MEASURABLE_TARGETS:
            reason = measurability_reason(record)
            if reason is not None:
                explanation = {
                    "visualization-only": "it is declared for visualisation, not for measurement",
                    "no-surface": "it is a radiance field with no derived mesh, so it has no "
                    "surface to measure against",
                }[reason]
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'"{record.name}" cannot be promoted to {target}: {explanation}.',
                        {"twinObjectId": twin_object_id, "reason": reason, "target": target},
                    )
                )
            if not record.aligned:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'"{record.name}" has not been aligned, so promoted geometry would land '
                        "wherever the capture happened to sit.",
                        {"twinObjectId": twin_object_id},
                    )
                )

        promotion = TwinPromotionRecord(
            id=self._runtime.ids.next("promo"),
            twin_object_id=twin_object_id,
            target=target,  # type: ignore[arg-type]
            target_id=options.get("target_id") or self._runtime.ids.next(target),
            promoted_at=self._runtime.clock.iso(),
            promoted_by=self._runtime.context.permissions.identity.id,
            notes=options.get("notes"),
        )
        self._stores.promotions.add(promotion)
        self._runtime.context.events.emit(TWIN_EVENTS.promoted, {"record": promotion})
        return ok(promotion)

    def origin_of(self, target_id: Id) -> TwinPromotionRecord | None:
        """What evidence produced this? Exactly the question people ask six months later."""
        return self._stores.promotions.find(lambda p: p.target_id == target_id)

    def history(self, twin_object_id: Id) -> tuple[TwinPromotionRecord, ...]:
        return self._stores.promotions.query(lambda p: p.twin_object_id == twin_object_id)
