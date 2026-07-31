"""``massingviser.plugins.twin`` -- captured reality alongside the authored model.

Twin records stay deliberately loose about BIM semantics. A scan, a sensor and a generated mesh are
*evidence about the world*, not authored building elements, and forcing them into IFC semantics on
arrival destroys information. Turning evidence into geometry is an explicit, later, **gated**
decision -- and the gate is measurability, because a radiance field renders convincingly and
measures badly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    ElementRef,
    Id,
    IsoTimestamp,
    TwinAlignmentRecord,
    TwinObjectRecord,
    TwinObservationRecord,
    TwinPromotionRecord,
    TwinTimelineRecord,
)


@runtime_checkable
class TwinObjectFactory(Protocol):
    """Turns a record into something a renderer can show.

    Registered per ``kind``, so support for a new capture format arrives as a plugin. The twin
    plugin itself never learns what a Gaussian splat is made of.
    """

    @property
    def kind(self) -> str: ...
    async def create(self, record: TwinObjectRecord) -> Result[Any, KernelError]: ...
    def dispose(self, runtime_object: Any) -> None: ...


TwinObjectFactoryToken: CapabilityToken[TwinObjectFactory] = create_capability_token("twin.factory")


@runtime_checkable
class TwinRegistryService(Protocol):
    async def register(self, record: TwinObjectRecord) -> Result[TwinObjectRecord, KernelError]: ...
    async def unregister(self, twin_object_id: Id) -> Result[None, KernelError]: ...
    async def materialise(self, twin_object_id: Id) -> Result[Any, KernelError]: ...
    def set_visible(self, twin_object_id: Id, visible: bool) -> None: ...
    def get(self, twin_object_id: Id) -> TwinObjectRecord | None: ...
    def list(self, **filter: Any) -> tuple[TwinObjectRecord, ...]: ...
    async def link(
        self, twin_object_id: Id, elements: Sequence[ElementRef]
    ) -> Result[TwinObjectRecord, KernelError]: ...


TwinRegistryToken: CapabilityToken[TwinRegistryService] = create_capability_token("twin.registry")


@dataclass(frozen=True)
class PointPair:
    source: tuple[float, float, float]
    target: tuple[float, float, float]


@runtime_checkable
class TwinAlignmentService(Protocol):
    async def set_transform(
        self, twin_object_id: Id, transform: Sequence[float]
    ) -> Result[TwinAlignmentRecord, KernelError]: ...
    #: Least-squares fit from matched control points. Reports its residual, always.
    async def align_by_points(
        self, twin_object_id: Id, pairs: Sequence[PointPair], *, allow_scale: bool = False
    ) -> Result[TwinAlignmentRecord, KernelError]: ...
    def history(self, twin_object_id: Id) -> tuple[TwinAlignmentRecord, ...]: ...
    async def revert(self, alignment_id: Id) -> Result[TwinAlignmentRecord, KernelError]: ...


TwinAlignmentToken: CapabilityToken[TwinAlignmentService] = create_capability_token(
    "twin.alignment"
)


@runtime_checkable
class TwinObservationService(Protocol):
    async def record(self, **observation: Any) -> Result[TwinObservationRecord, KernelError]: ...
    async def record_many(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> Result[int, KernelError]: ...
    def latest(self, twin_object_id: Id, metric: str) -> TwinObservationRecord | None: ...
    def query(self, **filter: Any) -> tuple[TwinObservationRecord, ...]: ...


TwinObservationToken: CapabilityToken[TwinObservationService] = create_capability_token(
    "twin.observations"
)


@runtime_checkable
class TwinTimelineService(Protocol):
    async def build(
        self, twin_object_id: Id, metric: str, from_time: IsoTimestamp, to_time: IsoTimestamp
    ) -> Result[TwinTimelineRecord, KernelError]: ...
    def value_at(self, timeline_id: Id, at: IsoTimestamp) -> TwinObservationRecord | None: ...


TwinTimelineToken: CapabilityToken[TwinTimelineService] = create_capability_token("twin.timeline")


PromotionTarget = "authoring | family | asset"


@runtime_checkable
class TwinPromotionService(Protocol):
    #: Refused for anything that cannot be measured, with the reason -- except ``asset``, because
    #: cataloguing a splat is perfectly legitimate.
    async def promote(
        self, twin_object_id: Id, target: str, **options: Any
    ) -> Result[TwinPromotionRecord, KernelError]: ...
    def origin_of(self, target_id: Id) -> TwinPromotionRecord | None: ...
    def history(self, twin_object_id: Id) -> tuple[TwinPromotionRecord, ...]: ...


TwinPromotionToken: CapabilityToken[TwinPromotionService] = create_capability_token(
    "twin.promotion"
)


class TWIN_COMMANDS:
    register = "twin.register"
    materialise = "twin.materialise"
    align_by_points = "twin.align.points"
    set_transform = "twin.align.set-transform"
    record_observation = "twin.observation.record"
    promote = "twin.promote"


class TWIN_PERMISSIONS:
    register = "twin.register"
    align = "twin.align"
    promote = "twin.promote"


class TWIN_EVENTS:
    registered = "twin.registered"
    aligned = "twin.aligned"
    observed = "twin.observed"
    promoted = "twin.promoted"
