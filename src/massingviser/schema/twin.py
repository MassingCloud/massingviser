from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .common import ElementRef, Id, IsoTimestamp, Provenance
from .geo import DatasetPurpose, Extent, GeoReference, RealityDerivatives

TwinObjectKind = Literal[
    "three-group",
    "gltf",
    "point-cloud",
    #: A radiance-field scene of oriented Gaussians.
    #:
    #: Kept a first-class kind rather than folded into "mesh-scan" because it is not a surface: it
    #: renders convincingly and measures badly, and the platform needs to tell them apart.
    "gaussian-splat",
    "mesh-scan",
    "image-anchor",
    "sensor",
    "equipment",
    "other",
]


@dataclass(frozen=True)
class TwinObjectRecord:
    """An observed or generated object living alongside the BIM models.

    Twin records stay deliberately loose about BIM semantics. A scan, a sensor and a generated mesh
    group are *evidence about the world*, not authored building elements -- forcing them into IFC
    semantics on arrival destroys information and blocks the workflows that make twins useful.
    Promotion into authored geometry is an explicit, later decision (``TwinPromotionRecord``).
    """

    id: Id
    name: str
    kind: TwinObjectKind
    created_at: IsoTimestamp
    provenance: Provenance
    #: Where the runtime object comes from -- a URI, or a factory id for generated content.
    source_uri: str | None = None
    factory_id: str | None = None
    transform: tuple[float, ...] = ()
    #: Alignment quality, 0..1. Distinct from ``provenance.confidence``, which is about the source.
    alignment_confidence: float | None = None
    aligned: bool = False
    visible: bool = True
    #: Where on Earth this sits.
    #:
    #: Distinct from ``transform``, which only places it relative to the project origin. Without
    #: this a scan cannot be checked against a survey, combined with GIS layers, or re-registered
    #: after the project origin moves.
    geo_reference: GeoReference | None = None
    extent: Extent | None = None
    #: Products from the same capture -- orthomosaic, point cloud, derived mesh, source imagery.
    derivatives: RealityDerivatives | None = None
    #: What this dataset may legitimately be used for. Defaults to context when unstated.
    purpose: DatasetPurpose | None = None
    captured_at: IsoTimestamp | None = None
    #: BIM elements this twin object is understood to correspond to.
    linked_elements: tuple[ElementRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: Why a dataset may not back a measurement.
UnmeasurableReason = Literal[
    #: Declared for looking at, not for taking numbers off.
    "visualization-only",
    #: A radiance field with no mesh derived from it -- there is no surface to measure against.
    "no-surface",
]


def measurability_reason(record: TwinObjectRecord) -> UnmeasurableReason | None:
    """Whether a dataset may back a measurement or a derived authored object.

    Lives on the schema rather than in the twin service because it is a property of the *record*,
    and everything that consumes twin records has to agree about it -- the promotion gate, a
    measurement tool, an engine exporter marking a layer as collidable. A second copy of this rule
    anywhere is a place where a splat quietly becomes measurable.
    """
    if record.purpose == "visualization":
        return "visualization-only"
    if record.kind == "gaussian-splat" and (
        record.derivatives is None or record.derivatives.mesh_uri is None
    ):
        return "no-surface"
    return None


def is_measurable(record: TwinObjectRecord) -> bool:
    return measurability_reason(record) is None


@dataclass(frozen=True)
class ControlPoint:
    source: tuple[float, float, float]
    target: tuple[float, float, float]


@dataclass(frozen=True)
class TwinAlignmentRecord:
    """A registration attempt that moved a twin object into project coordinates."""

    id: Id
    twin_object_id: Id
    method: Literal["manual", "three-point", "icp", "survey", "gps"]
    transform: tuple[float, ...]
    applied_at: IsoTimestamp
    applied_by: Id
    #: Root-mean-square residual in project units. Lower is better.
    rms_error: float | None = None
    control_points: tuple[ControlPoint, ...] = ()


@dataclass(frozen=True)
class TwinObservationRecord:
    """A time-stamped measurement or state reading attached to a twin object."""

    id: Id
    twin_object_id: Id
    metric: str
    value: float | str | bool
    observed_at: IsoTimestamp
    unit: str | None = None
    provenance: Provenance | None = None
    quality: Literal["good", "suspect", "bad"] | None = None


@dataclass(frozen=True)
class TwinTimelineRecord:
    """An ordered series used for playback and trend review."""

    id: Id
    twin_object_id: Id
    metric: str
    from_time: IsoTimestamp
    to_time: IsoTimestamp
    observation_ids: tuple[Id, ...] = ()


@dataclass(frozen=True)
class TwinPromotionRecord:
    """Record of a twin object being turned into authored content.

    Kept as its own record so the link back to the observation survives. Once a scan becomes a
    wall, the question "what evidence produced this?" is exactly the one people ask six months
    later.
    """

    id: Id
    twin_object_id: Id
    target: Literal["authoring", "family", "asset"]
    target_id: Id
    promoted_at: IsoTimestamp
    promoted_by: Id
    notes: str | None = None
