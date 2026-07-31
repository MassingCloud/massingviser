from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .common import Id, IsoTimestamp

Vec3 = "tuple[float, float, float]"


@dataclass(frozen=True)
class ProfileRecord:
    """A closed sketch outline that a mass is extruded from.

    Held separately from the mass so several options can share one footprint, and so editing the
    footprint updates every option built on it -- the common case in early design.
    """

    id: Id
    #: Outer boundary, in project coordinates. Implicitly closed; do not repeat the first point.
    points: tuple[tuple[float, float, float], ...]
    closed: bool = True
    name: str | None = None
    #: Openings -- courtyards, light wells, atria.
    holes: tuple[tuple[tuple[float, float, float], ...], ...] = ()
    #: Elevation of the sketch plane.
    base_elevation: float = 0.0


@dataclass(frozen=True)
class MassingObjectRecord:
    """A conceptual volume.

    ``story_heights`` is authoritative over ``total_height`` -- a mass with per-story heights is
    the normal case, and a single overall height cannot express a taller ground floor, which is
    close to universal in real buildings.
    """

    id: Id
    name: str
    profile_id: Id
    story_count: int
    story_heights: tuple[float, ...] = ()
    total_height: float = 0.0
    color: str | None = None
    opacity: float | None = None
    area: float | None = None
    gross_floor_area: float | None = None
    volume: float | None = None
    option_set_id: Id | None = None
    family_template_id: Id | None = None
    editable: bool = True


@dataclass(frozen=True)
class MassingStoryRecord:
    """One level of a mass, derived from the mass's profile and story heights."""

    id: Id
    massing_object_id: Id
    #: 0-based from the base of the mass.
    index: int
    elevation: float
    height: float
    name: str | None = None
    #: Per-story override; falls back to the parent mass's profile when absent.
    profile_id: Id | None = None
    area: float | None = None
    gross_floor_area: float | None = None
    programme: str | None = None
    excluded_from_gfa: bool = False


@dataclass(frozen=True)
class OptionSetRecord:
    """Named design alternative grouping several masses for side-by-side comparison."""

    id: Id
    name: str
    created_at: IsoTimestamp
    massing_object_ids: tuple[Id, ...] = ()
    description: str | None = None
    active: bool = False


@dataclass(frozen=True)
class MassingMetrics:
    """Derived quantities. Recomputed from geometry, never hand-edited."""

    massing_object_id: Id
    footprint_area: float
    gross_floor_area: float
    volume: float
    envelope_area: float
    story_count: int
    height: float
    computed_at: IsoTimestamp
    #: GFA divided by site area, when a site boundary is known.
    floor_area_ratio: float | None = None


@dataclass(frozen=True)
class LevelRecord:
    id: Id
    name: str
    elevation: float
    is_structural: bool = False


@dataclass(frozen=True)
class GridLineRecord:
    id: Id
    label: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]


@dataclass(frozen=True)
class SetbackRule:
    edge_index: int
    distance: float


@dataclass(frozen=True)
class SiteBoundaryRecord:
    id: Id
    points: tuple[tuple[float, float, float], ...]
    name: str | None = None
    area: float | None = None
    #: Planning limits that massing option studies are checked against.
    max_floor_area_ratio: float | None = None
    max_height: float | None = None
    setbacks: tuple[SetbackRule, ...] = ()
