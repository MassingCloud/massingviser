from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .common import Id, IsoTimestamp, Provenance
from .geo import GeoReference

ModelRole = Literal[
    #: Published, read-only reference from another discipline or consultant.
    "reference",
    #: Editable model this team authors.
    "working",
    #: Generated from massing, twin promotion, or another derivation.
    "derived",
]

ModelFormat = Literal["ifc", "fragments", "gltf", "obj", "point-cloud", "other"]


@dataclass(frozen=True)
class ModelRecord:
    """A model participating in the federation.

    ``version`` is a string rather than a number because it usually comes from somebody else's
    issue sheet ("C03", "2026-04-17-P2"), and normalising that to an integer loses the only
    identifier the wider project team actually recognises.
    """

    id: Id
    name: str
    role: ModelRole
    format: ModelFormat
    version: str
    source_uri: str | None = None
    #: Location of the converted payload, when conversion has happened.
    fragments_uri: str | None = None
    discipline: str | None = None
    #: Placement relative to the project origin, for models delivered on a different datum.
    transform: tuple[float, ...] | None = None
    #: Where the model sits on Earth.
    #:
    #: Distinct from ``transform``, and needed precisely in the case that comment describes: once a
    #: model arrives on a different datum, a matrix onto the project origin records where someone
    #: *put* it, not where it belongs. Only the georeference lets a discrepancy be settled against
    #: survey rather than argued about.
    geo_reference: GeoReference | None = None
    visible: bool = True
    load_by_default: bool = True
    element_count: int | None = None
    imported_at: IsoTimestamp | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class ProjectLocation:
    """Kept separate so a project can be relocated without touching model records."""

    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    #: Rotation from project north to true north, in degrees clockwise.
    true_north_angle: float | None = None
    epsg_code: str | None = None
    address: str | None = None


@dataclass(frozen=True)
class ProjectUnits:
    length: Literal["m", "mm", "ft", "in"] = "m"
    area: str = "m2"
    volume: str = "m3"
    currency: str = "GBP"


@dataclass(frozen=True)
class ProjectRecord:
    """The container everything else hangs off.

    Domain collections are referenced by id rather than embedded. A federated project accumulates
    tens of thousands of markups, quantities and observations; embedding them would mean rewriting
    the whole project document on every pin drop, and would make per-plugin schema versioning
    impossible.
    """

    id: Id
    name: str
    created_at: IsoTimestamp
    created_by: Id
    units: ProjectUnits = field(default_factory=ProjectUnits)
    number: str | None = None
    description: str | None = None
    client: str | None = None
    phase: str | None = None
    location: ProjectLocation | None = None
    model_ids: tuple[Id, ...] = ()
    #: Repositories this project may resolve family content from.
    family_repository_ids: tuple[Id, ...] = ()
    updated_at: IsoTimestamp | None = None


@dataclass(frozen=True)
class SessionStateRecord:
    """A saved session: which models were loaded and what the user was looking at.

    Reopening a federated project should not mean reloading twelve models and re-hiding nine of
    them, so load state is persisted explicitly rather than reconstructed.
    """

    id: Id
    project_id: Id
    saved_at: IsoTimestamp
    saved_by: Id
    loaded_model_ids: tuple[Id, ...] = ()
    active_viewpoint_id: Id | None = None
    open_panels: tuple[str, ...] = ()
