from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .common import Id, IsoTimestamp, Provenance

FamilyRepositoryKind = Literal["git", "local", "cloud-api", "enterprise-registry", "project-local"]


@dataclass(frozen=True)
class FamilyRepositoryRecord:
    """A content source.

    Deliberately descriptive rather than behavioural: the record says *where* content lives and the
    matching adapter knows *how* to fetch it. That split is what lets a new repository kind be
    added as a plugin instead of a change to the schema.
    """

    id: Id
    name: str
    kind: FamilyRepositoryKind
    uri: str
    read_only: bool = True
    branch: str | None = None
    #: Whether this repository may receive newly published packages.
    publishable: bool = False
    trusted: bool = False
    last_synced_at: IsoTimestamp | None = None


FamilyParameterType = Literal["number", "string", "boolean", "length", "area", "enum"]


@dataclass(frozen=True)
class FamilyParameterDefinition:
    name: str
    type: FamilyParameterType
    unit: str | None = None
    default_value: Any = None
    options: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    required: bool = False
    description: str | None = None


@dataclass(frozen=True)
class FamilyAsset:
    kind: str
    uri: str


@dataclass(frozen=True)
class FamilyPackageRecord:
    """A versioned unit of reusable content."""

    id: Id
    repository_id: Id
    name: str
    #: Stable identifier within its repository, e.g. ``massingcloud/tower-typology``.
    slug: str
    version: str
    category: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    parameters: tuple[FamilyParameterDefinition, ...] = ()
    preview_uri: str | None = None
    #: Content locations, resolved by the owning repository adapter.
    assets: tuple[FamilyAsset, ...] = ()
    #: Platform API range this package is built against.
    api_version: str | None = None
    license: str | None = None
    published_at: IsoTimestamp | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class FamilyInstanceRecord:
    """A placed instance of a family package."""

    id: Id
    package_id: Id
    package_version: str
    created_at: IsoTimestamp
    created_by: Id
    #: The package's stable slug, captured at placement.
    #:
    #: Package *ids* are catalogue entries and do not survive a repository re-sync; the slug is the
    #: identity that does. Without it, re-syncing a library orphans every instance placed from it.
    package_slug: str | None = None
    name: str | None = None
    transform: tuple[float, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    #: Model the instance was placed into, when it has been realised as geometry.
    model_id: Id | None = None
    host_element_id: int | str | None = None
    level_id: Id | None = None


@dataclass(frozen=True)
class FamilyValidationResult:
    """Result of checking a package against the running platform.

    User-created content is explicitly in scope for this platform, so packages are treated as
    untrusted input: a package must be able to fail validation and be refused rather than being
    loaded and hoped for.
    """

    package_id: Id
    version: str
    compatible: bool
    checked_at: IsoTimestamp
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
