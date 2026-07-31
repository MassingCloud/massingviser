"""``massingviser.plugins.families`` -- reusable content: repositories, resolution, placement.

Content here is **untrusted input**. A family package comes from a git repository, a vendor's
registry, or a colleague, and a platform that loads whatever it is handed has no way to refuse a
package built against an API it does not have. So packages are validated and can be refused, and
resolution is by semver range rather than by exact pin -- a project that pins every family to an
exact build never gets a fix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import (
    FamilyInstanceRecord,
    FamilyPackageRecord,
    FamilyParameterDefinition,
    FamilyRepositoryRecord,
    FamilyValidationResult,
    Id,
)


@dataclass(frozen=True)
class PackageQuery:
    text: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()
    repository_id: Id | None = None


@runtime_checkable
class FamilyRepositoryAdapter(Protocol):
    """Knows how to talk to one kind of content source.

    The *record* says where content lives; the adapter knows how to fetch it. That split is what
    lets a new repository kind arrive as a plugin instead of a schema change.
    """

    @property
    def kind(self) -> str: ...
    async def connect(self, record: FamilyRepositoryRecord) -> Result[None, KernelError]: ...
    async def discover(
        self, query: PackageQuery | None = None
    ) -> Result[Sequence[FamilyPackageRecord], KernelError]: ...
    async def versions(self, slug: str) -> Result[Sequence[str], KernelError]: ...
    async def fetch(self, slug: str, version: str) -> Result[FamilyPackageRecord, KernelError]: ...


FamilyRepositoryAdapterToken: CapabilityToken[FamilyRepositoryAdapter] = create_capability_token(
    "family.repository-adapter"
)


@runtime_checkable
class FamilyLibraryRegistryService(Protocol):
    async def add_repository(self, record: FamilyRepositoryRecord) -> Result[None, KernelError]: ...
    async def remove_repository(self, repository_id: Id) -> Result[None, KernelError]: ...
    def repositories(self) -> tuple[FamilyRepositoryRecord, ...]: ...
    async def sync(
        self, repository_id: Id | None = None
    ) -> Result[Mapping[str, Any], KernelError]: ...
    def search(self, query: PackageQuery | None = None) -> tuple[FamilyPackageRecord, ...]: ...


FamilyLibraryRegistryToken: CapabilityToken[FamilyLibraryRegistryService] = create_capability_token(
    "family.registry"
)


@runtime_checkable
class FamilyResolverService(Protocol):
    #: Highest cached version satisfying the range. A range, not a pin -- see the module docstring.
    async def resolve(
        self, slug: str, version_range: str | None = None
    ) -> Result[FamilyPackageRecord, KernelError]: ...
    def validate(self, package: FamilyPackageRecord) -> FamilyValidationResult: ...


FamilyResolverToken: CapabilityToken[FamilyResolverService] = create_capability_token(
    "family.resolver"
)


@dataclass(frozen=True)
class PlacementOptions:
    transform: tuple[float, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    model_id: Id | None = None
    level_id: Id | None = None
    host_element_id: str | int | None = None


@runtime_checkable
class FamilyPlacementService(Protocol):
    async def place(
        self, package_id: Id, version: str, options: PlacementOptions
    ) -> Result[FamilyInstanceRecord, KernelError]: ...
    async def move(
        self, instance_id: Id, transform: Sequence[float]
    ) -> Result[FamilyInstanceRecord, KernelError]: ...
    async def remove(self, instance_id: Id) -> Result[None, KernelError]: ...
    def instances(self, package_id: Id | None = None) -> tuple[FamilyInstanceRecord, ...]: ...


FamilyPlacementToken: CapabilityToken[FamilyPlacementService] = create_capability_token(
    "family.placement"
)


@runtime_checkable
class FamilyParameterService(Protocol):
    def definitions(self, package_id: Id) -> tuple[FamilyParameterDefinition, ...]: ...
    def get(self, instance_id: Id) -> Mapping[str, Any]: ...
    async def set(
        self, instance_id: Id, parameters: Mapping[str, Any]
    ) -> Result[FamilyInstanceRecord, KernelError]: ...
    #: Checked against the definitions before anything is written -- type, range and enum.
    def validate(
        self, package_id: Id, parameters: Mapping[str, Any]
    ) -> Result[None, KernelError]: ...


FamilyParameterToken: CapabilityToken[FamilyParameterService] = create_capability_token(
    "family.parameters"
)


@dataclass(frozen=True)
class UpgradeSummary:
    upgraded: int
    #: Instances whose parameters do not fit the new version. Left on the old one, and named --
    #: silently dropping a parameter is how an upgrade quietly changes a building.
    incompatible: tuple[tuple[Id, str], ...] = ()


@runtime_checkable
class FamilyVersionService(Protocol):
    def available(self, slug: str) -> tuple[str, ...]: ...
    async def upgrade(
        self, instance_ids: Sequence[Id], to_version: str
    ) -> Result[UpgradeSummary, KernelError]: ...


FamilyVersionToken: CapabilityToken[FamilyVersionService] = create_capability_token(
    "family.versions"
)


class FAMILY_COMMANDS:
    add_repository = "family.repository.add"
    sync_repositories = "family.repository.sync"
    search_packages = "family.package.search"
    place_instance = "family.instance.place"
    set_parameters = "family.instance.set-parameters"
    upgrade_instances = "family.instance.upgrade"


class FAMILY_PERMISSIONS:
    place = "family.place"
    manage_repositories = "family.repository.manage"
    publish = "family.publish"


class FAMILY_EVENTS:
    repository_synced = "family.repository.synced"
    instance_placed = "family.instance.placed"
    parameters_changed = "family.instance.parameters-changed"
    instances_upgraded = "family.instance.upgraded"
