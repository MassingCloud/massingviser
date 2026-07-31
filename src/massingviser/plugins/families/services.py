from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...kernel import (
    KERNEL_API_VERSION,
    KernelError,
    PluginContext,
    Result,
    err,
    ok,
    parse_semver,
    satisfies,
)
from ...schema import (
    FamilyInstanceRecord,
    FamilyPackageRecord,
    FamilyParameterDefinition,
    FamilyRepositoryRecord,
    FamilyValidationResult,
    Id,
)
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    FAMILY_EVENTS,
    FamilyRepositoryAdapterToken,
    PackageQuery,
    PlacementOptions,
    UpgradeSummary,
)


@dataclass(frozen=True)
class FamilyStores:
    repositories: RecordStore[FamilyRepositoryRecord]
    packages: RecordStore[FamilyPackageRecord]
    instances: RecordStore[FamilyInstanceRecord]


@dataclass(frozen=True)
class FamilyRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_family_stores(context: PluginContext) -> FamilyStores:
    return FamilyStores(
        repositories=create_record_store(context.state, "repositories"),
        packages=create_record_store(context.state, "packages"),
        instances=create_record_store(context.state, "instances"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


class FamilyLibraryRegistryServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: FamilyRuntime, stores: FamilyStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def _adapter_for(self, record: FamilyRepositoryRecord) -> Any:
        for provider in self._runtime.context.capabilities.get_all(FamilyRepositoryAdapterToken):
            if provider.value.kind == record.kind:
                return provider.value
        return None

    async def add_repository(self, record: FamilyRepositoryRecord) -> Result[None, KernelError]:
        if self._stores.repositories.has(record.id):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Repository "{record.id}" is already registered.',
                    {"repositoryId": record.id},
                )
            )
        adapter = self._adapter_for(record)
        if adapter is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    f'No adapter handles repository kind "{record.kind}".',
                    {"kind": record.kind},
                )
            )
        connected = await adapter.connect(record)
        if not connected.ok:
            return err(connected.error)
        self._stores.repositories.add(record)
        return ok(None)

    async def remove_repository(self, repository_id: Id) -> Result[None, KernelError]:
        if not self._stores.repositories.remove(repository_id):
            return err(_not_found("repository", repository_id))
        # Catalogue entries go; placed instances stay. An instance keeps its slug and version, so
        # removing a library does not delete work already done with it.
        self._stores.packages.remove_where(lambda p: p.repository_id == repository_id)
        return ok(None)

    def repositories(self) -> tuple[FamilyRepositoryRecord, ...]:
        return self._stores.repositories.all()

    async def sync(self, repository_id: Id | None = None) -> Result[Mapping[str, Any], KernelError]:
        targets = (
            [r for r in self._stores.repositories.all() if r.id == repository_id]
            if repository_id
            else list(self._stores.repositories.all())
        )
        if repository_id and not targets:
            return err(_not_found("repository", repository_id))

        discovered = 0
        failures: list[tuple[Id, str]] = []
        stamp = self._runtime.clock.iso()

        for record in targets:
            adapter = self._adapter_for(record)
            if adapter is None:
                failures.append((record.id, f"no adapter for kind {record.kind}"))
                continue
            found = await adapter.discover(None)
            if not found.ok:
                # One unreachable library must not stop the rest syncing.
                failures.append((record.id, found.error.message))
                continue
            for package in found.value:
                # Re-syncing replaces the catalogue entry for a slug+version rather than stacking
                # duplicates; ids are catalogue-local and do not survive a re-sync, which is
                # exactly why instances key on the slug.
                self._stores.packages.remove_where(
                    lambda p, s=package.slug, v=package.version: p.slug == s and p.version == v
                )
                self._stores.packages.add(package)
                discovered += 1
            self._stores.repositories.update(record.id, {"last_synced_at": stamp})

        summary = {"discovered": discovered, "failures": tuple(failures)}
        self._runtime.context.events.emit(FAMILY_EVENTS.repository_synced, summary)
        return ok(summary)

    def search(self, query: PackageQuery | None = None) -> tuple[FamilyPackageRecord, ...]:
        if query is None:
            return self._stores.packages.all()
        text = (query.text or "").lower()
        return self._stores.packages.query(
            lambda p: (
                (
                    not text
                    or text in p.name.lower()
                    or text in p.slug.lower()
                    or text in (p.description or "").lower()
                )
                and (query.category is None or p.category == query.category)
                and (not query.tags or set(query.tags).issubset(set(p.tags)))
                and (query.repository_id is None or p.repository_id == query.repository_id)
            )
        )


class FamilyResolverServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: FamilyRuntime, stores: FamilyStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def validate(self, package: FamilyPackageRecord) -> FamilyValidationResult:
        """Check a package against the running platform.

        Content is untrusted input: a package must be able to *fail* and be refused rather than
        being loaded in hope.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if package.api_version and not satisfies(KERNEL_API_VERSION, package.api_version):
            errors.append(
                f'built against platform API "{package.api_version}"; this platform is '
                f"{KERNEL_API_VERSION}"
            )
        if parse_semver(package.version) is None:
            errors.append(f'version "{package.version}" is not a semantic version')

        seen: set[str] = set()
        for parameter in package.parameters:
            if parameter.name in seen:
                errors.append(f'duplicate parameter "{parameter.name}"')
            seen.add(parameter.name)
            if parameter.type == "enum" and not parameter.options:
                errors.append(f'enum parameter "{parameter.name}" lists no options')
            if (
                parameter.min is not None
                and parameter.max is not None
                and parameter.min > parameter.max
            ):
                errors.append(f'parameter "{parameter.name}" has min above max')
            if parameter.required and parameter.default_value is None:
                # Not fatal: a required parameter with no default just has to be supplied.
                warnings.append(f'required parameter "{parameter.name}" has no default')

        if not package.assets:
            warnings.append("package carries no assets")
        if not package.license:
            warnings.append("package states no licence")

        return FamilyValidationResult(
            package_id=package.id,
            version=package.version,
            compatible=not errors,
            checked_at=self._runtime.clock.iso(),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    async def resolve(
        self, slug: str, version_range: str | None = None
    ) -> Result[FamilyPackageRecord, KernelError]:
        candidates = self._stores.packages.query(lambda p: p.slug == slug)
        if not candidates:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'No package with slug "{slug}". Sync the repository first.',
                    {"slug": slug},
                )
            )

        matching = [
            package
            for package in candidates
            if version_range is None or satisfies(package.version, version_range)
        ]
        if not matching:
            return err(
                KernelError(
                    "CAPABILITY_VERSION_MISMATCH",
                    f'No version of "{slug}" satisfies "{version_range}".',
                    {"slug": slug, "available": [p.version for p in candidates]},
                )
            )

        # Highest satisfying version, ordered properly rather than lexically -- "0.10.0" sorts
        # below "0.9.0" as a string, which would resolve a range to the wrong build.
        def order(package: FamilyPackageRecord) -> tuple[int, int, int]:
            parsed = parse_semver(package.version)
            return (parsed.major, parsed.minor, parsed.patch) if parsed else (-1, -1, -1)

        best = max(matching, key=order)
        report = self.validate(best)
        if not report.compatible:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Package "{slug}@{best.version}" is not compatible: '
                    f"{'; '.join(report.errors)}",
                    {"slug": slug, "errors": list(report.errors)},
                )
            )
        return ok(best)


class FamilyParameterServiceImpl:
    __slots__ = ("_runtime", "_stores")

    def __init__(self, runtime: FamilyRuntime, stores: FamilyStores) -> None:
        self._runtime = runtime
        self._stores = stores

    def definitions(self, package_id: Id) -> tuple[FamilyParameterDefinition, ...]:
        package = self._stores.packages.get(package_id)
        return package.parameters if package else ()

    def get(self, instance_id: Id) -> Mapping[str, Any]:
        instance = self._stores.instances.get(instance_id)
        return dict(instance.parameters) if instance else {}

    def validate(self, package_id: Id, parameters: Mapping[str, Any]) -> Result[None, KernelError]:
        package = self._stores.packages.get(package_id)
        if package is None:
            return err(_not_found("package", package_id))

        definitions = {definition.name: definition for definition in package.parameters}
        problems: list[str] = []

        unknown = set(parameters) - set(definitions)
        if unknown:
            # Refused rather than ignored: a misspelled parameter that is silently dropped looks
            # exactly like one that was applied.
            problems.append(f"unknown parameter(s): {', '.join(sorted(unknown))}")

        for name, definition in definitions.items():
            if name not in parameters:
                if definition.required and definition.default_value is None:
                    problems.append(f'"{name}" is required')
                continue
            value = parameters[name]
            if definition.type in ("number", "length", "area"):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    problems.append(f'"{name}" must be a number')
                    continue
                if definition.min is not None and value < definition.min:
                    problems.append(f'"{name}" is below its minimum of {definition.min}')
                if definition.max is not None and value > definition.max:
                    problems.append(f'"{name}" is above its maximum of {definition.max}')
            elif definition.type == "boolean" and not isinstance(value, bool):
                problems.append(f'"{name}" must be true or false')
            elif definition.type == "string" and not isinstance(value, str):
                problems.append(f'"{name}" must be a string')
            elif definition.type == "enum" and value not in definition.options:
                problems.append(f'"{name}" must be one of {", ".join(definition.options)}')

        if problems:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "; ".join(problems),
                    {"packageId": package_id, "problems": problems},
                )
            )
        return ok(None)

    def _with_defaults(
        self, package: FamilyPackageRecord, parameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        resolved = {
            definition.name: definition.default_value
            for definition in package.parameters
            if definition.default_value is not None
        }
        resolved.update(parameters)
        return resolved

    async def set(
        self, instance_id: Id, parameters: Mapping[str, Any]
    ) -> Result[FamilyInstanceRecord, KernelError]:
        instance = self._stores.instances.get(instance_id)
        if instance is None:
            return err(_not_found("instance", instance_id))

        merged = {**instance.parameters, **parameters}
        checked = self.validate(instance.package_id, merged)
        if not checked.ok:
            return err(checked.error)

        updated = self._stores.instances.update(instance_id, {"parameters": merged})
        self._runtime.context.events.emit(
            FAMILY_EVENTS.parameters_changed, {"instanceId": instance_id}
        )
        return ok(updated) if updated else err(_not_found("instance", instance_id))


class FamilyPlacementServiceImpl:
    __slots__ = ("_runtime", "_stores", "_parameters")

    def __init__(
        self,
        runtime: FamilyRuntime,
        stores: FamilyStores,
        parameters: FamilyParameterServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._parameters = parameters

    async def place(
        self, package_id: Id, version: str, options: PlacementOptions
    ) -> Result[FamilyInstanceRecord, KernelError]:
        package = self._stores.packages.get(package_id)
        if package is None:
            return err(_not_found("package", package_id))
        if package.version != version:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Package "{package.slug}" is at {package.version}, not {version}.',
                    {"packageId": package_id},
                )
            )

        merged = self._parameters._with_defaults(package, options.parameters)
        checked = self._parameters.validate(package_id, merged)
        if not checked.ok:
            return err(checked.error)

        record = FamilyInstanceRecord(
            id=self._runtime.ids.next("inst"),
            package_id=package_id,
            package_version=version,
            created_at=self._runtime.clock.iso(),
            created_by=self._runtime.context.permissions.identity.id,
            # Captured at placement. Package ids are catalogue entries and do not survive a
            # re-sync; the slug is the identity that does.
            package_slug=package.slug,
            name=options.name or package.name,
            transform=tuple(options.transform),
            parameters=merged,
            model_id=options.model_id,
            host_element_id=options.host_element_id,
            level_id=options.level_id,
        )
        self._stores.instances.add(record)
        self._runtime.context.events.emit(FAMILY_EVENTS.instance_placed, {"record": record})
        return ok(record)

    async def move(
        self, instance_id: Id, transform: Sequence[float]
    ) -> Result[FamilyInstanceRecord, KernelError]:
        updated = self._stores.instances.update(instance_id, {"transform": tuple(transform)})
        return ok(updated) if updated else err(_not_found("instance", instance_id))

    async def remove(self, instance_id: Id) -> Result[None, KernelError]:
        return (
            ok(None)
            if self._stores.instances.remove(instance_id)
            else err(_not_found("instance", instance_id))
        )

    def instances(self, package_id: Id | None = None) -> tuple[FamilyInstanceRecord, ...]:
        if package_id is None:
            return self._stores.instances.all()
        return self._stores.instances.query(lambda i: i.package_id == package_id)


class FamilyVersionServiceImpl:
    __slots__ = ("_runtime", "_stores", "_parameters")

    def __init__(
        self,
        runtime: FamilyRuntime,
        stores: FamilyStores,
        parameters: FamilyParameterServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._parameters = parameters

    def available(self, slug: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    package.version
                    for package in self._stores.packages.query(lambda p: p.slug == slug)
                }
            )
        )

    async def upgrade(
        self, instance_ids: Sequence[Id], to_version: str
    ) -> Result[UpgradeSummary, KernelError]:
        """Move instances onto a newer version of the same family.

        An instance whose parameters do not fit the new definitions is **left where it is** and
        named. Migrating it by dropping the offending parameter would change the building without
        telling anyone.
        """
        upgraded = 0
        incompatible: list[tuple[Id, str]] = []

        for instance_id in instance_ids:
            instance = self._stores.instances.get(instance_id)
            if instance is None:
                incompatible.append((instance_id, "no such instance"))
                continue
            slug = instance.package_slug
            if slug is None:
                incompatible.append((instance_id, "instance has no slug to resolve against"))
                continue

            target = self._stores.packages.find(
                lambda p, s=slug, v=to_version: p.slug == s and p.version == v
            )
            if target is None:
                incompatible.append((instance_id, f"{slug}@{to_version} is not in the catalogue"))
                continue

            checked = self._parameters.validate(target.id, dict(instance.parameters))
            if not checked.ok:
                incompatible.append((instance_id, checked.error.message))
                continue

            self._stores.instances.update(
                instance_id, {"package_id": target.id, "package_version": to_version}
            )
            upgraded += 1

        summary = UpgradeSummary(upgraded=upgraded, incompatible=tuple(incompatible))
        self._runtime.context.events.emit(FAMILY_EVENTS.instances_upgraded, {"summary": summary})
        return ok(summary)
