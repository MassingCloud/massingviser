from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..kernel import (
    KernelError,
    Result,
    VersionedDocument,
    attempt,
    err,
    ok,
)


@dataclass(frozen=True)
class MigrationDefinition:
    #: Schema identifier this migration applies to, e.g. ``"massingifc.massing.object"``.
    schema: str
    from_version: int
    to_version: int
    #: Pure transform. Must not mutate its input -- the caller may still hold the original.
    migrate: Callable[[Any], Any] = field(repr=False, default=lambda data: data)
    #: Human-readable note. Surfaced in migration reports and logs.
    description: str | None = None


@dataclass(frozen=True)
class MigrationPlan:
    schema: str
    from_version: int
    to_version: int
    steps: tuple[MigrationDefinition, ...]


class MigrationRegistry:
    """The platform's upgrade path.

    Every persisted format in the platform carries a schema id and an integer version, and this
    registry owns the rules for moving one forward. Keeping it separate from the persistence engine
    matters: the kernel must not know what a massing object is, and plugins must be able to
    register migrations for their own records at load time without patching the backbone.

    Implements the kernel's ``DocumentMigrator``, so handing an instance to ``create_kernel`` is
    all the wiring required.
    """

    __slots__ = ("_steps", "_latest")

    def __init__(self) -> None:
        #: schema -> from_version -> definition
        self._steps: dict[str, dict[int, MigrationDefinition]] = {}
        self._latest: dict[str, int] = {}

    def declare(self, schema: str, version: int) -> MigrationRegistry:
        """Declare the current version of a schema that has no migrations yet.

        Needed because "latest" cannot always be inferred: a brand-new v1 schema has no steps, and
        without an explicit declaration the engine would treat it as unknown and skip version
        checks entirely -- including the forward-incompatibility guard.
        """
        if version > self._latest.get(schema, 0):
            self._latest[schema] = version
        return self

    def register(self, definition: MigrationDefinition) -> MigrationRegistry:
        if definition.to_version <= definition.from_version:
            raise KernelError(
                "MIGRATION_FAILED",
                f'Migration for "{definition.schema}" must move forward '
                f"(got v{definition.from_version} -> v{definition.to_version}).",
                {
                    "schema": definition.schema,
                    "from": definition.from_version,
                    "to": definition.to_version,
                },
            )
        by_schema = self._steps.setdefault(definition.schema, {})
        if definition.from_version in by_schema:
            # Two routes out of the same version make the upgrade non-deterministic -- which one
            # runs would depend on registration order, and the two could disagree about the result.
            raise KernelError(
                "MIGRATION_FAILED",
                f'Duplicate migration from v{definition.from_version} for "{definition.schema}".',
                {"schema": definition.schema, "from": definition.from_version},
            )
        by_schema[definition.from_version] = definition
        self.declare(definition.schema, definition.to_version)
        return self

    def register_all(self, definitions: Sequence[MigrationDefinition]) -> MigrationRegistry:
        for definition in definitions:
            self.register(definition)
        return self

    def latest_version(self, schema: str) -> int | None:
        return self._latest.get(schema)

    def known_schemas(self) -> tuple[str, ...]:
        return tuple(sorted(self._latest))

    def plan(self, schema: str, from_version: int) -> Result[MigrationPlan, KernelError]:
        """Resolve the chain of steps from ``from_version`` to the schema's latest version."""
        target = self._latest.get(schema)
        if target is None:
            return err(
                KernelError(
                    "MIGRATION_PATH_MISSING", f'Unknown schema "{schema}".', {"schema": schema}
                )
            )
        steps: list[MigrationDefinition] = []
        version = from_version
        by_schema = self._steps.get(schema, {})

        while version < target:
            step = by_schema.get(version)
            if step is None:
                return err(
                    KernelError(
                        "MIGRATION_PATH_MISSING",
                        f'No migration from "{schema}" v{version} towards v{target}.',
                        {"schema": schema, "from": version, "target": target},
                    )
                )
            steps.append(step)
            version = step.to_version
        return ok(MigrationPlan(schema, from_version, version, tuple(steps)))

    def migrate(
        self, document: VersionedDocument[Any]
    ) -> Result[VersionedDocument[Any], KernelError]:
        target = self._latest.get(document.schema)
        if target is None or document.version == target:
            return ok(document)
        if document.version > target:
            return err(
                KernelError(
                    "SCHEMA_VERSION_UNSUPPORTED",
                    f'"{document.schema}" v{document.version} is newer than the '
                    f"supported v{target}.",
                    {
                        "schema": document.schema,
                        "found": document.version,
                        "supported": target,
                    },
                )
            )

        plan = self.plan(document.schema, document.version)
        if not plan.ok:
            return err(plan.error)

        data = document.data
        version = document.version
        for step in plan.value.steps:
            # A migration is third-party code with a long tail of edge cases (a field that was
            # optional in practice but not in the type, a None where a list was assumed). Failing
            # this one document with the step named beats an unhandled raise that loses the whole
            # project.
            applied = attempt(
                # Bound as defaults rather than captured. `attempt` runs this immediately, so late
                # binding is harmless today -- but a lambda in a loop that closes over the loop
                # variable is one refactor away from migrating every document with the last step.
                lambda step=step, data=data: step.migrate(data),
                "MIGRATION_FAILED",
                f'Migration "{document.schema}" v{step.from_version} -> v{step.to_version} failed.',
            )
            if not applied.ok:
                return err(
                    KernelError(
                        "MIGRATION_FAILED",
                        f'Migration "{document.schema}" v{step.from_version} -> '
                        f"v{step.to_version} failed: {applied.error.message}",
                        {
                            "schema": document.schema,
                            "from": step.from_version,
                            "to": step.to_version,
                        },
                        cause=applied.error,
                    )
                )
            data = applied.value
            version = step.to_version
        return ok(replace(document, version=version, data=data))

    def is_compatible(self, schema: str, version: int) -> bool:
        """Whether a document can be brought to the current version.

        Cheap, side-effect-free check for a compatibility screen -- a host can tell a user which
        files in a project directory will open before opening any of them.
        """
        target = self._latest.get(schema)
        if target is None or version > target:
            return False
        if version == target:
            return True
        return self.plan(schema, version).ok


def create_default_migration_registry() -> MigrationRegistry:
    """A registry with every shipped schema declared at its current version.

    Declaring them all up front is what makes the forward-incompatibility guard work from the first
    release rather than from the first migration.
    """
    from .schemas import CURRENT_VERSION

    registry = MigrationRegistry()
    for schema, version in CURRENT_VERSION.items():
        registry.declare(schema, version)
    return registry
