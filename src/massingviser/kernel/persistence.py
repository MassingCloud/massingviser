from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, runtime_checkable

from .errors import KernelError
from .result import Err, Ok, Result, attempt_async, err, ok
from .telemetry import NOOP_TELEMETRY, TelemetrySink

T = TypeVar("T")


@runtime_checkable
class StorageAdapter(Protocol):
    """Byte-agnostic key/value persistence.

    Kept this narrow so the same engine works over a filesystem, an object store, or an in-memory
    dict -- the kernel never learns which.
    """

    async def get(self, key: str) -> Any | None: ...
    async def put(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def keys(self, prefix: str = "") -> list[str]: ...


class MemoryStorageAdapter:
    """In-memory adapter for tests and ephemeral sessions.

    Deep-copies on both read and write. Handing back a live reference would let a caller mutate
    stored state without saving -- a bug that only shows up once a real adapter is swapped in,
    which is the worst time to find it.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        value = self._entries.get(key)
        return None if value is None else copy.deepcopy(value)

    async def put(self, key: str, value: Any) -> None:
        self._entries[key] = copy.deepcopy(value)

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    async def keys(self, prefix: str = "") -> list[str]:
        return sorted(key for key in self._entries if key.startswith(prefix))

    @property
    def size(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class VersionedDocument(Generic[T]):
    """Every persisted payload carries its schema identity and version inline.

    Storing the version *with* the data rather than alongside it is what makes migration possible
    years later: a file recovered from a backup, an email attachment, or a different deployment is
    still self-describing.
    """

    schema: str
    version: int
    saved_at: str
    data: T

    def to_dict(self) -> dict[str, Any]:
        # Written in the wire shape massingifc uses, so a document round-trips between the
        # TypeScript and Python implementations unchanged.
        return {
            "schema": self.schema,
            "version": self.version,
            "savedAt": self.saved_at,
            "data": self.data,
        }

    @staticmethod
    def from_mapping(value: Mapping[str, Any]) -> "VersionedDocument[Any]":
        return VersionedDocument(
            schema=value["schema"],
            version=value["version"],
            saved_at=value.get("savedAt", ""),
            data=value.get("data"),
        )


def is_versioned_document(value: Any) -> bool:
    if isinstance(value, VersionedDocument):
        return True
    if not isinstance(value, Mapping):
        return False
    version = value.get("version")
    return (
        isinstance(value.get("schema"), str)
        and isinstance(version, int)
        and not isinstance(version, bool)
    )


def _as_document(value: Any) -> VersionedDocument[Any]:
    return value if isinstance(value, VersionedDocument) else VersionedDocument.from_mapping(value)


@runtime_checkable
class DocumentMigrator(Protocol):
    """Upgrades documents to the current schema version.

    The kernel defines the interface but ships no implementation -- migration rules belong with the
    records they migrate (see ``massingviser.schema``), not in the backbone.
    """

    def latest_version(self, schema: str) -> int | None: ...
    def migrate(
        self, document: VersionedDocument[Any]
    ) -> "Result[VersionedDocument[Any], KernelError]": ...


@dataclass(frozen=True)
class BackupEntry:
    id: str
    key: str
    created_at: str


BACKUP_MARKER = "::backup::"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


class PersistenceEngine:
    """Versioned document store with migration and rollback.

    Reads are pure: ``load`` migrates the value it returns but does not write the upgraded form
    back. A read that silently rewrites storage is a surprising and occasionally destructive side
    effect -- opening a project to look at it should not mutate it. Callers that *want* the upgrade
    persisted ask for it explicitly via ``migrate_in_place``, which takes a backup first.
    """

    __slots__ = ("_adapter", "_migrator", "_telemetry", "_now", "_max_backups", "_backup_counter")

    def __init__(
        self,
        *,
        adapter: StorageAdapter,
        migrator: DocumentMigrator | None = None,
        telemetry: TelemetrySink | None = None,
        now: Callable[[], datetime] | None = None,
        max_backups: int = 5,
    ) -> None:
        self._adapter = adapter
        self._migrator = migrator
        self._telemetry = telemetry or NOOP_TELEMETRY
        self._now = now or _utc_now
        self._max_backups = max(0, max_backups)
        self._backup_counter = 0

    async def save(
        self,
        key: str,
        schema: str,
        data: T,
        *,
        backup: bool = False,
        version: int | None = None,
    ) -> Result[VersionedDocument[T], KernelError]:
        if backup:
            snapshot = await self.backup(key)
            if not snapshot.ok and snapshot.error.code != "STORAGE_FAILED":
                return err(snapshot.error)

        resolved_version = version
        if resolved_version is None and self._migrator is not None:
            resolved_version = self._migrator.latest_version(schema)
        if resolved_version is None:
            resolved_version = 1

        document = VersionedDocument(
            schema=schema, version=resolved_version, saved_at=_iso(self._now()), data=data
        )
        written = await attempt_async(
            lambda: self._adapter.put(key, document.to_dict()),
            "STORAGE_FAILED",
            f'Failed to write "{key}".',
        )
        if not written.ok:
            self._telemetry.error(written.error, {"key": key, "schema": schema})
            return err(written.error)
        self._telemetry.counter("persistence.save", 1, {"schema": schema})
        return ok(document)

    async def load(self, key: str) -> Result[VersionedDocument[Any] | None, KernelError]:
        """Load and migrate.

        Resolves to ``None`` for a key that was never written -- absence is a normal outcome (a
        first run), not a failure.
        """
        read = await attempt_async(
            lambda: self._adapter.get(key), "STORAGE_FAILED", f'Failed to read "{key}".'
        )
        if not read.ok:
            return err(read.error)
        if read.value is None:
            return ok(None)

        if not is_versioned_document(read.value):
            return err(
                KernelError(
                    "MIGRATION_FAILED",
                    f'Value at "{key}" is not a versioned document.',
                    {"key": key},
                )
            )
        return self._apply_migration(_as_document(read.value), key)

    async def migrate_in_place(self, key: str) -> Result[VersionedDocument[Any] | None, KernelError]:
        """Migrate the stored value and write it back, snapshotting the original first."""
        current = await self.load(key)
        if not current.ok:
            return err(current.error)
        if current.value is None:
            return ok(None)

        snapshot = await self.backup(key)
        if not snapshot.ok:
            return err(snapshot.error)

        document = current.value
        written = await attempt_async(
            lambda: self._adapter.put(key, document.to_dict()),
            "STORAGE_FAILED",
            f'Failed to write migrated "{key}".',
        )
        return ok(document) if written.ok else err(written.error)

    async def delete(self, key: str) -> Result[None, KernelError]:
        removed = await attempt_async(
            lambda: self._adapter.delete(key), "STORAGE_FAILED", f'Failed to delete "{key}".'
        )
        return ok(None) if removed.ok else err(removed.error)

    async def keys(self, prefix: str = "") -> list[str]:
        return [key for key in await self._adapter.keys(prefix) if BACKUP_MARKER not in key]

    async def backup(self, key: str) -> Result[BackupEntry | None, KernelError]:
        existing = await attempt_async(
            lambda: self._adapter.get(key),
            "STORAGE_FAILED",
            f'Failed to read "{key}" for backup.',
        )
        if not existing.ok:
            return err(existing.error)
        if existing.value is None:
            return ok(None)  # nothing to snapshot yet

        created_at = _iso(self._now())
        # The counter disambiguates snapshots taken inside the same millisecond, which is entirely
        # possible during a batch migration and would otherwise silently overwrite the previous one.
        suffix = format(self._backup_counter, "x")
        self._backup_counter += 1
        backup_id = f"{created_at}-{suffix}"
        backup_key = f"{key}{BACKUP_MARKER}{backup_id}"

        written = await attempt_async(
            lambda: self._adapter.put(backup_key, existing.value),
            "STORAGE_FAILED",
            f'Failed to write backup for "{key}".',
        )
        if not written.ok:
            return err(written.error)

        await self._prune(key)
        self._telemetry.counter("persistence.backup", 1)
        return ok(BackupEntry(id=backup_id, key=key, created_at=created_at))

    async def list_backups(self, key: str) -> list[BackupEntry]:
        prefix = f"{key}{BACKUP_MARKER}"
        keys = await self._adapter.keys(prefix)
        entries = []
        for backup_key in keys:
            backup_id = backup_key[len(prefix) :]
            entries.append(
                BackupEntry(id=backup_id, key=key, created_at=backup_id[: backup_id.rfind("-")])
            )
        return sorted(entries, key=lambda entry: entry.id)

    async def rollback(
        self, key: str, backup_id: str
    ) -> Result[VersionedDocument[Any] | None, KernelError]:
        backup_key = f"{key}{BACKUP_MARKER}{backup_id}"
        snapshot = await attempt_async(
            lambda: self._adapter.get(backup_key),
            "STORAGE_FAILED",
            f'Failed to read backup "{backup_id}".',
        )
        if not snapshot.ok:
            return err(snapshot.error)
        if snapshot.value is None:
            return err(
                KernelError(
                    "STORAGE_FAILED",
                    f'No backup "{backup_id}" for "{key}".',
                    {"key": key, "backupId": backup_id},
                )
            )
        # Snapshot what we are about to replace, so a rollback is itself reversible.
        safety = await self.backup(key)
        if not safety.ok:
            return err(safety.error)

        value = snapshot.value
        written = await attempt_async(
            lambda: self._adapter.put(key, value), "STORAGE_FAILED", f'Failed to restore "{key}".'
        )
        if not written.ok:
            return err(written.error)
        self._telemetry.counter("persistence.rollback", 1)
        return ok(_as_document(value) if is_versioned_document(value) else None)

    def namespaced(self, prefix: str) -> "NamespacedPersistence":
        """A view whose keys are transparently prefixed. Sandboxes each plugin's stored data."""
        return NamespacedPersistence(self, prefix)

    # -- internals ---------------------------------------------------------------------------

    def _apply_migration(
        self, document: VersionedDocument[Any], key: str
    ) -> Result[VersionedDocument[Any], KernelError]:
        if self._migrator is None:
            return ok(document)
        latest = self._migrator.latest_version(document.schema)
        if latest is None:
            return ok(document)  # schema owned by nobody currently loaded
        if document.version == latest:
            return ok(document)

        if document.version > latest:
            # Written by a newer release. Guessing at a downgrade risks destroying fields this
            # build does not know about, so refuse and let the host tell the user to upgrade.
            return err(
                KernelError(
                    "SCHEMA_VERSION_UNSUPPORTED",
                    f'"{key}" is at schema {document.schema} v{document.version}, '
                    f"newer than the supported v{latest}.",
                    {
                        "key": key,
                        "schema": document.schema,
                        "found": document.version,
                        "supported": latest,
                    },
                )
            )

        migrated = self._migrator.migrate(document)
        if not migrated.ok:
            self._telemetry.error(migrated.error, {"key": key})
            return migrated
        self._telemetry.counter(
            "persistence.migrate",
            1,
            {"schema": document.schema, "from": document.version, "to": migrated.value.version},
        )
        return migrated

    async def _prune(self, key: str) -> None:
        if self._max_backups == 0:
            return
        backups = await self.list_backups(key)
        excess = len(backups) - self._max_backups
        if excess <= 0:
            return
        # `backups[:excess]` with a negative `excess` is not an empty slice -- it drops entries from
        # the *end*, so a store holding fewer backups than the limit would prune the newest ones.
        # The TypeScript original is a counting loop, where a negative bound is simply a no-op.
        for entry in backups[:excess]:
            try:
                await self._adapter.delete(f"{key}{BACKUP_MARKER}{entry.id}")
            except Exception:  # noqa: BLE001
                pass  # pruning is housekeeping -- dropping an old snapshot must not fail the save


class NamespacedPersistence:
    """Key-prefixed facade over a ``PersistenceEngine``."""

    __slots__ = ("_engine", "_prefix")

    def __init__(self, engine: PersistenceEngine, prefix: str) -> None:
        self._engine = engine
        self._prefix = prefix if prefix.endswith(":") else f"{prefix}:"

    def _scoped(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def save(self, key: str, schema: str, data: Any, **options: Any):
        return self._engine.save(self._scoped(key), schema, data, **options)

    def load(self, key: str):
        return self._engine.load(self._scoped(key))

    def delete(self, key: str):
        return self._engine.delete(self._scoped(key))

    def backup(self, key: str):
        return self._engine.backup(self._scoped(key))

    def list_backups(self, key: str):
        return self._engine.list_backups(self._scoped(key))

    def rollback(self, key: str, backup_id: str):
        return self._engine.rollback(self._scoped(key), backup_id)

    async def keys(self) -> list[str]:
        keys = await self._engine.keys(self._prefix)
        return [key[len(self._prefix) :] for key in keys]
