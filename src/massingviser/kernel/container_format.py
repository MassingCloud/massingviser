"""Containers: a project as a single addressable package.

``PersistenceEngine`` stores individual versioned documents against a key/value adapter. That is
the right mechanism for settings and per-plugin state, and the wrong one for a *project* -- which
is one thing a user opens, saves and sends, holding models, records and binary payloads together.

Without this, "open a project package" has to be invented later by whichever plugin needs it first,
and every other plugin then reaches around it. A container is a mechanism, not a business
capability, so it belongs in the kernel by the same rule that keeps massing and markup out of it.

The unity property matters as much as the packaging: a container opens as a whole. Models and the
records that reference them cannot be opened separately, because there is no API that opens one
without the other. Formats plug in as adapters -- a native project package, ISO 21597, whatever
comes next -- and the kernel never learns what any of them contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol, Sequence, runtime_checkable

from .disposable import Disposable, to_disposable
from .errors import KernelError
from .events import EventBus
from .persistence import (
    DocumentMigrator,
    StorageAdapter,
    VersionedDocument,
    is_versioned_document,
)
from .result import Err, Ok, Result, attempt_async, err, ok
from .telemetry import NOOP_TELEMETRY, TelemetrySink

ContainerEntryKind = Literal["document", "blob"]


@dataclass(frozen=True)
class ContainerEntry:
    #: Path within the container, ``/``-separated, no leading slash.
    path: str
    kind: ContainerEntryKind
    #: Schema id, for documents. Enables migration without opening every entry.
    schema: str | None = None
    version: int | None = None
    media_type: str | None = None
    byte_length: int | None = None
    modified_at: str | None = None


@dataclass(frozen=True)
class ContainerManifest:
    container_id: str
    #: Identifies the adapter that owns this format, e.g. ``"massingviser.project"``.
    format_id: str
    format_version: int
    name: str
    created_at: str
    entries: tuple[ContainerEntry, ...] = ()
    modified_at: str | None = None
    #: Application and version that last wrote it. Invaluable when a file will not open.
    producer: str | None = None


@dataclass(frozen=True)
class ContainerSource:
    """Where a container is being opened from.

    Adapters interpret whichever fields they support.
    """

    uri: str | None = None
    name: str | None = None
    bytes: bytes | None = None
    #: Key/value store, for adapters backed by one rather than by a file.
    storage: StorageAdapter | None = None


@dataclass(frozen=True)
class ContainerCreateInit:
    container_id: str
    name: str
    producer: str | None = None
    storage: StorageAdapter | None = None


@dataclass(frozen=True)
class ContainerAdapterOptions:
    #: Applied by the adapter when reading documents.
    #:
    #: Passed in rather than assumed: a container may be years old, and migration is the difference
    #: between opening it and refusing it.
    migrator: DocumentMigrator | None = None
    now: Callable[[], datetime] | None = None


@runtime_checkable
class OpenContainer(Protocol):
    """A container that is currently open.

    Documents and blobs are deliberately separate operations. A model payload is megabytes of
    opaque bytes and a markup record is a small JSON object; treating them alike would force one of
    them through the wrong path -- parsing blobs, or base64-ing geometry into a document.
    """

    @property
    def manifest(self) -> ContainerManifest: ...
    @property
    def dirty(self) -> bool: ...
    def entries(self, kind: ContainerEntryKind | None = None) -> tuple[ContainerEntry, ...]: ...
    def has(self, path: str) -> bool: ...
    async def read_document(self, path: str) -> Result[Any, KernelError]: ...
    async def write_document(self, path: str, schema: str, data: Any) -> Result[Any, KernelError]: ...
    async def read_blob(self, path: str) -> Result[Any, KernelError]: ...
    async def write_blob(
        self, path: str, data: bytes, media_type: str | None = None
    ) -> Result[None, KernelError]: ...
    async def remove(self, path: str) -> Result[None, KernelError]: ...
    def dispose(self) -> None: ...


@runtime_checkable
class ContainerAdapter(Protocol):
    @property
    def format_id(self) -> str: ...
    @property
    def display_name(self) -> str: ...
    #: File extensions this adapter claims, without the dot.
    @property
    def extensions(self) -> tuple[str, ...]: ...
    def can_open(self, source: ContainerSource) -> bool: ...
    async def open(
        self, source: ContainerSource, options: ContainerAdapterOptions | None = None
    ) -> Result[OpenContainer, KernelError]: ...
    async def create(
        self, init: ContainerCreateInit, options: ContainerAdapterOptions | None = None
    ) -> Result[OpenContainer, KernelError]: ...
    async def save(
        self, container: OpenContainer, target: ContainerSource | None = None
    ) -> Result[ContainerManifest, KernelError]: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


class ContainerService:
    """Opens, holds and saves the active container.

    Exactly one container is active at a time. That is the enforcement point for project/model
    unity: there is no path that opens a model on its own, so the class of bug where a project and
    its model drift apart because app code opened them separately cannot be written.

    Note this does *not* legislate how many models a container holds. That is a product decision --
    a single-model authoring tool and a federated coordination project are both legitimate -- and
    baking either into the kernel would be exactly the feature-in-the-backbone mistake the
    architecture exists to prevent.
    """

    __slots__ = ("_adapters", "_events", "_telemetry", "_migrator", "_now", "_active", "_active_adapter")

    def __init__(
        self,
        *,
        events: EventBus | None = None,
        telemetry: TelemetrySink | None = None,
        migrator: DocumentMigrator | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._adapters: dict[str, ContainerAdapter] = {}
        self._events = events
        self._telemetry = telemetry or NOOP_TELEMETRY
        self._migrator = migrator
        self._now = now or _utc_now
        self._active: OpenContainer | None = None
        self._active_adapter: ContainerAdapter | None = None

    def register_adapter(self, adapter: ContainerAdapter) -> Disposable:
        if adapter.format_id in self._adapters:
            raise KernelError(
                "SERVICE_DUPLICATE",
                f'A container adapter for "{adapter.format_id}" is already registered.',
                {"formatId": adapter.format_id},
            )
        self._adapters[adapter.format_id] = adapter
        return to_disposable(lambda: self._adapters.pop(adapter.format_id, None))

    def adapters(self) -> tuple[ContainerAdapter, ...]:
        return tuple(self._adapters.values())

    @property
    def active(self) -> OpenContainer | None:
        return self._active

    def _options(self) -> ContainerAdapterOptions:
        return ContainerAdapterOptions(migrator=self._migrator, now=self._now)

    async def create(
        self, format_id: str, init: ContainerCreateInit
    ) -> Result[OpenContainer, KernelError]:
        adapter = self._adapters.get(format_id)
        if adapter is None:
            return err(self._unknown_format(format_id))

        closed = await self.close()
        if not closed.ok:
            return err(closed.error)

        created = await attempt_async(
            lambda: adapter.create(init, self._options()),
            "STORAGE_FAILED",
            f'Adapter "{format_id}" failed to create a container.',
        )
        if not created.ok:
            return err(created.error)
        if not created.value.ok:
            return created.value

        self._adopt(created.value.value, adapter, "created")
        return created.value

    async def open(
        self, source: ContainerSource, format_id: str | None = None
    ) -> Result[OpenContainer, KernelError]:
        """Open a container, selecting an adapter by ``format_id`` or by asking each one."""
        adapter: ContainerAdapter | None
        if format_id:
            adapter = self._adapters.get(format_id)
        else:
            adapter = None
            for candidate in self.adapters():
                try:
                    if candidate.can_open(source):
                        adapter = candidate
                        break
                except Exception:  # noqa: BLE001
                    continue  # an adapter that raises while sniffing is simply not a match

        if adapter is None:
            return err(
                self._unknown_format(format_id)
                if format_id
                else KernelError(
                    "STORAGE_FAILED",
                    "No container adapter recognised this source.",
                    {"uri": source.uri},
                )
            )

        closed = await self.close()
        if not closed.ok:
            return err(closed.error)

        opened = await attempt_async(
            lambda: adapter.open(source, self._options()),
            "STORAGE_FAILED",
            f'Adapter "{adapter.format_id}" failed to open the container.',
        )
        if not opened.ok:
            return err(opened.error)
        if not opened.value.ok:
            return opened.value

        self._adopt(opened.value.value, adapter, "opened")
        return opened.value

    async def save(
        self, target: ContainerSource | None = None
    ) -> Result[ContainerManifest, KernelError]:
        if self._active is None or self._active_adapter is None:
            return err(self._no_active())

        adapter = self._active_adapter
        container = self._active
        saved = await attempt_async(
            lambda: adapter.save(container, target),
            "STORAGE_FAILED",
            "Failed to save the container.",
        )
        if not saved.ok:
            return err(saved.error)
        if not saved.value.ok:
            return saved.value

        self._telemetry.counter("container.saved", 1, {"formatId": adapter.format_id})
        if self._events is not None:
            self._events.emit("container.saved", {"manifest": saved.value.value})
        return saved.value

    async def close(self, *, force: bool = False) -> Result[None, KernelError]:
        """Close the active container.

        Refuses when there are unsaved changes unless ``force`` is set. Silently discarding a
        user's work to satisfy an "open" call is not a trade the kernel gets to make on their
        behalf.
        """
        container = self._active
        if container is None:
            return ok(None)

        if container.dirty and not force:
            return err(
                KernelError(
                    "STORAGE_FAILED",
                    "The container has unsaved changes.",
                    {"containerId": container.manifest.container_id},
                )
            )

        manifest = container.manifest
        self._active = None
        self._active_adapter = None
        try:
            container.dispose()
        except Exception:  # noqa: BLE001
            pass  # teardown failure must not leave the service holding a half-closed container
        if self._events is not None:
            self._events.emit("container.closed", {"manifest": manifest})
        return ok(None)

    def _adopt(
        self, container: OpenContainer, adapter: ContainerAdapter, reason: str
    ) -> None:
        self._active = container
        self._active_adapter = adapter
        self._telemetry.counter(f"container.{reason}", 1, {"formatId": adapter.format_id})
        # One event carrying the whole manifest: consumers learn about the project and everything
        # in it at the same instant, which is what makes partial-open states unrepresentable.
        if self._events is not None:
            self._events.emit(f"container.{reason}", {"manifest": container.manifest})

    def _unknown_format(self, format_id: str | None) -> KernelError:
        return KernelError(
            "SERVICE_NOT_FOUND",
            f'No container adapter for "{format_id}".',
            {"formatId": format_id, "available": list(self._adapters)},
        )

    def _no_active(self) -> KernelError:
        return KernelError("STORAGE_FAILED", "No container is open.", {})


# ---------------------------------------------------------------------------------------------
# Built-in adapter
# ---------------------------------------------------------------------------------------------

NATIVE_FORMAT_ID = "massingviser.project"
NATIVE_FORMAT_VERSION = 1


@dataclass
class _Entry:
    meta: ContainerEntry
    document: VersionedDocument[Any] | None = None
    blob: bytes | None = None


class InMemoryContainer:
    __slots__ = ("_manifest", "_entries", "_migrator", "_now", "_dirty", "_disposed")

    def __init__(
        self, manifest: ContainerManifest, options: ContainerAdapterOptions | None = None
    ) -> None:
        options = options or ContainerAdapterOptions()
        self._manifest = manifest
        self._entries: dict[str, _Entry] = {}
        self._migrator = options.migrator
        self._now = options.now or _utc_now
        self._dirty = False
        self._disposed = False
        for entry in manifest.entries:
            self._entries[entry.path] = _Entry(meta=entry)

    @property
    def manifest(self) -> ContainerManifest:
        return replace(self._manifest, entries=tuple(e.meta for e in self._entries.values()))

    @property
    def dirty(self) -> bool:
        return self._dirty

    def entries(self, kind: ContainerEntryKind | None = None) -> tuple[ContainerEntry, ...]:
        all_entries = tuple(entry.meta for entry in self._entries.values())
        if kind is None:
            return all_entries
        return tuple(entry for entry in all_entries if entry.kind == kind)

    def has(self, path: str) -> bool:
        return path in self._entries

    async def read_document(self, path: str) -> Result[Any, KernelError]:
        self._assert_live()
        entry = self._entries.get(path)
        if entry is None or entry.document is None:
            return ok(None)
        document = entry.document
        if self._migrator is None:
            return ok(document)

        latest = self._migrator.latest_version(document.schema)
        if latest is None or latest == document.version:
            return ok(document)
        if document.version > latest:
            return err(
                KernelError(
                    "SCHEMA_VERSION_UNSUPPORTED",
                    f'"{path}" is at {document.schema} v{document.version}, '
                    f"newer than the supported v{latest}.",
                    {"path": path, "found": document.version, "supported": latest},
                )
            )
        return self._migrator.migrate(document)

    async def write_document(self, path: str, schema: str, data: Any) -> Result[Any, KernelError]:
        self._assert_live()
        version = self._migrator.latest_version(schema) if self._migrator else None
        if version is None:
            version = 1
        document = VersionedDocument(
            schema=schema, version=version, saved_at=_iso(self._now()), data=data
        )
        self._entries[path] = _Entry(
            meta=ContainerEntry(
                path=path,
                kind="document",
                schema=schema,
                version=version,
                modified_at=document.saved_at,
            ),
            document=document,
        )
        self._dirty = True
        return ok(document)

    async def read_blob(self, path: str) -> Result[Any, KernelError]:
        self._assert_live()
        entry = self._entries.get(path)
        return ok(entry.blob if entry else None)

    async def write_blob(
        self, path: str, data: bytes, media_type: str | None = None
    ) -> Result[None, KernelError]:
        self._assert_live()
        self._entries[path] = _Entry(
            meta=ContainerEntry(
                path=path,
                kind="blob",
                byte_length=len(data),
                modified_at=_iso(self._now()),
                media_type=media_type,
            ),
            blob=bytes(data),
        )
        self._dirty = True
        return ok(None)

    async def remove(self, path: str) -> Result[None, KernelError]:
        self._assert_live()
        if self._entries.pop(path, None) is not None:
            self._dirty = True
        return ok(None)

    def mark_saved(self, modified_at: str) -> ContainerManifest:
        """Called by the adapter after a successful save."""
        self._manifest = replace(self._manifest, modified_at=modified_at)
        self._dirty = False
        return self.manifest

    def snapshot(self) -> tuple[_Entry, ...]:
        return tuple(self._entries.values())

    def dispose(self) -> None:
        self._disposed = True
        self._entries.clear()

    def _assert_live(self) -> None:
        if self._disposed:
            raise KernelError("CONTAINER_DISPOSED", "The container is closed.", {})


def _manifest_key(container_id: str) -> str:
    return f"container:{container_id}:manifest"


def _entry_key(container_id: str, path: str) -> str:
    return f"container:{container_id}:entry:{path}"


class StorageContainerAdapter:
    """Reference container adapter backed by a ``StorageAdapter``.

    Real enough to build and test against -- it round-trips documents and blobs through whatever
    key/value store the host supplies, including the in-memory one. A file-backed ``.mass`` adapter
    and an ISO 21597 adapter implement the same interface; nothing above ``ContainerAdapter``
    changes when they arrive.
    """

    format_id = NATIVE_FORMAT_ID
    display_name = "MassingViser project"
    # `.mass` is the current project-file extension; `.mmproj` is its predecessor and stays
    # readable so containers written before the rename still open. Order matters -- the first entry
    # is what a host offers when saving.
    extensions = ("mass", "mmproj")

    __slots__ = ("_storage",)

    def __init__(self, storage: StorageAdapter | None = None) -> None:
        self._storage = storage

    def can_open(self, source: ContainerSource) -> bool:
        if source.storage or self._storage:
            return source.uri is None or any(
                source.uri.endswith(f".{ext}") for ext in self.extensions
            )
        return False

    def _resolve(self, source: Any) -> StorageAdapter | None:
        return getattr(source, "storage", None) or self._storage

    async def create(
        self, init: ContainerCreateInit, options: ContainerAdapterOptions | None = None
    ) -> Result[OpenContainer, KernelError]:
        options = options or ContainerAdapterOptions()
        now = options.now or _utc_now
        manifest = ContainerManifest(
            container_id=init.container_id,
            format_id=self.format_id,
            format_version=NATIVE_FORMAT_VERSION,
            name=init.name,
            created_at=_iso(now()),
            entries=(),
            producer=init.producer,
        )
        return ok(InMemoryContainer(manifest, options))

    async def open(
        self, source: ContainerSource, options: ContainerAdapterOptions | None = None
    ) -> Result[OpenContainer, KernelError]:
        options = options or ContainerAdapterOptions()
        storage = self._resolve(source)
        if storage is None:
            return err(KernelError("STORAGE_FAILED", "No storage adapter supplied.", {}))

        container_id = source.name or source.uri
        if not container_id:
            return err(KernelError("STORAGE_FAILED", "Container source names no container.", {}))

        raw = await storage.get(_manifest_key(container_id))
        if raw is None:
            return err(
                KernelError(
                    "STORAGE_FAILED",
                    f'No container stored as "{container_id}".',
                    {"containerId": container_id},
                )
            )
        manifest = _manifest_from_mapping(raw)
        if manifest.format_version > NATIVE_FORMAT_VERSION:
            return err(
                KernelError(
                    "SCHEMA_VERSION_UNSUPPORTED",
                    f"Container format v{manifest.format_version} is newer than the "
                    f"supported v{NATIVE_FORMAT_VERSION}.",
                    {"containerId": container_id, "found": manifest.format_version},
                )
            )

        container = InMemoryContainer(manifest, options)
        for entry in manifest.entries:
            value = await storage.get(_entry_key(container_id, entry.path))
            if value is None:
                continue
            if entry.kind == "document" and is_versioned_document(value):
                document = (
                    value
                    if isinstance(value, VersionedDocument)
                    else VersionedDocument.from_mapping(value)
                )
                await container.write_document(entry.path, document.schema, document.data)
            elif entry.kind == "blob":
                await container.write_blob(entry.path, value, entry.media_type)
        # Loading is not editing: a freshly-opened container must not look unsaved.
        container.mark_saved(manifest.modified_at or manifest.created_at)
        return ok(container)

    async def save(
        self, container: OpenContainer, target: ContainerSource | None = None
    ) -> Result[ContainerManifest, KernelError]:
        storage = self._resolve(target) if target is not None else self._storage
        if storage is None:
            return err(KernelError("STORAGE_FAILED", "No storage adapter supplied.", {}))
        if not isinstance(container, InMemoryContainer):
            return err(
                KernelError("STORAGE_FAILED", "Container was not produced by this adapter.", {})
            )

        container_id = (target.name if target else None) or container.manifest.container_id
        saved_at = _iso(_utc_now())

        for entry in container.snapshot():
            value: Any = entry.document if entry.meta.kind == "document" else entry.blob
            if value is None:
                continue
            if isinstance(value, VersionedDocument):
                value = value.to_dict()
            await storage.put(_entry_key(container_id, entry.meta.path), value)

        manifest = container.mark_saved(saved_at)
        manifest = replace(manifest, container_id=container_id)
        await storage.put(_manifest_key(container_id), _manifest_to_mapping(manifest))
        return ok(manifest)


def _manifest_to_mapping(manifest: ContainerManifest) -> dict[str, Any]:
    return {
        "containerId": manifest.container_id,
        "formatId": manifest.format_id,
        "formatVersion": manifest.format_version,
        "name": manifest.name,
        "createdAt": manifest.created_at,
        "modifiedAt": manifest.modified_at,
        "producer": manifest.producer,
        "entries": [
            {
                "path": entry.path,
                "kind": entry.kind,
                "schema": entry.schema,
                "version": entry.version,
                "mediaType": entry.media_type,
                "byteLength": entry.byte_length,
                "modifiedAt": entry.modified_at,
            }
            for entry in manifest.entries
        ],
    }


def _manifest_from_mapping(value: Any) -> ContainerManifest:
    if isinstance(value, ContainerManifest):
        return value
    return ContainerManifest(
        container_id=value["containerId"],
        format_id=value["formatId"],
        format_version=value["formatVersion"],
        name=value["name"],
        created_at=value["createdAt"],
        modified_at=value.get("modifiedAt"),
        producer=value.get("producer"),
        entries=tuple(
            ContainerEntry(
                path=entry["path"],
                kind=entry["kind"],
                schema=entry.get("schema"),
                version=entry.get("version"),
                media_type=entry.get("mediaType"),
                byte_length=entry.get("byteLength"),
                modified_at=entry.get("modifiedAt"),
            )
            for entry in value.get("entries", ())
        ),
    )
