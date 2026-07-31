from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import ElementRef, Id, element_key
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import (
    AUTHORING_EVENTS,
    AuthoringSession,
    ConstraintRecord,
    EditHistoryEntry,
    EditOperation,
    GeometryBackendToken,
    LevelSourceToken,
    PublishPreview,
    PublishResult,
    SketchPlane,
)


@dataclass(frozen=True)
class AuthoringStores:
    sessions: RecordStore[AuthoringSession]
    history: RecordStore[EditHistoryEntry]
    constraints: RecordStore[ConstraintRecord]


@dataclass(frozen=True)
class AuthoringRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_authoring_stores(context: PluginContext) -> AuthoringStores:
    return AuthoringStores(
        sessions=create_record_store(context.state, "sessions"),
        history=create_record_store(context.state, "history"),
        constraints=create_record_store(context.state, "constraints"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


def _no_backend() -> KernelError:
    return KernelError(
        "CAPABILITY_NOT_FOUND",
        "No geometry backend is installed. Authoring owns sessions, history and the publish gate; "
        "a plugin must provide the modelling through GeometryBackendToken.",
        {},
    )


def resolve_sketch_plane(plane: SketchPlane, levels: Sequence[Any]) -> SketchPlane:
    """Resolve a level-hosted sketch plane to a world elevation.

    The offset is kept alongside the resolved origin rather than folded into it, so the plane still
    tracks its level when the level moves -- which is the entire reason a plane is hosted rather
    than placed.
    """
    if plane.level_id is None:
        return plane
    level = next((candidate for candidate in levels if candidate.id == plane.level_id), None)
    if level is None:
        return plane
    return replace(
        plane, origin=(plane.origin[0], plane.origin[1], level.elevation + plane.offset)
    )


class AuthoringSessionServiceImpl:
    __slots__ = ("_runtime", "_stores", "_active", "_plane")

    def __init__(self, runtime: AuthoringRuntime, stores: AuthoringStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._active: Id | None = None
        self._plane = SketchPlane()

    def current(self) -> AuthoringSession | None:
        return self._stores.sessions.get(self._active) if self._active else None

    async def open(self, model_id: Id) -> Result[AuthoringSession, KernelError]:
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())
        existing = self.current()
        if existing is not None and not existing.published:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Session "{existing.id}" is still open with unpublished edits. Publish or '
                    "discard it before opening another.",
                    {"sessionId": existing.id},
                )
            )

        version = backend.current_version(model_id)
        if version is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'The backend does not know model "{model_id}".',
                    {"modelId": model_id},
                )
            )

        session = AuthoringSession(
            id=self._runtime.ids.next("session"),
            model_id=model_id,
            # Captured now, and never updated. It is the baseline every conflict check compares
            # against; refreshing it would silently accept somebody else's concurrent edit.
            base_version=version,
            opened_at=self._runtime.clock.iso(),
            opened_by=self._runtime.context.permissions.identity.id,
        )
        self._stores.sessions.add(session)
        self._active = session.id
        self._runtime.context.events.emit(
            AUTHORING_EVENTS.session_opened, {"session": session}
        )
        return ok(session)

    async def discard(self, session_id: Id) -> Result[None, KernelError]:
        session = self._stores.sessions.get(session_id)
        if session is None:
            return err(_not_found("session", session_id))
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())

        # Reverted through the backend, newest first, rather than merely forgotten -- otherwise
        # "discard" leaves the geometry exactly as the abandoned session left it.
        entries = [
            entry
            for entry in self._stores.history.query(lambda e: e.session_id == session_id)
            if not entry.reverted
        ]
        for entry in reversed(entries):
            reverted = await backend.revert(entry.operations)
            if not reverted.ok:
                return err(reverted.error)
            self._stores.history.update(entry.id, {"reverted": True})

        self._stores.sessions.update(
            session_id, {"closed_at": self._runtime.clock.iso()}
        )
        if self._active == session_id:
            self._active = None
        self._runtime.context.events.emit(
            AUTHORING_EVENTS.session_closed, {"sessionId": session_id, "discarded": True}
        )
        return ok(None)

    async def close(self, session_id: Id) -> Result[AuthoringSession, KernelError]:
        session = self._stores.sessions.get(session_id)
        if session is None:
            return err(_not_found("session", session_id))
        if not session.published and self._stores.history.find(
            lambda e: e.session_id == session_id and not e.reverted
        ):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "This session has unpublished edits. Publish or discard them explicitly.",
                    {"sessionId": session_id},
                )
            )
        updated = self._stores.sessions.update(
            session_id, {"closed_at": self._runtime.clock.iso()}
        )
        if self._active == session_id:
            self._active = None
        return ok(updated) if updated else err(_not_found("session", session_id))

    def set_sketch_plane(self, plane: SketchPlane) -> None:
        levels = self._runtime.context.capabilities.get(LevelSourceToken)
        self._plane = resolve_sketch_plane(plane, levels.levels() if levels else ())

    def sketch_plane(self) -> SketchPlane:
        return self._plane


class EditCommandServiceImpl:
    __slots__ = ("_runtime", "_stores", "_sessions")

    def __init__(
        self,
        runtime: AuthoringRuntime,
        stores: AuthoringStores,
        sessions: AuthoringSessionServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._sessions = sessions

    def add_constraint(self, constraint: ConstraintRecord) -> None:
        self._stores.constraints.add(constraint)

    def constraints(self) -> tuple[ConstraintRecord, ...]:
        return self._stores.constraints.all()

    def can_apply(self, operations: Sequence[EditOperation]) -> Result[None, KernelError]:
        if not operations:
            return err(KernelError("COMMAND_FAILED", "An edit with no operations does nothing.", {}))
        if self._sessions.current() is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "No authoring session is open. Edits belong to a session so they can be "
                    "reverted, published or discarded as one.",
                    {},
                )
            )
        for operation in operations:
            if operation.kind != "create" and operation.element is None:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'A "{operation.kind}" operation must name the element it acts on.',
                        {"kind": operation.kind},
                    )
                )
        return ok(None)

    async def apply(
        self, operations: Sequence[EditOperation]
    ) -> Result[tuple[ElementRef, ...], KernelError]:
        allowed = self.can_apply(operations)
        if not allowed.ok:
            return err(allowed.error)
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())

        session = self._sessions.current()
        assert session is not None

        applied = await backend.apply(operations)
        if not applied.ok:
            return err(applied.error)
        elements = tuple(applied.value)

        # Constraints are checked *after* the edit lands, then the edit is rolled back if one
        # broke. Checking beforehand would need a predictive model of the backend, which is the
        # backend's job and not something authoring can second-guess.
        broken = [
            constraint
            for constraint in self._stores.constraints.all()
            if not backend.evaluate_constraint(constraint)
        ]
        if broken:
            await backend.revert(operations)
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "The edit was rolled back because it broke: "
                    + "; ".join(c.description or c.id for c in broken),
                    {"constraints": [c.id for c in broken]},
                )
            )

        entry = EditHistoryEntry(
            id=self._runtime.ids.next("edit"),
            session_id=session.id,
            label=f"{operations[0].kind} x{len(operations)}"
            if len(operations) > 1
            else operations[0].kind,
            operations=tuple(operations),
            elements=elements,
            applied_at=self._runtime.clock.iso(),
        )
        self._stores.history.add(entry)
        self._runtime.context.events.emit(AUTHORING_EVENTS.edit_applied, {"entry": entry})
        return ok(elements)


class EditHistoryServiceImpl:
    """Session-scoped, reversible edit history.

    Separate from the kernel's command history on purpose. The kernel undoes *commands*; this
    undoes *geometry*, against a backend that has to be told. Coalescing lives here too, because a
    drag that produced forty edits has to undo once.
    """

    __slots__ = ("_runtime", "_stores", "_sessions")

    def __init__(
        self,
        runtime: AuthoringRuntime,
        stores: AuthoringStores,
        sessions: AuthoringSessionServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._sessions = sessions

    def _session_entries(self) -> list[EditHistoryEntry]:
        session = self._sessions.current()
        if session is None:
            return []
        return list(self._stores.history.query(lambda e: e.session_id == session.id))

    def entries(self) -> tuple[EditHistoryEntry, ...]:
        return tuple(self._session_entries())

    def can_undo(self) -> bool:
        return any(not entry.reverted for entry in self._session_entries())

    def can_redo(self) -> bool:
        return any(entry.reverted for entry in self._session_entries())

    async def undo(self) -> Result[EditHistoryEntry, KernelError]:
        live = [entry for entry in self._session_entries() if not entry.reverted]
        if not live:
            return err(KernelError("COMMAND_NOT_FOUND", "Nothing to undo in this session.", {}))
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())

        entry = live[-1]
        reverted = await backend.revert(entry.operations)
        if not reverted.ok:
            return err(reverted.error)
        updated = self._stores.history.update(entry.id, {"reverted": True})
        self._runtime.context.events.emit(AUTHORING_EVENTS.history_changed, {"entry": updated})
        return ok(updated) if updated else err(_not_found("edit", entry.id))

    async def redo(self) -> Result[EditHistoryEntry, KernelError]:
        reverted_entries = [entry for entry in self._session_entries() if entry.reverted]
        if not reverted_entries:
            return err(KernelError("COMMAND_NOT_FOUND", "Nothing to redo in this session.", {}))
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())

        entry = reverted_entries[0]
        applied = await backend.apply(entry.operations)
        if not applied.ok:
            return err(applied.error)
        updated = self._stores.history.update(
            entry.id, {"reverted": False, "elements": tuple(applied.value)}
        )
        self._runtime.context.events.emit(AUTHORING_EVENTS.history_changed, {"entry": updated})
        return ok(updated) if updated else err(_not_found("edit", entry.id))

    async def coalesce(
        self, label: str, entry_ids: Sequence[Id]
    ) -> Result[EditHistoryEntry, KernelError]:
        wanted = [self._stores.history.get(entry_id) for entry_id in entry_ids]
        missing = [
            entry_id for entry_id, entry in zip(entry_ids, wanted) if entry is None
        ]
        if missing:
            return err(_not_found("edit", missing[0]))
        entries = [entry for entry in wanted if entry is not None]
        if len({entry.session_id for entry in entries}) != 1:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "Entries from different sessions cannot be coalesced -- they do not undo as "
                    "one act.",
                    {},
                )
            )

        merged = EditHistoryEntry(
            id=self._runtime.ids.next("edit"),
            session_id=entries[0].session_id,
            label=label,
            operations=tuple(op for entry in entries for op in entry.operations),
            elements=tuple(el for entry in entries for el in entry.elements),
            applied_at=entries[-1].applied_at,
        )
        for entry in entries:
            self._stores.history.remove(entry.id)
        self._stores.history.add(merged)
        return ok(merged)


class PublishServiceImpl:
    __slots__ = ("_runtime", "_stores", "_sessions")

    def __init__(
        self,
        runtime: AuthoringRuntime,
        stores: AuthoringStores,
        sessions: AuthoringSessionServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._sessions = sessions

    async def preview(self, session_id: Id) -> Result[PublishPreview, KernelError]:
        session = self._stores.sessions.get(session_id)
        if session is None:
            return err(_not_found("session", session_id))
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())

        touched: dict[str, ElementRef] = {}
        for entry in self._stores.history.query(
            lambda e: e.session_id == session_id and not e.reverted
        ):
            for element in entry.elements:
                touched[element_key(element)] = element

        # Anything somebody else moved since this session opened. `base_version` is the baseline,
        # which is why it is captured once and never refreshed.
        conflicts = tuple(
            element
            for element in touched.values()
            if backend.changed_since(element, session.base_version)
        )
        return ok(PublishPreview(changed=tuple(touched.values()), conflicts=conflicts))

    async def publish(
        self, session_id: Id, *, version: str, force: bool = False
    ) -> Result[PublishResult, KernelError]:
        """Commit a session's edits, refusing when someone else got there first.

        ``force`` exists because sometimes the other edit really is the stale one -- but it is a
        deliberate act with a name, not the default. Publishing over a concurrent change silently
        is how one person's afternoon disappears.
        """
        session = self._stores.sessions.get(session_id)
        if session is None:
            return err(_not_found("session", session_id))
        if session.published:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Session "{session_id}" has already been published.',
                    {"sessionId": session_id},
                )
            )
        backend = self._runtime.context.capabilities.get(GeometryBackendToken)
        if backend is None:
            return err(_no_backend())

        previewed = await self.preview(session_id)
        if not previewed.ok:
            return err(previewed.error)
        preview = previewed.value

        if not preview.changed:
            return err(
                KernelError(
                    "COMMAND_FAILED", "This session changed nothing.", {"sessionId": session_id}
                )
            )
        if preview.conflicts and not force:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"{len(preview.conflicts)} element(s) changed since this session opened: "
                    + ", ".join(e.global_id for e in preview.conflicts[:5])
                    + ". Re-open against the current version, or publish with force.",
                    {
                        "sessionId": session_id,
                        "conflicts": [e.global_id for e in preview.conflicts],
                    },
                )
            )

        published = await backend.publish(session.model_id, version)
        if not published.ok:
            return err(published.error)

        self._stores.sessions.update(
            session_id, {"published": True, "closed_at": self._runtime.clock.iso()}
        )
        result = PublishResult(
            session_id=session_id,
            model_id=session.model_id,
            version=version,
            published=len(preview.changed),
        )
        self._runtime.context.events.emit(AUTHORING_EVENTS.published, {"result": result})
        return ok(result)
