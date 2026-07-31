"""``massingviser.plugins.authoring`` -- edit sessions, reversible history, conflict-checked publish.

An edit session is a *transaction against a model somebody else may also be editing*. The
interesting operation is therefore not applying an edit -- it is publishing one, because that is
where two people's work meets. Publishing checks whether anything it touches has moved since the
session opened, and refuses rather than overwriting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import ElementRef, Id, IsoTimestamp

EditKind = Literal["create", "modify", "delete", "move"]


@dataclass(frozen=True)
class EditOperation:
    kind: EditKind
    #: Absent for ``create`` -- there is nothing to point at yet.
    element: ElementRef | None = None
    ifc_class: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    transform: tuple[float, ...] | None = None
    level_id: Id | None = None


@dataclass(frozen=True)
class SketchPlane:
    """The plane an edit is drawn on.

    Stored as origin plus normal rather than as a level id, because a sketch plane is often
    *offset* from a level -- a soffit, a working plane 900 mm up -- and collapsing the two loses
    that offset the moment the level moves.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    level_id: Id | None = None
    offset: float = 0.0


@dataclass(frozen=True)
class ConstraintRecord:
    id: Id
    kind: Literal["distance", "alignment", "level", "custom"]
    elements: tuple[ElementRef, ...] = ()
    value: float | None = None
    tolerance: float = 1e-6
    description: str | None = None


@dataclass(frozen=True)
class AuthoringSession:
    id: Id
    model_id: Id
    #: Model version the session opened against. The other half of every conflict check.
    base_version: str
    opened_at: IsoTimestamp
    opened_by: Id
    closed_at: IsoTimestamp | None = None
    published: bool = False


@runtime_checkable
class GeometryBackend(Protocol):
    """Whatever actually holds and mutates geometry.

    Authoring owns sessions, history, constraints and the publish gate. It never owns a solid
    modeller, which is why the same plugin runs against a viewer, a headless kernel, or a fake.
    """

    async def apply(
        self, operations: Sequence[EditOperation]
    ) -> Result[Sequence[ElementRef], KernelError]: ...
    async def revert(self, operations: Sequence[EditOperation]) -> Result[None, KernelError]: ...
    def current_version(self, model_id: Id) -> str | None: ...
    #: Has this element moved since that version -- i.e. did somebody else touch it?
    def changed_since(self, element: ElementRef, since_version: str) -> bool: ...
    async def publish(self, model_id: Id, version: str) -> Result[None, KernelError]: ...
    def evaluate_constraint(self, constraint: ConstraintRecord) -> bool: ...


GeometryBackendToken: CapabilityToken[GeometryBackend] = create_capability_token(
    "authoring.geometry"
)


@dataclass(frozen=True)
class Level:
    id: Id
    name: str
    elevation: float


@runtime_checkable
class LevelSource(Protocol):
    def levels(self) -> Sequence[Level]: ...


LevelSourceToken: CapabilityToken[LevelSource] = create_capability_token("authoring.levels")


@runtime_checkable
class AuthoringSessionService(Protocol):
    async def open(self, model_id: Id) -> Result[AuthoringSession, KernelError]: ...
    def current(self) -> AuthoringSession | None: ...
    #: Throws the session's edits away. Reverted through the backend, not just forgotten.
    async def discard(self, session_id: Id) -> Result[None, KernelError]: ...
    async def close(self, session_id: Id) -> Result[AuthoringSession, KernelError]: ...
    def set_sketch_plane(self, plane: SketchPlane) -> None: ...
    def sketch_plane(self) -> SketchPlane: ...


AuthoringSessionToken: CapabilityToken[AuthoringSessionService] = create_capability_token(
    "authoring.session"
)


@runtime_checkable
class EditCommandService(Protocol):
    async def apply(
        self, operations: Sequence[EditOperation]
    ) -> Result[tuple[ElementRef, ...], KernelError]: ...
    def can_apply(self, operations: Sequence[EditOperation]) -> Result[None, KernelError]: ...
    def add_constraint(self, constraint: ConstraintRecord) -> None: ...
    def constraints(self) -> tuple[ConstraintRecord, ...]: ...


EditCommandToken: CapabilityToken[EditCommandService] = create_capability_token("authoring.edit")


@dataclass(frozen=True)
class EditHistoryEntry:
    id: Id
    session_id: Id
    label: str
    operations: tuple[EditOperation, ...]
    elements: tuple[ElementRef, ...]
    applied_at: IsoTimestamp
    reverted: bool = False


@runtime_checkable
class EditHistoryService(Protocol):
    def entries(self) -> tuple[EditHistoryEntry, ...]: ...
    async def undo(self) -> Result[EditHistoryEntry, KernelError]: ...
    async def redo(self) -> Result[EditHistoryEntry, KernelError]: ...
    def can_undo(self) -> bool: ...
    def can_redo(self) -> bool: ...
    #: Collapses several entries into one, so a drag that produced forty edits undoes once.
    async def coalesce(
        self, label: str, entry_ids: Sequence[Id]
    ) -> Result[EditHistoryEntry, KernelError]: ...


EditHistoryToken: CapabilityToken[EditHistoryService] = create_capability_token("authoring.history")


@dataclass(frozen=True)
class PublishPreview:
    changed: tuple[ElementRef, ...]
    #: Elements someone else moved since this session opened.
    conflicts: tuple[ElementRef, ...]


@dataclass(frozen=True)
class PublishResult:
    session_id: Id
    model_id: Id
    version: str
    published: int


@runtime_checkable
class PublishService(Protocol):
    async def preview(self, session_id: Id) -> Result[PublishPreview, KernelError]: ...
    async def publish(
        self, session_id: Id, *, version: str, force: bool = False
    ) -> Result[PublishResult, KernelError]: ...


PublishToken: CapabilityToken[PublishService] = create_capability_token("authoring.publish")


class AUTHORING_COMMANDS:
    open_session = "authoring.session.open"
    discard_session = "authoring.session.discard"
    apply_edit = "authoring.edit.apply"
    publish = "authoring.publish"
    undo = "authoring.undo"
    redo = "authoring.redo"
    set_sketch_plane = "authoring.sketch-plane.activate"


class AUTHORING_PERMISSIONS:
    edit = "authoring.edit"
    publish = "authoring.publish"


class AUTHORING_EVENTS:
    session_opened = "authoring.session.opened"
    session_closed = "authoring.session.closed"
    edit_applied = "authoring.edit.applied"
    history_changed = "authoring.history.changed"
    published = "authoring.published"
