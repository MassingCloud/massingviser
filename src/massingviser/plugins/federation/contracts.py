"""``massingviser.plugins.federation`` -- composing a project out of several models.

The property that matters is **id-preserving revision replacement**. When a consultant issues C03 to
supersede C02, everything anchored to that model -- pins, clashes, 4D links, takeoff -- must still
point at it. Replacing a model by removing and re-adding gives it a new id and silently orphans all
of them, which is why replacement is its own operation rather than a remove followed by an add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import Id, IsoTimestamp, ModelRecord, ProjectRecord, SessionStateRecord

LoadPhase = Literal["unloaded", "loading", "loaded", "failed"]


@dataclass(frozen=True)
class ModelLoadState:
    model_id: Id
    phase: LoadPhase = "unloaded"
    visible: bool = True
    #: Why it failed. Kept on the state rather than raised, so one bad model does not stop a
    #: federated project opening -- the other eleven still load and this one is reported.
    error: str | None = None
    loaded_at: IsoTimestamp | None = None


@runtime_checkable
class ModelLoaderPort(Protocol):
    """Whatever actually gets bytes into a viewer.

    Federation owns the *bookkeeping* -- what belongs to the project, what is loaded, what is
    visible -- and never the loading itself.
    """

    async def load(self, record: ModelRecord) -> Result[None, KernelError]: ...
    async def unload(self, model_id: Id) -> Result[None, KernelError]: ...
    async def set_transform(
        self, model_id: Id, transform: Sequence[float]
    ) -> Result[None, KernelError]: ...


ModelLoaderPortToken: CapabilityToken[ModelLoaderPort] = create_capability_token(
    "federation.loader"
)


@runtime_checkable
class FederationService(Protocol):
    async def open_project(self, project: ProjectRecord) -> Result[None, KernelError]: ...
    async def close_project(self) -> Result[None, KernelError]: ...
    def current_project(self) -> ProjectRecord | None: ...

    async def add_model(self, record: ModelRecord) -> Result[None, KernelError]: ...
    async def remove_model(self, model_id: Id) -> Result[None, KernelError]: ...
    def models(self) -> tuple[ModelRecord, ...]: ...

    async def load(self, model_id: Id) -> Result[None, KernelError]: ...
    async def unload(self, model_id: Id) -> Result[None, KernelError]: ...
    async def load_defaults(self) -> Result[tuple[ModelLoadState, ...], KernelError]: ...
    def state(self, model_id: Id) -> ModelLoadState | None: ...
    def states(self) -> tuple[ModelLoadState, ...]: ...

    def set_visible(self, model_id: Id, visible: bool) -> None: ...
    async def set_transform(
        self, model_id: Id, transform: Sequence[float]
    ) -> Result[None, KernelError]: ...

    #: Supersede a model with a new revision, keeping its id so nothing anchored to it is orphaned.
    async def replace_revision(
        self, model_id: Id, record: ModelRecord
    ) -> Result[ModelRecord, KernelError]: ...


FederationToken: CapabilityToken[FederationService] = create_capability_token(
    "federation.service"
)


@runtime_checkable
class SessionStateService(Protocol):
    async def capture(self) -> Result[SessionStateRecord, KernelError]: ...
    async def restore(self, state: SessionStateRecord) -> Result[None, KernelError]: ...


SessionStateToken: CapabilityToken[SessionStateService] = create_capability_token(
    "federation.session"
)


class FEDERATION_COMMANDS:
    open_project = "federation.project.open"
    close_project = "federation.project.close"
    add_model = "federation.model.add"
    replace_revision = "federation.model.replace-revision"
    save_session = "federation.session.save"
    restore_session = "federation.session.restore"


class FEDERATION_PERMISSIONS:
    manage_models = "federation.model.manage"
    open_project = "federation.project.open"


class FEDERATION_EVENTS:
    project_opened = "federation.project.opened"
    project_closed = "federation.project.closed"
    model_added = "federation.model.added"
    model_removed = "federation.model.removed"
    model_state_changed = "federation.model.state-changed"
    revision_replaced = "federation.model.revision-replaced"
    session_captured = "federation.session.captured"
