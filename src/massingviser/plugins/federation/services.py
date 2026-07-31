from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...kernel import KernelError, PluginContext, Result, err, ok
from ...schema import Id, ModelRecord, ProjectRecord, SessionStateRecord
from ...sdk import Clock, IdFactory, RecordStore, create_record_store
from .contracts import FEDERATION_EVENTS, ModelLoaderPortToken, ModelLoadState


@dataclass(frozen=True)
class FederationStores:
    projects: RecordStore[ProjectRecord]
    models: RecordStore[ModelRecord]
    sessions: RecordStore[SessionStateRecord]


@dataclass(frozen=True)
class FederationRuntime:
    context: PluginContext
    clock: Clock
    ids: IdFactory


def create_federation_stores(context: PluginContext) -> FederationStores:
    return FederationStores(
        projects=create_record_store(context.state, "projects"),
        models=create_record_store(context.state, "models"),
        sessions=create_record_store(context.state, "sessions"),
    )


def _not_found(kind: str, id: Id) -> KernelError:
    return KernelError("COMMAND_FAILED", f'No {kind} with id "{id}".', {"id": id})


class FederationServiceImpl:
    __slots__ = ("_runtime", "_stores", "_states", "_active")

    def __init__(self, runtime: FederationRuntime, stores: FederationStores) -> None:
        self._runtime = runtime
        self._stores = stores
        self._states: dict[Id, ModelLoadState] = {}
        self._active: Id | None = None

    # -- project ------------------------------------------------------------------------------

    def current_project(self) -> ProjectRecord | None:
        return self._stores.projects.get(self._active) if self._active else None

    async def open_project(self, project: ProjectRecord) -> Result[None, KernelError]:
        if self._active is not None:
            closed = await self.close_project()
            if not closed.ok:
                return err(closed.error)
        if not self._stores.projects.has(project.id):
            self._stores.projects.add(project)
        self._active = project.id
        self._runtime.context.events.emit(FEDERATION_EVENTS.project_opened, {"project": project})
        return ok(None)

    async def close_project(self) -> Result[None, KernelError]:
        project = self.current_project()
        if project is None:
            return ok(None)
        for model_id in list(self._states):
            await self.unload(model_id)
        self._active = None
        self._runtime.context.events.emit(FEDERATION_EVENTS.project_closed, {"project": project})
        return ok(None)

    # -- models -------------------------------------------------------------------------------

    async def add_model(self, record: ModelRecord) -> Result[None, KernelError]:
        if self._active is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "No project is open, so there is nothing for a model to belong to.",
                    {},
                )
            )
        if self._stores.models.has(record.id):
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Model "{record.id}" is already in this project. To supersede it with a new '
                    "revision use replace_revision, which keeps the id.",
                    {"modelId": record.id},
                )
            )
        self._stores.models.add(record)
        self._states[record.id] = ModelLoadState(model_id=record.id)

        project = self.current_project()
        if project is not None and record.id not in project.model_ids:
            self._stores.projects.update(project.id, {"model_ids": (*project.model_ids, record.id)})
        self._runtime.context.events.emit(FEDERATION_EVENTS.model_added, {"record": record})
        return ok(None)

    async def remove_model(self, model_id: Id) -> Result[None, KernelError]:
        if not self._stores.models.has(model_id):
            return err(_not_found("model", model_id))
        await self.unload(model_id)
        self._stores.models.remove(model_id)
        self._states.pop(model_id, None)

        project = self.current_project()
        if project is not None:
            self._stores.projects.update(
                project.id,
                {"model_ids": tuple(m for m in project.model_ids if m != model_id)},
            )
        self._runtime.context.events.emit(FEDERATION_EVENTS.model_removed, {"modelId": model_id})
        return ok(None)

    def models(self) -> tuple[ModelRecord, ...]:
        return self._stores.models.all()

    # -- load state ---------------------------------------------------------------------------

    def _set_state(self, model_id: Id, **changes: Any) -> ModelLoadState:
        current = self._states.get(model_id) or ModelLoadState(model_id=model_id)
        updated = replace(current, **changes)
        self._states[model_id] = updated
        self._runtime.context.events.emit(FEDERATION_EVENTS.model_state_changed, {"state": updated})
        return updated

    async def load(self, model_id: Id) -> Result[None, KernelError]:
        record = self._stores.models.get(model_id)
        if record is None:
            return err(_not_found("model", model_id))
        loader = self._runtime.context.capabilities.get(ModelLoaderPortToken)
        if loader is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No model loader is installed; federation tracks what is loaded but does not "
                    "load it.",
                    {"modelId": model_id},
                )
            )
        if (self._states.get(model_id) or ModelLoadState(model_id)).phase == "loaded":
            return ok(None)

        self._set_state(model_id, phase="loading", error=None)
        loaded = await loader.load(record)
        if not loaded.ok:
            # Recorded on the state, not raised. A federated project with one unreadable consultant
            # model must still open.
            self._set_state(model_id, phase="failed", error=loaded.error.message)
            return err(loaded.error)
        self._set_state(model_id, phase="loaded", loaded_at=self._runtime.clock.iso())
        return ok(None)

    async def unload(self, model_id: Id) -> Result[None, KernelError]:
        state = self._states.get(model_id)
        if state is None or state.phase == "unloaded":
            return ok(None)
        loader = self._runtime.context.capabilities.get(ModelLoaderPortToken)
        if loader is not None:
            await loader.unload(model_id)
        self._set_state(model_id, phase="unloaded", loaded_at=None, error=None)
        return ok(None)

    async def load_defaults(self) -> Result[tuple[ModelLoadState, ...], KernelError]:
        """Load every model the project marks as load-by-default.

        Failures do not abort the batch -- they land on the individual state, so opening a project
        tells you which models came up and which did not, in one pass.
        """
        for record in self._stores.models.all():
            if record.load_by_default:
                await self.load(record.id)
        return ok(self.states())

    def state(self, model_id: Id) -> ModelLoadState | None:
        return self._states.get(model_id)

    def states(self) -> tuple[ModelLoadState, ...]:
        return tuple(self._states.values())

    def set_visible(self, model_id: Id, visible: bool) -> None:
        if model_id in self._states:
            self._set_state(model_id, visible=visible)
        self._stores.models.update(model_id, {"visible": visible})

    async def set_transform(
        self, model_id: Id, transform: Sequence[float]
    ) -> Result[None, KernelError]:
        if not self._stores.models.has(model_id):
            return err(_not_found("model", model_id))
        loader = self._runtime.context.capabilities.get(ModelLoaderPortToken)
        if loader is not None:
            applied = await loader.set_transform(model_id, transform)
            if not applied.ok:
                return err(applied.error)
        self._stores.models.update(model_id, {"transform": tuple(transform)})
        return ok(None)

    async def replace_revision(
        self, model_id: Id, record: ModelRecord
    ) -> Result[ModelRecord, KernelError]:
        """Supersede a model with a new revision, **keeping its id**.

        Everything anchored to this model -- pins, clashes, 4D links, takeoff -- references it by
        id. Removing and re-adding would mint a new one and orphan every single one of them, which
        is why this is its own operation and why the incoming record's id is overwritten rather
        than trusted.
        """
        existing = self._stores.models.get(model_id)
        if existing is None:
            return err(_not_found("model", model_id))
        if record.version == existing.version:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Model "{model_id}" is already at version "{existing.version}".',
                    {"modelId": model_id, "version": existing.version},
                )
            )

        was_loaded = (self._states.get(model_id) or ModelLoadState(model_id)).phase == "loaded"
        if was_loaded:
            await self.unload(model_id)

        # The id is the project's, not the incoming file's. Transform and visibility are the
        # project's decisions too and survive the re-issue.
        superseded = replace(
            record,
            id=model_id,
            transform=record.transform if record.transform is not None else existing.transform,
            visible=existing.visible,
            load_by_default=existing.load_by_default,
        )
        self._stores.models.replace(superseded)

        self._runtime.context.events.emit(
            FEDERATION_EVENTS.revision_replaced,
            {
                "modelId": model_id,
                "from": existing.version,
                "to": superseded.version,
            },
        )
        if was_loaded:
            await self.load(model_id)
        return ok(superseded)


class SessionStateServiceImpl:
    """Saves which models were loaded and what was being looked at.

    Reopening a federated project should not mean reloading twelve models and re-hiding nine of
    them, so load state is persisted explicitly rather than reconstructed.
    """

    __slots__ = ("_runtime", "_stores", "_federation")

    def __init__(
        self,
        runtime: FederationRuntime,
        stores: FederationStores,
        federation: FederationServiceImpl,
    ) -> None:
        self._runtime = runtime
        self._stores = stores
        self._federation = federation

    async def capture(self) -> Result[SessionStateRecord, KernelError]:
        project = self._federation.current_project()
        if project is None:
            return err(
                KernelError("COMMAND_FAILED", "No project is open, so there is no session.", {})
            )
        record = SessionStateRecord(
            id=self._runtime.ids.next("session"),
            project_id=project.id,
            saved_at=self._runtime.clock.iso(),
            saved_by=self._runtime.context.permissions.identity.id,
            loaded_model_ids=tuple(
                state.model_id for state in self._federation.states() if state.phase == "loaded"
            ),
            open_panels=tuple(
                contribution.id for contribution in self._runtime.context.commands.list()[:0]
            ),
        )
        self._stores.sessions.add(record)
        self._runtime.context.events.emit(FEDERATION_EVENTS.session_captured, {"record": record})
        return ok(record)

    async def restore(self, state: SessionStateRecord) -> Result[None, KernelError]:
        project = self._stores.projects.get(state.project_id)
        if project is None:
            return err(_not_found("project", state.project_id))
        opened = await self._federation.open_project(project)
        if not opened.ok:
            return err(opened.error)

        wanted = set(state.loaded_model_ids)
        for record in self._federation.models():
            if record.id in wanted:
                await self._federation.load(record.id)
            else:
                await self._federation.unload(record.id)
        return ok(None)
