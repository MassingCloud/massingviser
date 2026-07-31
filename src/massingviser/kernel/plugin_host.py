from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Generic, Literal, Protocol, TypeVar

from .capabilities import CapabilityProvider, CapabilityRegistry, CapabilityToken
from .commands import CommandBus, CommandDefinition, CommandInfo
from .container import ServiceContainer
from .disposable import Disposable, DisposableStore
from .errors import KernelError
from .events import EmitReport, EventBus, EventHandler
from .logging import Logger, create_logger
from .permissions import PermissionService
from .persistence import NamespacedPersistence, PersistenceEngine
from .result import Result, attempt_async, err, ok
from .semver import satisfies
from .state import Slice, StateStore
from .telemetry import TelemetrySink
from .ui_registry import UIContribution, UIExtensionRegistry

THost = TypeVar("THost")
T = TypeVar("T")


@dataclass(frozen=True)
class PluginDependency:
    id: str
    #: Semver range the dependency must satisfy. Defaults to any version.
    version: str | None = None
    #: Missing optional dependencies are skipped rather than failing activation.
    optional: bool = False


@dataclass(frozen=True)
class PluginManifest:
    id: str
    version: str
    #: Kernel API range this plugin was built against, e.g. ``"^1.0.0"``.
    #:
    #: Checked before activation so an out-of-date plugin is refused with a clear message instead
    #: of failing later at an arbitrary call site with an ``AttributeError``.
    api_version: str
    name: str | None = None
    description: str | None = None
    dependencies: tuple[PluginDependency, ...] = ()
    #: Permission actions the plugin intends to use. Advisory -- surfaced for review and governance.
    permissions: tuple[str, ...] = ()


class Plugin(Protocol):
    manifest: PluginManifest

    def activate(self, context: PluginContext) -> Any: ...


PluginStatus = Literal["registered", "active", "inactive", "failed", "quarantined"]


@dataclass(frozen=True)
class PluginRecord:
    manifest: PluginManifest
    status: PluginStatus
    error: KernelError | None
    activated_at: str | None


@dataclass(frozen=True)
class ActivationReport:
    activated: tuple[str, ...] = ()
    failed: tuple[tuple[str, KernelError], ...] = ()
    #: Not attempted because a required dependency failed or was missing.
    skipped: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------------------------
# Context facades
#
# Every registration below is added to the plugin's DisposableStore, which is what deactivation
# drains. Returning the disposable as well lets a plugin release something early if it wants to.
# ---------------------------------------------------------------------------------------------


class PluginCommands:
    __slots__ = ("_bus", "_track")

    def __init__(self, bus: CommandBus, track: Callable[[Disposable], Disposable]) -> None:
        self._bus = bus
        self._track = track

    def register(self, definition: CommandDefinition) -> Disposable:
        return self._track(self._bus.register(definition))

    def execute(self, command_id: str, params: Any = None, **options: Any):
        return self._bus.execute(command_id, params, **options)

    def has(self, command_id: str) -> bool:
        return self._bus.has(command_id)

    def list(self) -> list[CommandInfo]:
        """Every registered command.

        Read-only enumeration, for plugins whose job *is* the command surface -- a palette, a
        keyboard binding editor, a diagnostics view. Without it such a plugin has to reach past the
        context to the kernel bus, which is exactly the coupling the context exists to prevent.
        """
        return self._bus.list()


class PluginEvents:
    __slots__ = ("_bus", "_track")

    def __init__(self, bus: EventBus, track: Callable[[Disposable], Disposable]) -> None:
        self._bus = bus
        self._track = track

    def on(self, type_: str, handler: EventHandler) -> Disposable:
        return self._track(self._bus.on(type_, handler))

    def once(self, type_: str, handler: EventHandler) -> Disposable:
        return self._track(self._bus.once(type_, handler))

    def emit(self, type_: str, payload: Any = None) -> EmitReport:
        return self._bus.emit(type_, payload)


class PluginState:
    __slots__ = ("_store", "_plugin_id")

    def __init__(self, store: StateStore, plugin_id: str) -> None:
        self._store = store
        self._plugin_id = plugin_id

    def define_slice(self, name: str, initial: T) -> Slice[T]:
        """Define a slice under this plugin's namespace. Released automatically on deactivate."""
        return self._store.define_slice(f"{self._plugin_id}/{name}", initial)

    def get_slice(self, namespace: str) -> Slice[Any] | None:
        """Read another namespace, for cross-plugin composition. Fully qualified name required."""
        return self._store.get_slice(namespace)


class PluginCapabilities:
    __slots__ = ("_registry", "_plugin_id", "_track")

    def __init__(
        self,
        registry: CapabilityRegistry,
        plugin_id: str,
        track: Callable[[Disposable], Disposable],
    ) -> None:
        self._registry = registry
        self._plugin_id = plugin_id
        self._track = track

    def provide(self, token: CapabilityToken[T], value: T, **options: Any) -> Disposable:
        options.setdefault("plugin_id", self._plugin_id)
        return self._track(self._registry.provide(token, value, **options))

    def get(self, token: CapabilityToken[T], **options: Any) -> T | None:
        return self._registry.get(token, **options)

    def require(self, token: CapabilityToken[T], **options: Any) -> Result[T, KernelError]:
        return self._registry.require(token, **options)

    def get_all(self, token: CapabilityToken[T]) -> tuple[CapabilityProvider[T], ...]:
        """Every provider of a capability, highest priority first.

        Some capabilities are inherently many-to-one -- validation rules, import adapters, metric
        providers -- where the consumer aggregates rather than chooses.
        """
        return self._registry.get_all(token)

    def has(self, token: CapabilityToken[Any]) -> bool:
        return self._registry.has(token)


class PluginUI:
    __slots__ = ("_registry", "_plugin_id", "_track")

    def __init__(
        self,
        registry: UIExtensionRegistry[Any],
        plugin_id: str,
        track: Callable[[Disposable], Disposable],
    ) -> None:
        self._registry = registry
        self._plugin_id = plugin_id
        self._track = track

    def register(self, contribution: UIContribution) -> Disposable:
        return self._track(
            self._registry.register(replace(contribution, plugin_id=self._plugin_id))
        )

    def by_point(self, point: str) -> tuple[UIContribution, ...]:
        """Every contribution at a point, in sort order.

        Read-only enumeration, for plugins whose job *is* the UI surface -- a shell tracking which
        panels exist, a layout editor, a diagnostics view. Without it such a plugin has to reach
        past the context to the kernel registry, which is exactly the coupling the context exists
        to prevent. Same reasoning as ``PluginCommands.list``.
        """
        return self._registry.by_point(point)

    def points(self) -> tuple[str, ...]:
        return self._registry.points()


@dataclass(frozen=True)
class PluginContext:
    """Everything a plugin is allowed to touch.

    The context is the *entire* API surface -- a plugin never imports the kernel's singletons
    directly. That is what makes deactivation reliable: every registration made through this object
    is tracked in ``subscriptions`` and released together, so a plugin cannot leave a command,
    panel or listener behind after it unloads.
    """

    plugin_id: str
    manifest: PluginManifest
    services: ServiceContainer
    commands: PluginCommands
    events: PluginEvents
    state: PluginState
    capabilities: PluginCapabilities
    ui: PluginUI
    storage: NamespacedPersistence
    permissions: PermissionService
    telemetry: TelemetrySink
    logger: Logger
    #: Add teardown for anything the plugin creates outside the tracked registries.
    subscriptions: DisposableStore


@dataclass
class _PluginEntry:
    plugin: Any
    status: PluginStatus
    error: KernelError | None = None
    activated_at: str | None = None
    store: DisposableStore | None = None
    scope: ServiceContainer | None = None


class PluginHost(Generic[THost]):
    """Registers, orders, activates and isolates plugins.

    The hard requirement is that no plugin can take down the kernel. Three things enforce it:
    activation runs inside ``attempt_async`` so a raise or a failed await becomes a ``Result``;
    every registration the plugin made is rolled back when it fails, so a half-activated plugin
    leaves no live commands or panels; and the failure is recorded as ``quarantined`` so an
    automatic retry cannot turn one broken plugin into an activation loop.
    """

    __slots__ = (
        "_container",
        "_commands",
        "_events",
        "_state",
        "_capabilities",
        "_ui",
        "_persistence",
        "_permissions",
        "_telemetry",
        "_api_version",
        "_now",
        "_entries",
    )

    def __init__(
        self,
        *,
        container: ServiceContainer,
        commands: CommandBus,
        events: EventBus,
        state: StateStore,
        capabilities: CapabilityRegistry,
        ui: UIExtensionRegistry[THost],
        persistence: PersistenceEngine,
        permissions: PermissionService,
        telemetry: TelemetrySink,
        api_version: str,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._container = container
        self._commands = commands
        self._events = events
        self._state = state
        self._capabilities = capabilities
        self._ui = ui
        self._persistence = persistence
        self._permissions = permissions
        self._telemetry = telemetry
        self._api_version = api_version
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._entries: dict[str, _PluginEntry] = {}

    def register(self, plugin: Any) -> Result[None, KernelError]:
        manifest: PluginManifest = plugin.manifest
        if manifest.id in self._entries:
            return err(
                KernelError(
                    "PLUGIN_DUPLICATE",
                    f'Plugin "{manifest.id}" is already registered.',
                    {"pluginId": manifest.id},
                )
            )
        if not satisfies(self._api_version, manifest.api_version):
            return err(
                KernelError(
                    "PLUGIN_API_INCOMPATIBLE",
                    f'Plugin "{manifest.id}" requires kernel API "{manifest.api_version}"; '
                    f"this kernel is {self._api_version}.",
                    {
                        "pluginId": manifest.id,
                        "required": manifest.api_version,
                        "actual": self._api_version,
                    },
                )
            )
        self._entries[manifest.id] = _PluginEntry(plugin=plugin, status="registered")
        return ok(None)

    def list(self) -> tuple[PluginRecord, ...]:
        return tuple(
            PluginRecord(
                manifest=entry.plugin.manifest,
                status=entry.status,
                error=entry.error,
                activated_at=entry.activated_at,
            )
            for entry in self._entries.values()
        )

    def status(self, plugin_id: str) -> PluginStatus | None:
        entry = self._entries.get(plugin_id)
        return entry.status if entry else None

    def is_active(self, plugin_id: str) -> bool:
        entry = self._entries.get(plugin_id)
        return entry is not None and entry.status == "active"

    def reset(self, plugin_id: str) -> Result[None, KernelError]:
        """Clear a quarantine so the plugin becomes eligible for activation again."""
        entry = self._entries.get(plugin_id)
        if entry is None:
            return err(self._not_found(plugin_id))
        if entry.status == "active":
            return ok(None)
        entry.status = "registered"
        entry.error = None
        return ok(None)

    async def activate(self, plugin_id: str) -> Result[None, KernelError]:
        entry = self._entries.get(plugin_id)
        if entry is None:
            return err(self._not_found(plugin_id))
        if entry.status == "active":
            return ok(None)
        if entry.status == "quarantined":
            return err(
                KernelError(
                    "PLUGIN_QUARANTINED",
                    f'Plugin "{plugin_id}" is quarantined after a failure.',
                    {
                        "pluginId": plugin_id,
                        "cause": entry.error.message if entry.error else None,
                    },
                )
            )

        dependencies = self._check_dependencies(entry.plugin.manifest)
        if not dependencies.ok:
            entry.status = "failed"
            entry.error = dependencies.error
            return err(dependencies.error)

        store = DisposableStore()
        scope = self._container.create_scope(f"plugin:{plugin_id}")
        store.add(scope)
        entry.store = store
        entry.scope = scope

        context = self._create_context(entry.plugin.manifest, scope, store)
        outcome = await attempt_async(
            lambda: entry.plugin.activate(context),
            "PLUGIN_ACTIVATION_FAILED",
            f'Plugin "{plugin_id}" failed to activate.',
        )

        if not outcome.ok:
            # Roll back everything the plugin managed to register before it failed. A
            # partially-activated plugin is worse than an absent one: its commands and panels are
            # live but its own invariants are not, so leaving them wired up guarantees a second,
            # more confusing failure later.
            store.dispose_collecting()
            self._remove_slices_of(plugin_id)
            entry.store = None
            entry.scope = None
            entry.status = "quarantined"
            entry.error = outcome.error
            self._telemetry.error(outcome.error, {"pluginId": plugin_id})
            self._events.emit("plugin.failed", {"pluginId": plugin_id, "error": outcome.error})
            return err(outcome.error)

        entry.status = "active"
        entry.error = None
        entry.activated_at = self._now().isoformat().replace("+00:00", "Z")
        self._telemetry.counter("plugin.activated", 1, {"pluginId": plugin_id})
        self._events.emit("plugin.activated", {"pluginId": plugin_id})
        return ok(None)

    async def deactivate(self, plugin_id: str) -> Result[None, KernelError]:
        entry = self._entries.get(plugin_id)
        if entry is None:
            return err(self._not_found(plugin_id))
        if entry.status != "active":
            return ok(None)

        dependents = self._active_dependents_of(plugin_id)
        if dependents:
            return err(
                KernelError(
                    "PLUGIN_DEPENDENCY_MISSING",
                    f'Cannot deactivate "{plugin_id}" while {", ".join(dependents)} depend on it.',
                    {"pluginId": plugin_id, "dependents": dependents},
                )
            )

        # The plugin's own teardown runs first, then the kernel drains its registrations. Order
        # matters: `deactivate` may want to flush state through services that the store is about to
        # dispose. A raise here is recorded but does not stop the drain -- otherwise a plugin could
        # make itself permanently unloadable.
        deactivate = getattr(entry.plugin, "deactivate", None)
        outcome = await attempt_async(
            lambda: deactivate() if callable(deactivate) else None,
            "PLUGIN_ACTIVATION_FAILED",
            f'Plugin "{plugin_id}" failed to deactivate cleanly.',
        )

        self._remove_slices_of(plugin_id)
        if entry.store is not None:
            entry.store.dispose_collecting()
        entry.store = None
        entry.scope = None
        entry.status = "inactive"
        entry.activated_at = None

        self._events.emit("plugin.deactivated", {"pluginId": plugin_id})
        if not outcome.ok:
            entry.error = outcome.error
            self._telemetry.error(outcome.error, {"pluginId": plugin_id})
            return err(outcome.error)
        return ok(None)

    async def activate_all(self) -> ActivationReport:
        """Activate every registered plugin in dependency order, isolating individual failures."""
        activated: list[str] = []
        failed: list[tuple[str, KernelError]] = []
        skipped: list[tuple[str, str]] = []

        order = self._topological_order()
        if not order.ok:
            return ActivationReport(
                activated=(),
                failed=(("*", order.error),),
                skipped=tuple((id_, order.error.message) for id_ in self._entries),
            )

        broken: set[str] = set()
        for plugin_id in order.value:
            entry = self._entries.get(plugin_id)
            if entry is None or entry.status == "active":
                continue

            blocking = [
                dependency.id
                for dependency in entry.plugin.manifest.dependencies
                if not dependency.optional and dependency.id in broken
            ]
            if blocking:
                broken.add(plugin_id)
                skipped.append((plugin_id, f"dependency unavailable: {', '.join(blocking)}"))
                continue

            result = await self.activate(plugin_id)
            if result.ok:
                activated.append(plugin_id)
            else:
                broken.add(plugin_id)
                failed.append((plugin_id, result.error))

        return ActivationReport(tuple(activated), tuple(failed), tuple(skipped))

    async def deactivate_all(self) -> None:
        """Deactivate every active plugin, dependents before their dependencies."""
        order = self._topological_order()
        ids = list(reversed(order.value)) if order.ok else list(self._entries)
        for plugin_id in ids:
            entry = self._entries.get(plugin_id)
            if entry is not None and entry.status == "active":
                await self.deactivate(plugin_id)

    # -- internals ---------------------------------------------------------------------------

    def _create_context(
        self, manifest: PluginManifest, scope: ServiceContainer, store: DisposableStore
    ) -> PluginContext:
        plugin_id = manifest.id

        def track(disposable: Disposable) -> Disposable:
            store.add(disposable)
            return disposable

        return PluginContext(
            plugin_id=plugin_id,
            manifest=manifest,
            services=scope,
            commands=PluginCommands(self._commands, track),
            events=PluginEvents(self._events, track),
            state=PluginState(self._state, plugin_id),
            capabilities=PluginCapabilities(self._capabilities, plugin_id, track),
            ui=PluginUI(self._ui, plugin_id, track),
            storage=self._persistence.namespaced(plugin_id),
            permissions=self._permissions,
            telemetry=self._telemetry,
            logger=create_logger(self._telemetry, plugin_id),
            subscriptions=store,
        )

    def _remove_slices_of(self, plugin_id: str) -> None:
        self._state.remove_slice(plugin_id)
        prefix = f"{plugin_id}/"
        for namespace in [n for n in self._state.snapshot() if n.startswith(prefix)]:
            self._state.remove_slice(namespace)

    def _check_dependencies(self, manifest: PluginManifest) -> Result[None, KernelError]:
        for dependency in manifest.dependencies:
            entry = self._entries.get(dependency.id)
            if entry is None:
                if dependency.optional:
                    continue
                return err(
                    KernelError(
                        "PLUGIN_DEPENDENCY_MISSING",
                        f'Plugin "{manifest.id}" requires "{dependency.id}", '
                        "which is not registered.",
                        {"pluginId": manifest.id, "dependency": dependency.id},
                    )
                )
            if dependency.version and not satisfies(
                entry.plugin.manifest.version, dependency.version
            ):
                return err(
                    KernelError(
                        "PLUGIN_DEPENDENCY_MISSING",
                        f'Plugin "{manifest.id}" requires "{dependency.id}@{dependency.version}"; '
                        f"found {entry.plugin.manifest.version}.",
                        {
                            "pluginId": manifest.id,
                            "dependency": dependency.id,
                            "required": dependency.version,
                        },
                    )
                )
            if not dependency.optional and entry.status != "active":
                return err(
                    KernelError(
                        "PLUGIN_DEPENDENCY_MISSING",
                        f'Plugin "{manifest.id}" requires "{dependency.id}" to be active '
                        f"(it is {entry.status}).",
                        {
                            "pluginId": manifest.id,
                            "dependency": dependency.id,
                            "status": entry.status,
                        },
                    )
                )
        return ok(None)

    def _active_dependents_of(self, plugin_id: str) -> list[str]:
        return [
            entry.plugin.manifest.id
            for entry in self._entries.values()
            if entry.status == "active"
            and any(
                dependency.id == plugin_id and not dependency.optional
                for dependency in entry.plugin.manifest.dependencies
            )
        ]

    def _topological_order(self) -> Result[list[str], KernelError]:
        """Depth-first topological sort.

        Reports the actual cycle rather than a generic failure -- with a dozen plugins loaded,
        "there is a cycle" is not a debuggable message.
        """
        sorted_ids: list[str] = []
        state: dict[str, str] = {}
        path: list[str] = []

        def visit(plugin_id: str) -> KernelError | None:
            current = state.get(plugin_id)
            if current == "done":
                return None
            if current == "visiting":
                cycle = " -> ".join([*path[path.index(plugin_id) :], plugin_id])
                return KernelError(
                    "PLUGIN_DEPENDENCY_CYCLE", f"Plugin dependency cycle: {cycle}", {"cycle": cycle}
                )
            entry = self._entries.get(plugin_id)
            if entry is None:
                return None  # unregistered dependency; reported by _check_dependencies

            state[plugin_id] = "visiting"
            path.append(plugin_id)
            for dependency in entry.plugin.manifest.dependencies:
                failure = visit(dependency.id)
                if failure is not None:
                    return failure
            path.pop()
            state[plugin_id] = "done"
            sorted_ids.append(plugin_id)
            return None

        for plugin_id in list(self._entries):
            failure = visit(plugin_id)
            if failure is not None:
                return err(failure)
        return ok(sorted_ids)

    def _not_found(self, plugin_id: str) -> KernelError:
        return KernelError(
            "PLUGIN_NOT_FOUND", f'No plugin registered as "{plugin_id}".', {"pluginId": plugin_id}
        )
