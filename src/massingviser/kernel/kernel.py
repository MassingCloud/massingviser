from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar

from .capabilities import CapabilityRegistry
from .commands import CommandBus
from .container import ServiceContainer, ServiceToken, create_service_token
from .container_format import ContainerService, StorageContainerAdapter
from .events import EventBus
from .errors import KernelError
from .logging import Logger, create_logger
from .permissions import ALLOW_ALL, Identity, PermissionEvaluator, PermissionService
from .persistence import DocumentMigrator, MemoryStorageAdapter, PersistenceEngine, StorageAdapter
from .plugin_host import ActivationReport, Plugin, PluginHost, PluginRecord
from .result import Result
from .state import StateStore
from .telemetry import NOOP_TELEMETRY, TelemetrySink
from .ui_registry import UIExtensionRegistry

THost = TypeVar("THost")

#: The kernel's own API version.
#:
#: Plugins declare a range against this. Bump the major only for a breaking change to
#: ``PluginContext`` or to a kernel service's contract -- the whole point of the number is that a
#: plugin built today keeps working, or fails loudly at load with an actionable message.
KERNEL_API_VERSION = "1.0.0"

EventBusToken: ServiceToken[EventBus] = create_service_token("kernel.events")
CommandBusToken: ServiceToken[CommandBus] = create_service_token("kernel.commands")
StateStoreToken: ServiceToken[StateStore] = create_service_token("kernel.state")
CapabilityRegistryToken: ServiceToken[CapabilityRegistry] = create_service_token("kernel.capabilities")
PersistenceToken: ServiceToken[PersistenceEngine] = create_service_token("kernel.persistence")
PermissionsToken: ServiceToken[PermissionService] = create_service_token("kernel.permissions")
TelemetryToken: ServiceToken[TelemetrySink] = create_service_token("kernel.telemetry")
LoggerToken: ServiceToken[Logger] = create_service_token("kernel.logger")
ContainerServiceToken: ServiceToken[ContainerService] = create_service_token("kernel.containers")


@dataclass(frozen=True)
class KernelDiagnostics:
    api_version: str
    plugins: tuple[PluginRecord, ...]
    capabilities: tuple[str, ...]
    commands: int
    ui_points: tuple[str, ...]
    history: dict[str, int]
    state_namespaces: tuple[str, ...]


class Kernel(Generic[THost]):
    """The assembled backbone.

    Everything here is a *mechanism*, never a feature: there is no viewer, no markup, no massing in
    this object or anywhere it imports. Business capability arrives exclusively through plugins, so
    the kernel can be versioned and kept stable while the platform above it changes freely.
    """

    __slots__ = (
        "api_version",
        "services",
        "events",
        "commands",
        "state",
        "capabilities",
        "ui",
        "persistence",
        "containers",
        "permissions",
        "telemetry",
        "plugins",
        "logger",
    )

    def __init__(
        self,
        *,
        storage: StorageAdapter | None = None,
        migrator: DocumentMigrator | None = None,
        telemetry: TelemetrySink | None = None,
        permission_evaluator: PermissionEvaluator | None = None,
        identity: Identity | None = None,
        history_limit: int = 100,
        max_backups: int = 5,
        api_version: str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.api_version = api_version or KERNEL_API_VERSION
        self.telemetry: TelemetrySink = telemetry or NOOP_TELEMETRY
        self.logger = create_logger(self.telemetry, "kernel")

        self.services = ServiceContainer("kernel")
        self.events = EventBus()
        self.state = StateStore()
        self.capabilities = CapabilityRegistry()
        self.ui: UIExtensionRegistry[THost] = UIExtensionRegistry()

        self.permissions = PermissionService()
        self.permissions.set_evaluator(permission_evaluator or ALLOW_ALL)
        if identity is not None:
            self.permissions.set_identity(identity)

        # One storage adapter serves both the per-document engine and the container service, so a
        # host that swaps in a filesystem or an object store gets both at once rather than half a
        # persisted app.
        adapter = storage or MemoryStorageAdapter()

        self.persistence = PersistenceEngine(
            adapter=adapter,
            migrator=migrator,
            telemetry=self.telemetry,
            max_backups=max_backups,
            now=now,
        )

        self.containers = ContainerService(
            events=self.events, telemetry=self.telemetry, migrator=migrator, now=now
        )
        # The reference adapter is registered by default so a host has a working container format
        # from the first line of code; a real `.mass` or ISO 21597 adapter simply registers
        # alongside it.
        self.containers.register_adapter(StorageContainerAdapter(adapter))

        self.commands = CommandBus(
            permissions=self.permissions,
            telemetry=self.telemetry,
            history_limit=history_limit,
            now=(lambda: now().timestamp() * 1000.0) if now else None,
        )

        self.plugins: PluginHost[THost] = PluginHost(
            container=self.services,
            commands=self.commands,
            events=self.events,
            state=self.state,
            capabilities=self.capabilities,
            ui=self.ui,
            persistence=self.persistence,
            permissions=self.permissions,
            telemetry=self.telemetry,
            api_version=self.api_version,
            now=now,
        )

        # Core services are also resolvable through the container, so a plugin's own services can
        # take them as constructor dependencies rather than closing over the context object.
        self.services.register_value(EventBusToken, self.events)
        self.services.register_value(CommandBusToken, self.commands)
        self.services.register_value(StateStoreToken, self.state)
        self.services.register_value(CapabilityRegistryToken, self.capabilities)
        self.services.register_value(PersistenceToken, self.persistence)
        self.services.register_value(ContainerServiceToken, self.containers)
        self.services.register_value(PermissionsToken, self.permissions)
        self.services.register_value(TelemetryToken, self.telemetry)
        self.services.register_value(LoggerToken, self.logger)

    def use(self, plugin: Plugin) -> Result[None, KernelError]:
        """Register a plugin. Does not activate it -- call ``start``, or ``plugins.activate(id)``."""
        return self.plugins.register(plugin)

    async def start(self) -> ActivationReport:
        """Activate every registered plugin in dependency order."""
        report = await self.plugins.activate_all()
        self.telemetry.event(
            "kernel.started",
            {
                "activated": len(report.activated),
                "failed": len(report.failed),
                "skipped": len(report.skipped),
            },
        )
        self.events.emit("kernel.started", report)
        return report

    async def stop(self) -> None:
        """Deactivate everything, dependents first. The kernel stays usable afterwards."""
        await self.plugins.deactivate_all()
        self.events.emit("kernel.stopped", {})

    def diagnostics(self) -> KernelDiagnostics:
        return KernelDiagnostics(
            api_version=self.api_version,
            plugins=self.plugins.list(),
            capabilities=self.capabilities.tokens(),
            commands=len(self.commands.list()),
            ui_points=self.ui.points(),
            history=self.commands.history_size,
            state_namespaces=tuple(self.state.snapshot()),
        )

    def dispose(self) -> None:
        # Synchronous teardown for host shutdown. Callers wanting orderly plugin `deactivate` hooks
        # should `await stop()` first; this is the last-resort path that guarantees release.
        self.events.clear()
        self.commands.clear_history()
        self.services.dispose()


def create_kernel(**options: Any) -> Kernel[Any]:
    return Kernel(**options)
