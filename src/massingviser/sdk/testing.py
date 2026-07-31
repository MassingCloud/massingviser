"""Test a plugin against a real kernel rather than a mock.

A mocked kernel proves a plugin calls the API you thought it would. It does not prove the plugin
activates, that its registrations roll back when it fails, or that its dependencies resolve --
which is where plugin bugs actually live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..kernel import (
    InMemoryTelemetrySink,
    Kernel,
    KernelError,
    MemoryStorageAdapter,
    Result,
    create_kernel,
)
from ..schema.versioning import MigrationRegistry, create_default_migration_registry
from .runtime import FixedClock, SequentialIdFactory


@dataclass
class TestHarness:
    kernel: Kernel[Any]
    telemetry: InMemoryTelemetrySink
    storage: MemoryStorageAdapter
    migrator: MigrationRegistry
    clock: FixedClock
    ids: SequentialIdFactory

    async def load(self, *plugins: Any) -> None:
        """Register and activate plugins, raising on the first failure.

        Raising rather than returning a ``Result`` is deliberate: in a test, a plugin that will not
        activate is a failed test, and unwrapping it at every call site is noise.
        """
        for plugin in plugins:
            registered = self.kernel.use(plugin)
            if not registered.ok:
                raise registered.error
        report = await self.kernel.start()
        if report.failed:
            plugin_id, error = report.failed[0]
            raise error
        if report.skipped:
            plugin_id, reason = report.skipped[0]
            raise KernelError(
                "PLUGIN_DEPENDENCY_MISSING", f'Plugin "{plugin_id}" was skipped: {reason}'
            )

    async def execute(self, command_id: str, params: Any = None) -> Result[Any, KernelError]:
        return await self.kernel.commands.execute(command_id, params)

    def capability(self, token: Any) -> Any:
        """Resolve a capability, raising if absent. Same reasoning as ``load``."""
        result = self.kernel.capabilities.require(token)
        if not result.ok:
            raise result.error
        return result.value

    async def dispose(self) -> None:
        await self.kernel.stop()
        self.kernel.dispose()


def create_test_harness(**kernel_options: Any) -> TestHarness:
    telemetry = InMemoryTelemetrySink()
    storage = MemoryStorageAdapter()
    migrator = create_default_migration_registry()
    clock = FixedClock()
    ids = SequentialIdFactory()

    kernel_options.setdefault("telemetry", telemetry)
    kernel_options.setdefault("storage", storage)
    kernel_options.setdefault("migrator", migrator)
    kernel_options.setdefault("now", clock.now)

    kernel = create_kernel(**kernel_options)
    return TestHarness(kernel, telemetry, storage, migrator, clock, ids)
