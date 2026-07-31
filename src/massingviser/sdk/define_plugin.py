from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..kernel import KERNEL_API_VERSION, PluginContext, PluginDependency, PluginManifest


@dataclass(frozen=True)
class _DefinedPlugin:
    manifest: PluginManifest
    _activate: Callable[[PluginContext], Any]
    _deactivate: Callable[[], Any] | None = None

    def activate(self, context: PluginContext) -> Any:
        return self._activate(context)

    def deactivate(self) -> Any:
        return self._deactivate() if self._deactivate is not None else None


def define_plugin(
    *,
    id: str,
    version: str,
    activate: Callable[[PluginContext], Any],
    name: str | None = None,
    description: str | None = None,
    api_version: str | None = None,
    dependencies: Sequence[PluginDependency] = (),
    permissions: Sequence[str] = (),
    deactivate: Callable[[], Any] | None = None,
) -> _DefinedPlugin:
    """Build a ``Plugin`` from a flat definition.

    The value over writing the object by hand is the manifest defaults -- chiefly ``api_version``.
    A plugin that forgets it would either be rejected at load or, worse, silently claim
    compatibility it has not been tested for.
    """
    manifest = PluginManifest(
        id=id,
        version=version,
        api_version=api_version or f"^{KERNEL_API_VERSION}",
        name=name,
        description=description,
        dependencies=tuple(dependencies),
        permissions=tuple(permissions),
    )
    return _DefinedPlugin(manifest, activate, deactivate)
