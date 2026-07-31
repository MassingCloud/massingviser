from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .disposable import Disposable, to_disposable
from .errors import KernelError
from .result import Result, err, ok
from .semver import satisfies

T = TypeVar("T")


@dataclass(frozen=True)
class CapabilityToken(Generic[T]):
    """A named, versioned extension point.

    Capabilities are how a business feature attaches without the kernel knowing it exists. The
    kernel ships the *token* (a contract) and never an implementation; a plugin supplies the
    implementation; consumers ask for the token. Nothing in the kernel imports a feature package,
    which is the property that keeps it small and stable across releases.
    """

    id: str


def create_capability_token(id: str) -> CapabilityToken[T]:
    return CapabilityToken(id)


@dataclass(frozen=True)
class CapabilityProvider(Generic[T]):
    token: str
    value: T
    #: Semantic version of the *implementation*, matched against consumer ranges.
    version: str
    plugin_id: str | None
    #: Higher wins when several plugins provide the same capability. Defaults to 0.
    priority: int


class CapabilityRegistry:
    """Registry of capability implementations.

    Several providers may register against one token -- two family-repository adapters, say -- so
    lookups return the highest-priority match and ``get_all`` exposes the rest for hosts that want
    to aggregate rather than choose.
    """

    __slots__ = ("_providers", "_listeners")

    def __init__(self) -> None:
        self._providers: dict[str, list[CapabilityProvider[Any]]] = {}
        self._listeners: dict[Callable[[str], None], None] = {}

    def provide(
        self,
        token: CapabilityToken[T],
        value: T,
        *,
        version: str = "1.0.0",
        plugin_id: str | None = None,
        priority: int = 0,
    ) -> Disposable:
        provider = CapabilityProvider(token.id, value, version, plugin_id, priority)
        providers = self._providers.setdefault(token.id, [])
        providers.append(provider)
        # Sorted on write so every read is O(1) -- capability lookup sits on hot UI paths.
        # `sort` is stable, so equal priorities keep registration order.
        providers.sort(key=lambda p: -p.priority)
        self._notify(token.id)

        def _revoke() -> None:
            current = self._providers.get(token.id)
            if current is None:
                return
            for index, candidate in enumerate(current):
                if candidate is provider:
                    current.pop(index)
                    break
            if not current:
                self._providers.pop(token.id, None)
            self._notify(token.id)

        return to_disposable(_revoke)

    def has(self, token: CapabilityToken[Any]) -> bool:
        return bool(self._providers.get(token.id))

    def get(self, token: CapabilityToken[T], *, version: str | None = None) -> T | None:
        """Highest-priority implementation satisfying the optional version range, or ``None``."""
        providers = self._providers.get(token.id)
        if not providers:
            return None
        if version is None:
            return providers[0].value
        for provider in providers:
            if satisfies(provider.version, version):
                return provider.value
        return None

    def require(
        self, token: CapabilityToken[T], *, version: str | None = None
    ) -> Result[T, KernelError]:
        """Like ``get``, but reports *why* it failed -- absent versus present-but-incompatible."""
        providers = self._providers.get(token.id)
        if not providers:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    f'No provider for capability "{token.id}".',
                    {"capability": token.id},
                )
            )
        value = self.get(token, version=version)
        if value is None:
            return err(
                KernelError(
                    "CAPABILITY_VERSION_MISMATCH",
                    f'No provider for "{token.id}" satisfies "{version}".',
                    {
                        "capability": token.id,
                        "requested": version,
                        "available": [p.version for p in providers],
                    },
                )
            )
        return ok(value)

    def get_all(self, token: CapabilityToken[T]) -> tuple[CapabilityProvider[T], ...]:
        return tuple(self._providers.get(token.id, ()))

    def tokens(self) -> tuple[str, ...]:
        """Every token that currently has at least one provider. Used by the diagnostics surface."""
        return tuple(self._providers)

    def on_did_change(self, listener: Callable[[str], None]) -> Disposable:
        self._listeners[listener] = None
        return to_disposable(lambda: self._listeners.pop(listener, None))

    def _notify(self, token_id: str) -> None:
        for listener in list(self._listeners):
            try:
                listener(token_id)
            except Exception:  # noqa: BLE001
                pass  # registry bookkeeping must not fail because a listener did
