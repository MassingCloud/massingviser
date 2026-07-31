from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from .disposable import Disposable, DisposableStore, to_disposable
from .errors import KernelError
from .result import Result, err, ok

T = TypeVar("T")


@dataclass(eq=False)
class ServiceToken(Generic[T]):
    """A typed key for a service.

    Tokens are *nominal* -- equality is object identity, not the name -- so two unrelated packages
    that both want a ``"logger"`` cannot collide, and the phantom ``T`` lets ``resolve`` infer the
    service type without a cast at the call site.
    """

    name: str

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"ServiceToken({self.name!r})"


def create_service_token(name: str) -> ServiceToken[T]:
    return ServiceToken(name)


ServiceFactory = Callable[["ServiceContainer"], Any]

#: ``singleton`` -- built at most once per owning container, then cached and reused.
#: ``transient`` -- rebuilt on every ``resolve``; the container never caches or disposes these.
ServiceLifetime = Literal["singleton", "transient"]


@dataclass(frozen=True)
class _Registration:
    factory: ServiceFactory
    lifetime: ServiceLifetime
    dispose_on_release: bool


def _is_disposable(value: object) -> bool:
    return callable(getattr(value, "dispose", None))


class ServiceContainer:
    """Hierarchical service registry.

    Child scopes are the isolation mechanism the plugin host relies on: each plugin resolves
    through its own scope, so it can override or add services for itself without mutating the
    kernel's registry, and disposing the scope releases exactly what that plugin created.

    Lookup walks to the parent, but a singleton is cached on the container that *owns* the
    registration -- so a kernel-level service stays a genuine singleton no matter how many plugin
    scopes resolve it.
    """

    __slots__ = (
        "label",
        "_parent",
        "_registrations",
        "_instances",
        "_owned",
        "_resolving",
        "_disposed",
    )

    def __init__(self, label: str = "root", parent: ServiceContainer | None = None) -> None:
        self.label = label
        self._parent = parent
        self._registrations: dict[ServiceToken[Any], _Registration] = {}
        self._instances: dict[ServiceToken[Any], Any] = {}
        self._owned = DisposableStore()
        # Shared across the whole container tree so a cycle spanning parent and child is caught.
        self._resolving: list[ServiceToken[Any]] = parent._resolving if parent else []
        self._disposed = False

    @property
    def disposed(self) -> bool:
        return self._disposed

    def register(
        self,
        token: ServiceToken[T],
        factory: Callable[[ServiceContainer], T],
        *,
        lifetime: ServiceLifetime = "singleton",
        dispose_on_release: bool = True,
    ) -> Disposable:
        self._assert_live()
        if token in self._registrations:
            raise KernelError(
                "SERVICE_DUPLICATE",
                f'Service "{token.name}" is already registered in scope "{self.label}".',
                {"service": token.name, "scope": self.label},
            )
        self._registrations[token] = _Registration(factory, lifetime, dispose_on_release)

        def _unregister() -> None:
            self._release_instance(token)
            self._registrations.pop(token, None)

        return to_disposable(_unregister)

    def register_value(self, token: ServiceToken[T], value: T) -> Disposable:
        """Register an already-constructed value.

        The container does not take ownership of its lifetime.
        """
        return self.register(
            token, lambda _c: value, lifetime="singleton", dispose_on_release=False
        )

    def has(self, token: ServiceToken[Any]) -> bool:
        """True if this container or any ancestor can supply the token."""
        if token in self._registrations:
            return True
        return self._parent.has(token) if self._parent else False

    def resolve(self, token: ServiceToken[T]) -> T:
        self._assert_live()
        owner = self._owner_of(token)
        if owner is None:
            raise KernelError(
                "SERVICE_NOT_FOUND",
                f'No service registered for "{token.name}".',
                {"service": token.name, "scope": self.label},
            )
        return owner._instantiate(token)

    def try_resolve(self, token: ServiceToken[T]) -> Result[T, KernelError]:
        """Non-raising ``resolve``, for callers that treat a missing service as an ordinary branch."""
        try:
            return ok(self.resolve(token))
        except KernelError as thrown:
            return err(thrown)
        except Exception:  # noqa: BLE001
            return err(
                KernelError(
                    "SERVICE_NOT_FOUND",
                    f'Failed to resolve "{token.name}".',
                    {"service": token.name},
                )
            )

    def create_scope(self, label: str) -> ServiceContainer:
        self._assert_live()
        scope = ServiceContainer(label, self)
        self._owned.add(scope)
        return scope

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for token in list(self._instances):
            self._release_instance(token)
        self._instances.clear()
        self._registrations.clear()
        self._owned.dispose()

    # -- internals ---------------------------------------------------------------------------

    def _owner_of(self, token: ServiceToken[Any]) -> ServiceContainer | None:
        if token in self._registrations:
            return self
        return self._parent._owner_of(token) if self._parent else None

    def _instantiate(self, token: ServiceToken[T]) -> T:
        registration = self._registrations.get(token)
        if registration is None:
            raise KernelError(
                "SERVICE_NOT_FOUND",
                f'No service registered for "{token.name}".',
                {"service": token.name},
            )

        if registration.lifetime == "singleton" and token in self._instances:
            return self._instances[token]

        # A factory that resolves its own token -- directly or through a chain -- would otherwise
        # recurse until the stack blows, producing a RecursionError with none of the useful
        # context. Reporting the actual cycle is the difference between a two-minute fix and an
        # afternoon.
        if token in self._resolving:
            cycle = " -> ".join(t.name for t in [*self._resolving, token])
            raise KernelError(
                "SERVICE_CIRCULAR", f"Circular service dependency: {cycle}", {"cycle": cycle}
            )

        self._resolving.append(token)
        try:
            instance = registration.factory(self)
        finally:
            self._resolving.pop()

        if registration.lifetime == "singleton":
            self._instances[token] = instance
            if registration.dispose_on_release and _is_disposable(instance):
                self._owned.add(instance)
        return instance

    def _release_instance(self, token: ServiceToken[Any]) -> None:
        registration = self._registrations.get(token)
        instance = self._instances.pop(token, None)
        if (
            registration is not None
            and registration.dispose_on_release
            and _is_disposable(instance)
        ):
            try:
                instance.dispose()
            except Exception:  # noqa: BLE001
                pass  # a service that raises on teardown must not block the container closing

    def _assert_live(self) -> None:
        if self._disposed:
            raise KernelError(
                "CONTAINER_DISPOSED",
                f'Container "{self.label}" is disposed.',
                {"scope": self.label},
            )
