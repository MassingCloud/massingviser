"""Deterministic teardown.

Every kernel subsystem hands back a ``Disposable`` instead of an ad-hoc ``off()``/``remove()``
pair. That matters most for plugins: deactivating a plugin must reliably release its
subscriptions, commands, panels and services, and the only way to guarantee that is for the plugin
host to hold a single store it can drain.
"""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class Disposable(Protocol):
    def dispose(self) -> None: ...


TDisposable = TypeVar("TDisposable", bound=Disposable)


class _FnDisposable:
    __slots__ = ("_fn", "_done")

    def __init__(self, fn: Callable[[], None]) -> None:
        self._fn = fn
        self._done = False

    def dispose(self) -> None:
        if self._done:
            return  # idempotent: double-dispose is a no-op, not a second side effect
        self._done = True
        self._fn()


def to_disposable(fn: Callable[[], None]) -> Disposable:
    return _FnDisposable(fn)


class _NoopDisposable:
    __slots__ = ()

    def dispose(self) -> None:
        return None


#: A ``Disposable`` that does nothing. Useful as a safe return from a no-op registration path.
NOOP_DISPOSABLE: Disposable = _NoopDisposable()


class DisposableStore:
    """An ordered collection of disposables, drained in reverse registration order.

    Reverse order is deliberate -- later registrations may depend on earlier ones (a panel
    registered after the service it renders), so tearing down newest-first mirrors construction and
    avoids disposing something still in use.
    """

    __slots__ = ("_items", "_disposed")

    def __init__(self) -> None:
        self._items: list[Disposable] = []
        self._disposed = False

    @property
    def disposed(self) -> bool:
        return self._disposed

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size(self) -> int:
        return len(self._items)

    def add(self, item: TDisposable) -> TDisposable:
        """Add a disposable.

        If the store is already disposed the item is disposed immediately rather than retained --
        otherwise a late registration during teardown would leak forever.
        """
        if self._disposed:
            item.dispose()
            return item
        self._items.append(item)
        return item

    def dispose(self) -> None:
        self.dispose_collecting()

    def dispose_collecting(self) -> list[BaseException]:
        """Dispose everything, collecting rather than propagating failures.

        One badly-behaved disposable must not strand the rest -- a plugin whose panel teardown
        raises would otherwise leave its event subscriptions live and keep receiving kernel traffic
        after deactivation. Errors are returned so the caller (the plugin host) can quarantine and
        report.
        """
        if self._disposed:
            return []
        self._disposed = True
        errors: list[BaseException] = []
        for item in reversed(self._items):
            try:
                item.dispose()
            except Exception as thrown:  # noqa: BLE001
                errors.append(thrown)
        self._items = []
        return errors

    def __enter__(self) -> "DisposableStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.dispose()
