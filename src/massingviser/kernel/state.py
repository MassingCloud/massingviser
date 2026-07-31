from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

from .disposable import Disposable, to_disposable
from .errors import KernelError

T = TypeVar("T")
U = TypeVar("U")

StateListener = Callable[[Any, Any], None]

_PRIMITIVES = (int, float, str, bool, bytes, type(None))


def same_value(a: Any, b: Any) -> bool:
    """Identity for objects, value equality for primitives -- JavaScript's ``Object.is``.

    Reference identity is the load-bearing half: state values are treated as immutable, so a write
    that produces a new object *is* a change even when its contents happen to match, and a write
    that hands back the same object is not. Comparing containers by value here would suppress
    notifications the record store depends on.
    """
    if a is b:
        # NaN is the one value that is not equal to itself; Object.is calls it equal.
        return True
    if isinstance(a, _PRIMITIVES) and isinstance(b, _PRIMITIVES):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        # bool is a subclass of int in Python; keep True and 1 distinct as JS would.
        if isinstance(a, bool) != isinstance(b, bool):
            return False
        return a == b
    return False


@dataclass(frozen=True)
class StateChange:
    namespace: str
    next: Any
    previous: Any


class _SliceRecord:
    __slots__ = ("value", "listeners")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.listeners: dict[StateListener, None] = {}


class Slice(Generic[T]):
    """A namespaced, independently-subscribable region of application state.

    State is deliberately partitioned rather than held as one global object: a plugin owns its own
    namespace, subscribers only wake for changes to the slice they asked about, and persistence can
    version each namespace's payload separately (which is what makes per-plugin schema migration
    possible at all).
    """

    __slots__ = ("namespace", "_store", "_record")

    def __init__(self, namespace: str, store: "StateStore", record: _SliceRecord) -> None:
        self.namespace = namespace
        self._store = store
        self._record = record

    def get(self) -> T:
        return self._record.value

    def set(self, next_: T) -> None:
        self._store._commit(self.namespace, self._record, next_)

    def update(self, recipe: Callable[[T], T]) -> None:
        self._store._commit(self.namespace, self._record, recipe(self._record.value))

    def subscribe(self, listener: Callable[[T, T], None]) -> Disposable:
        self._record.listeners[listener] = None
        return to_disposable(lambda: self._record.listeners.pop(listener, None))

    def select(
        self,
        selector: Callable[[T], U],
        listener: Callable[[U, U], None],
        equals: Callable[[U, U], bool] = same_value,
    ) -> Disposable:
        """Subscribe to a derived value, notified only when the projection actually changes."""
        current = selector(self._record.value)

        def _on_change(next_: T, _previous: T) -> None:
            nonlocal current
            projected = selector(next_)
            if equals(projected, current):
                return
            previous = current
            current = projected
            listener(projected, previous)

        return self.subscribe(_on_change)


class StateStore:
    """The central store.

    Values are treated as immutable -- ``update`` must return a new value rather than mutating in
    place. That is what lets ``select`` compare by identity and what makes a snapshot safe to hand
    to the persistence engine without cloning.
    """

    __slots__ = ("_slices", "_pending", "_observers", "_transaction_depth", "_dirty")

    def __init__(self) -> None:
        self._slices: dict[str, _SliceRecord] = {}
        # State restored before its owning slice exists.
        #
        # Persistence loads a whole project up front, but a plugin's slice only appears when that
        # plugin activates -- which may be seconds later, or never. Parking the value here means
        # load order stops mattering: whoever arrives second finds the other waiting.
        self._pending: dict[str, Any] = {}
        self._observers: dict[Callable[[StateChange], None], None] = {}
        self._transaction_depth = 0
        self._dirty: dict[str, Any] = {}

    def define_slice(self, namespace: str, initial: T) -> Slice[T]:
        if namespace in self._slices:
            raise KernelError(
                "STATE_NAMESPACE_CONFLICT",
                f'State namespace "{namespace}" is already defined.',
                {"namespace": namespace},
            )
        restored = namespace in self._pending
        record = _SliceRecord(self._pending.pop(namespace) if restored else initial)
        self._slices[namespace] = record
        return Slice(namespace, self, record)

    def has_slice(self, namespace: str) -> bool:
        return namespace in self._slices

    def get_slice(self, namespace: str) -> "Slice[Any] | None":
        record = self._slices.get(namespace)
        return Slice(namespace, self, record) if record else None

    def remove_slice(self, namespace: str) -> None:
        """Release a slice and its listeners. Used when a plugin deactivates."""
        record = self._slices.pop(namespace, None)
        if record is not None:
            record.listeners.clear()

    def snapshot(self) -> dict[str, Any]:
        """A plain dict of every live slice, suitable for handing to persistence."""
        out = {namespace: record.value for namespace, record in self._slices.items()}
        out.update(self._pending)
        return out

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot.

        Namespaces with no live slice are parked (see ``_pending``) rather than dropped, so a
        plugin activating later still receives its persisted state.
        """

        def _apply() -> None:
            for namespace, value in snapshot.items():
                record = self._slices.get(namespace)
                if record is not None:
                    self._commit(namespace, record, value)
                else:
                    self._pending[namespace] = value

        self.transaction(_apply)

    def transaction(self, fn: Callable[[], None]) -> None:
        """Batch writes so subscribers see one notification per slice at the end.

        Without this, a command touching five slices produces five render passes; with it, one.
        Nested transactions are counted, not re-entered, so a composite command can safely wrap
        sub-commands that transact on their own.
        """
        self._transaction_depth += 1
        try:
            fn()
        finally:
            self._transaction_depth -= 1
            if self._transaction_depth == 0:
                self._flush()

    def observe(self, observer: Callable[[StateChange], None]) -> Disposable:
        self._observers[observer] = None
        return to_disposable(lambda: self._observers.pop(observer, None))

    # -- internals ---------------------------------------------------------------------------

    def _commit(self, namespace: str, record: _SliceRecord, next_: Any) -> None:
        previous = record.value
        if same_value(previous, next_):
            return  # no-op writes must not wake subscribers
        record.value = next_

        if self._transaction_depth > 0:
            # Keep the *earliest* previous value: at flush time a subscriber cares about the state
            # before the transaction opened, not about intermediate steps it never observed.
            self._dirty.setdefault(namespace, previous)
            return
        self._notify(namespace, record, next_, previous)

    def _flush(self) -> None:
        if not self._dirty:
            return
        dirty = self._dirty
        self._dirty = {}
        for namespace, previous in dirty.items():
            record = self._slices.get(namespace)
            if record is not None:
                self._notify(namespace, record, record.value, previous)

    def _notify(self, namespace: str, record: _SliceRecord, next_: Any, previous: Any) -> None:
        for listener in list(record.listeners):
            try:
                listener(next_, previous)
            except Exception:  # noqa: BLE001
                # A raising subscriber must not prevent the remaining subscribers from seeing the
                # change, nor unwind the command that performed the write.
                pass
        change = StateChange(namespace, next_, previous)
        for observer in list(self._observers):
            try:
                observer(change)
            except Exception:  # noqa: BLE001
                pass  # observers are diagnostics only
