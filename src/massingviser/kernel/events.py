from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .disposable import Disposable, to_disposable
from .errors import KernelError, to_kernel_error

EventHandler = Callable[[Any], None]


@dataclass(frozen=True)
class EmitReport:
    type: str
    delivered: int
    #: Failures raised by individual handlers. Empty on a clean emit.
    errors: tuple[KernelError, ...] = ()


class EventBus:
    """Synchronous publish/subscribe.

    Two properties the rest of the kernel depends on:

    1. **``emit`` never raises.** A subscriber is frequently plugin code, and one plugin raising in
       a ``selection.changed`` handler must not abort delivery to the other nine or unwind the
       caller that published the event. Failures are collected into the report and surfaced through
       telemetry instead.
    2. **The handler list is snapshotted before dispatch.** Handlers routinely subscribe or
       unsubscribe in response to an event; without a snapshot that mutates the collection
       mid-iteration, which silently skips handlers or delivers to one that just unsubscribed.
    """

    __slots__ = ("_handlers", "_observers")

    def __init__(self) -> None:
        # dict preserves insertion order, which gives handlers a stable, predictable delivery order
        # while still supporting O(1) removal -- a plain list would make unsubscribe O(n).
        self._handlers: dict[str, dict[EventHandler, None]] = {}
        self._observers: dict[Callable[[str, Any], None], None] = {}

    def on(self, type_: str, handler: EventHandler) -> Disposable:
        bucket = self._handlers.setdefault(type_, {})
        bucket[handler] = None

        def _off() -> None:
            current = self._handlers.get(type_)
            if current is None:
                return
            current.pop(handler, None)
            if not current:
                self._handlers.pop(type_, None)

        return to_disposable(_off)

    def once(self, type_: str, handler: EventHandler) -> Disposable:
        holder: dict[str, Disposable] = {}

        def _wrapped(payload: Any) -> None:
            holder["subscription"].dispose()
            handler(payload)

        subscription = self.on(type_, _wrapped)
        holder["subscription"] = subscription
        return subscription

    def observe(self, observer: Callable[[str, Any], None]) -> Disposable:
        """Observe every event regardless of topic.

        Intended for diagnostics, recording and telemetry -- observers cannot alter delivery and
        their failures are swallowed, so this is never a back-channel for business logic.
        """
        self._observers[observer] = None
        return to_disposable(lambda: self._observers.pop(observer, None))

    def emit(self, type_: str, payload: Any = None) -> EmitReport:
        errors: list[KernelError] = []
        delivered = 0

        bucket = self._handlers.get(type_)
        if bucket:
            for handler in list(bucket):
                try:
                    handler(payload)
                    delivered += 1
                except Exception as thrown:  # noqa: BLE001
                    errors.append(
                        to_kernel_error(
                            thrown, "COMMAND_FAILED", f'Event handler for "{type_}" failed.'
                        )
                    )

        for observer in list(self._observers):
            try:
                observer(type_, payload)
            except Exception:  # noqa: BLE001
                pass  # diagnostics must never influence delivery

        return EmitReport(type=type_, delivered=delivered, errors=tuple(errors))

    def listener_count(self, type_: str) -> int:
        return len(self._handlers.get(type_, ()))

    def clear(self) -> None:
        self._handlers.clear()
        self._observers.clear()
