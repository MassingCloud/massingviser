"""Where diagnostics go.

The kernel emits telemetry unconditionally and lets the host decide whether it is discarded, kept
in memory, or shipped somewhere. Nothing in the kernel reads telemetry back, so a sink is always
safe to replace with the no-op.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from .disposable import Disposable, to_disposable
from .errors import KernelError

TelemetryTags = Mapping[str, "str | int | float | bool"]


@runtime_checkable
class TelemetrySink(Protocol):
    def counter(self, name: str, value: float = 1, tags: TelemetryTags | None = None) -> None: ...
    def timing(self, name: str, milliseconds: float, tags: TelemetryTags | None = None) -> None: ...
    def event(self, name: str, data: Mapping[str, Any] | None = None) -> None: ...
    def error(self, error: KernelError, context: Mapping[str, Any] | None = None) -> None: ...


class NoopTelemetrySink:
    __slots__ = ()

    def counter(self, name: str, value: float = 1, tags: TelemetryTags | None = None) -> None:
        return None

    def timing(self, name: str, milliseconds: float, tags: TelemetryTags | None = None) -> None:
        return None

    def event(self, name: str, data: Mapping[str, Any] | None = None) -> None:
        return None

    def error(self, error: KernelError, context: Mapping[str, Any] | None = None) -> None:
        return None


NOOP_TELEMETRY: TelemetrySink = NoopTelemetrySink()


@dataclass(frozen=True)
class TelemetryRecord:
    kind: Literal["counter", "timing", "event", "error"]
    name: str
    value: float | None = None
    tags: TelemetryTags | None = None
    data: Mapping[str, Any] | None = None
    error: KernelError | None = None


class InMemoryTelemetrySink:
    """Retains a bounded ring of recent records.

    Bounded on purpose: this backs the diagnostics panel and long-running sessions would otherwise
    grow it without limit -- a memory leak dressed up as observability.
    """

    __slots__ = ("_records",)

    def __init__(self, limit: int = 1000) -> None:
        self._records: deque[TelemetryRecord] = deque(maxlen=max(1, limit))

    @property
    def records(self) -> list[TelemetryRecord]:
        return list(self._records)

    def counter(self, name: str, value: float = 1, tags: TelemetryTags | None = None) -> None:
        self._records.append(TelemetryRecord("counter", name, value, tags))

    def timing(self, name: str, milliseconds: float, tags: TelemetryTags | None = None) -> None:
        self._records.append(TelemetryRecord("timing", name, milliseconds, tags))

    def event(self, name: str, data: Mapping[str, Any] | None = None) -> None:
        self._records.append(TelemetryRecord("event", name, data=data))

    def error(self, error: KernelError, context: Mapping[str, Any] | None = None) -> None:
        self._records.append(TelemetryRecord("error", error.code, data=context, error=error))

    def clear(self) -> None:
        self._records.clear()


class CompositeTelemetrySink:
    """Fans out to several sinks, isolating each so one failing sink cannot suppress the others."""

    __slots__ = ("_sinks",)

    def __init__(self) -> None:
        self._sinks: list[TelemetrySink] = []

    def add(self, sink: TelemetrySink) -> Disposable:
        self._sinks.append(sink)

        def _remove() -> None:
            if sink in self._sinks:
                self._sinks.remove(sink)

        return to_disposable(_remove)

    def _each(self, fn: Callable[[TelemetrySink], None]) -> None:
        for sink in list(self._sinks):
            try:
                fn(sink)
            except Exception:  # noqa: BLE001
                pass  # telemetry is strictly best-effort; a broken sink is never an app failure

    def counter(self, name: str, value: float = 1, tags: TelemetryTags | None = None) -> None:
        self._each(lambda s: s.counter(name, value, tags))

    def timing(self, name: str, milliseconds: float, tags: TelemetryTags | None = None) -> None:
        self._each(lambda s: s.timing(name, milliseconds, tags))

    def event(self, name: str, data: Mapping[str, Any] | None = None) -> None:
        self._each(lambda s: s.event(name, data))

    def error(self, error: KernelError, context: Mapping[str, Any] | None = None) -> None:
        self._each(lambda s: s.error(error, context))
