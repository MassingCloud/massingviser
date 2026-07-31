"""A scoped logger backed by the telemetry sink.

Logging routes through telemetry rather than ``print`` or the stdlib ``logging`` root so a host can
capture plugin output the same way it captures everything else -- and so a headless or embedded
deployment has somewhere for it to go other than a terminal nobody is reading.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from .errors import to_kernel_error
from .telemetry import TelemetrySink

LogLevel = Literal["debug", "info", "warn", "error"]


@runtime_checkable
class Logger(Protocol):
    def debug(self, message: str, data: Mapping[str, Any] | None = None) -> None: ...
    def info(self, message: str, data: Mapping[str, Any] | None = None) -> None: ...
    def warn(self, message: str, data: Mapping[str, Any] | None = None) -> None: ...
    def error(
        self,
        message: str,
        cause: object = None,
        data: Mapping[str, Any] | None = None,
    ) -> None: ...


class _ScopedLogger:
    __slots__ = ("_telemetry", "_scope")

    def __init__(self, telemetry: TelemetrySink, scope: str) -> None:
        self._telemetry = telemetry
        self._scope = scope

    def _emit(self, level: LogLevel, message: str, data: Mapping[str, Any] | None) -> None:
        self._telemetry.event(
            f"log.{level}", {"scope": self._scope, "message": message, **dict(data or {})}
        )

    def debug(self, message: str, data: Mapping[str, Any] | None = None) -> None:
        self._emit("debug", message, data)

    def info(self, message: str, data: Mapping[str, Any] | None = None) -> None:
        self._emit("info", message, data)

    def warn(self, message: str, data: Mapping[str, Any] | None = None) -> None:
        self._emit("warn", message, data)

    def error(
        self,
        message: str,
        cause: object = None,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self._telemetry.error(
            to_kernel_error(cause, "PLUGIN_ACTIVATION_FAILED", message),
            {"scope": self._scope, "message": message, **dict(data or {})},
        )


def create_logger(telemetry: TelemetrySink, scope: str) -> Logger:
    return _ScopedLogger(telemetry, scope)
