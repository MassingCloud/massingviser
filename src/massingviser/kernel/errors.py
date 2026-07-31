"""Kernel error taxonomy.

Every failure the kernel can produce carries a stable machine-readable ``code``. Host applications
and plugins branch on the code, never on the message text -- messages are for humans and are free
to change, codes are part of the kernel's contract and are not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

KernelErrorCode = Literal[
    "SERVICE_NOT_FOUND",
    "SERVICE_DUPLICATE",
    "SERVICE_CIRCULAR",
    "CONTAINER_DISPOSED",
    "COMMAND_NOT_FOUND",
    "COMMAND_DUPLICATE",
    "COMMAND_FAILED",
    "PERMISSION_DENIED",
    "CAPABILITY_NOT_FOUND",
    "CAPABILITY_VERSION_MISMATCH",
    "PLUGIN_DUPLICATE",
    "PLUGIN_NOT_FOUND",
    "PLUGIN_DEPENDENCY_MISSING",
    "PLUGIN_DEPENDENCY_CYCLE",
    "PLUGIN_API_INCOMPATIBLE",
    "PLUGIN_ACTIVATION_FAILED",
    "PLUGIN_QUARANTINED",
    "STORAGE_FAILED",
    "MIGRATION_FAILED",
    "MIGRATION_PATH_MISSING",
    "SCHEMA_VERSION_UNSUPPORTED",
    "STATE_NAMESPACE_CONFLICT",
]


class KernelError(Exception):
    """A kernel failure with a stable code and structured, loggable context."""

    __slots__ = ("code", "message", "details", "cause")

    def __init__(
        self,
        code: KernelErrorCode,
        message: str,
        details: Mapping[str, Any] | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code: KernelErrorCode = code
        self.message = message
        # Arbitrary structured context. Safe to log; must never contain secrets.
        self.details: Mapping[str, Any] = MappingProxyType(dict(details or {}))
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def __repr__(self) -> str:
        return f"KernelError({self.code!r}, {self.message!r})"

    def __str__(self) -> str:
        return self.message


def to_kernel_error(
    value: object,
    fallback_code: KernelErrorCode,
    fallback_message: str,
) -> KernelError:
    """Coerce anything raised into a ``KernelError``.

    Plugin code is third-party by definition and can raise a bare string via ``TypeError``, a
    custom exception whose ``__str__`` raises again, or something that is not an exception at all.
    Everything crossing the kernel boundary funnels through here so the rest of the system only
    ever handles one error shape.
    """
    if isinstance(value, KernelError):
        return value
    if isinstance(value, BaseException):
        try:
            text = str(value)
        except Exception:  # a __str__ that raises is exactly the case this guard exists for
            text = value.__class__.__name__
        return KernelError(fallback_code, text or fallback_message, {}, cause=value)

    try:
        described = value if isinstance(value, str) else json.dumps(value)
    except (TypeError, ValueError):
        described = str(value)
    return KernelError(fallback_code, described or fallback_message, {"thrown": described})
