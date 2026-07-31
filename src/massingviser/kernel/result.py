"""Explicit success/failure values.

The kernel's non-negotiable is that *no plugin can crash the base viewer*. Exceptions propagate by
default and satisfy that only if every call site remembers to guard; a ``Result`` inverts this,
making the failure branch visible in the type. Kernel APIs that invoke plugin-supplied code return
``Result`` rather than raising.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .errors import KernelError, KernelErrorCode, to_kernel_error

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    @property
    def ok(self) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

    @property
    def ok(self) -> bool:
        return False

    def __bool__(self) -> bool:
        return False


Result = Ok[T] | Err[E]


def ok(value: T = None) -> Ok[T]:  # type: ignore[assignment]
    return Ok(value)


def err(error: E) -> Err[E]:
    return Err(error)


def is_ok(result: Result[T, E]) -> bool:
    return result.ok


def is_err(result: Result[T, E]) -> bool:
    return not result.ok


def unwrap(result: Result[T, E]) -> T:
    """Unwrap a success, or raise the failure. Use at trust boundaries, not in plugin code."""
    if result.ok:
        return result.value  # type: ignore[union-attr]
    error = result.error  # type: ignore[union-attr]
    if isinstance(error, BaseException):
        raise error
    raise KernelError("COMMAND_FAILED", str(error))


def unwrap_or(result: Result[T, E], fallback: T) -> T:
    """Unwrap a success, or fall back. Never raises."""
    return result.value if result.ok else fallback  # type: ignore[union-attr]


def map_ok(result: Result[T, E], fn: Callable[[T], U]) -> Result[U, E]:
    return Ok(fn(result.value)) if result.ok else result  # type: ignore[union-attr,return-value]


def map_err(result: Result[T, E], fn: Callable[[E], F]) -> Result[T, F]:
    return result if result.ok else Err(fn(result.error))  # type: ignore[union-attr,return-value]


def attempt(
    fn: Callable[[], T],
    code: KernelErrorCode,
    message: str,
) -> Result[T, KernelError]:
    """Run a synchronous function, converting any raise into an ``Err[KernelError]``."""
    try:
        return Ok(fn())
    except Exception as thrown:  # noqa: BLE001 -- funnelling third-party failures is the point
        return Err(to_kernel_error(thrown, code, message))


async def attempt_async(
    fn: Callable[[], Any],
    code: KernelErrorCode,
    message: str,
) -> Result[Any, KernelError]:
    """Run a possibly-async function, converting a synchronous raise *or* a failed await into an
    ``Err[KernelError]``.

    Plugin entry points are untyped by nature -- a plugin may declare ``activate`` as a plain
    ``def`` and still return a coroutine -- so both paths need catching.

    ``CancelledError`` derives from ``BaseException``, so it deliberately passes through: a
    cancelled task is the caller unwinding, not a plugin failing.
    """
    try:
        outcome = fn()
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return Ok(outcome)
    except Exception as thrown:  # noqa: BLE001
        return Err(to_kernel_error(thrown, code, message))


async def resolve(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise return it unchanged."""
    return await value if inspect.isawaitable(value) else value
