"""Ports for the two things every capability plugin needs and no plugin should reach for directly.

Time and identity are the classic sources of untestable code: a service that calls
``datetime.now()`` or ``uuid4()`` internally produces a different record every run, so its tests
either assert nothing or assert on a moving target. Injecting both makes a plugin's output a pure
function of its input.
"""

from __future__ import annotations

import itertools
import uuid
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...
    def iso(self) -> str: ...


@runtime_checkable
class IdFactory(Protocol):
    def next(self, prefix: str = "") -> str: ...


class SystemClock:
    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def iso(self) -> str:
        return self.now().isoformat().replace("+00:00", "Z")


class FixedClock:
    """A clock that does not move unless told to. For tests and deterministic replays."""

    __slots__ = ("_moment",)

    def __init__(self, moment: datetime | None = None) -> None:
        self._moment = moment or datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._moment

    def iso(self) -> str:
        return self._moment.isoformat().replace("+00:00", "Z")

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._moment = self._moment + timedelta(seconds=seconds)


class UuidIdFactory:
    __slots__ = ()

    def next(self, prefix: str = "") -> str:
        value = uuid.uuid4().hex[:12]
        return f"{prefix}-{value}" if prefix else value


class SequentialIdFactory:
    """Monotonic ids scoped per prefix. Readable in test failures and stable across runs."""

    __slots__ = ("_counters",)

    def __init__(self) -> None:
        self._counters: dict[str, itertools.count] = {}

    def next(self, prefix: str = "") -> str:
        counter = self._counters.get(prefix)
        if counter is None:
            counter = itertools.count(1)
            self._counters[prefix] = counter
        index = next(counter)
        return f"{prefix}-{index}" if prefix else str(index)


DEFAULT_CLOCK: Clock = SystemClock()
DEFAULT_IDS: IdFactory = UuidIdFactory()
