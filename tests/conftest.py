"""Test support.

Async tests run through a plain ``asyncio.run`` wrapper rather than ``pytest-asyncio``. The kernel
is the thing under test and it makes no assumptions about the loop it runs on; adding a plugin to
manage one would be a dependency that exists only to hide two lines.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

import pytest


def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Wrap coroutine test functions so they run on a fresh event loop."""
    for item in items:
        test = getattr(item, "obj", None)
        if inspect.iscoroutinefunction(test):
            item.obj = _sync(test)


def _sync(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@pytest.fixture()
def harness():
    from massingviser.sdk import create_test_harness

    return create_test_harness()
