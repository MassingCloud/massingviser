"""Getting a synchronous browser callback onto the kernel's event loop.

The kernel is async and single-threaded by design: the command bus, the state store and the plugin
host all assume that no two mutations interleave. viser's GUI and scene callbacks arrive on its own
websocket threads. Letting those threads touch the kernel directly would be a data race that
manifests as a mass with the wrong story count once a week and never reproduces.

So the kernel gets one owning thread with one event loop, and everything from the browser is
marshalled onto it. This class is the only place that boundary is crossed.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ..kernel import Kernel, KernelError, Result, err

T = TypeVar("T")

#: Ceiling on a single marshalled call. A command that hangs must surface as an error in the
#: browser rather than wedging the websocket thread that dispatched it.
DEFAULT_TIMEOUT_SECONDS = 30.0


class KernelBridge:
    """Owns the kernel's event loop thread and marshals work onto it."""

    __slots__ = ("_kernel", "_loop", "_thread", "_ready", "_closed")

    def __init__(self, kernel: Kernel[Any]) -> None:
        self._kernel = kernel
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="massingviser-kernel", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    @property
    def kernel(self) -> Kernel[Any]:
        return self._kernel

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The kernel's own loop, so a second front end can marshal onto the same thread.

        Exposed rather than duplicated: two front ends each owning a loop would be two kernels, and
        an edit in one would be invisible to the other.
        """
        return self._loop

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def run(self, coro: Awaitable[T], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> T:
        """Await a coroutine on the kernel loop and return its value on the calling thread."""
        if self._closed:
            raise RuntimeError("The kernel bridge is closed.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        return future.result(timeout)

    def read(self, fn: Callable[[], T], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> T:
        """Run a *synchronous* kernel read on the loop thread.

        Reads go through here too. A plain call from a websocket thread would see a record store
        mid-write -- the state store swaps whole tuples, so a reader can legitimately observe the
        old collection, but it can also observe a half-updated index.
        """
        if self._closed:
            raise RuntimeError("The kernel bridge is closed.")
        future: asyncio.Future[T] = asyncio.run_coroutine_threadsafe(  # type: ignore[assignment]
            _call(fn), self._loop
        )
        return future.result(timeout)  # type: ignore[union-attr]

    def execute(
        self, command_id: str, params: Any = None, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> Result[Any, KernelError]:
        """Run a command. Never raises -- the bus already returns a ``Result``, and a timeout or a
        dead loop is converted into one so a GUI callback cannot take down the websocket thread."""
        if self._closed:
            # Checked before the coroutine is built: constructing one and then abandoning it emits
            # a "never awaited" warning that looks like a leak and is not.
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Command "{command_id}" was not dispatched: the session is closed.',
                    {"commandId": command_id},
                )
            )
        try:
            return self.run(self._kernel.commands.execute(command_id, params), timeout)
        except Exception as thrown:  # noqa: BLE001
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Command "{command_id}" could not be dispatched: {thrown}',
                    {"commandId": command_id},
                )
            )

    def start(self) -> Any:
        return self.run(self._kernel.start())

    def undo(self) -> Result[Any, KernelError]:
        return self.run(self._kernel.commands.undo())

    def redo(self) -> Result[Any, KernelError]:
        return self.run(self._kernel.commands.redo())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(self._kernel.stop(), self._loop).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass  # shutting down: a plugin that will not deactivate must not block process exit
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._kernel.dispose()

    def __enter__(self) -> KernelBridge:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


async def _call(fn: Callable[[], T]) -> T:
    return fn()
