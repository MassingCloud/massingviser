from __future__ import annotations

import contextvars
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .disposable import Disposable, to_disposable
from .errors import KernelError
from .permissions import Identity, PermissionRequest, PermissionService
from .result import Result, attempt_async, err
from .telemetry import NOOP_TELEMETRY, TelemetrySink


@dataclass(frozen=True)
class CommandInvocation:
    command_id: str
    params: Any = None


@dataclass(frozen=True)
class CommandExecutionContext:
    command_id: str
    identity: Identity
    #: Host-supplied cancellation handle, passed through untouched.
    signal: Any = None
    #: 0 for a user-initiated command; higher when one command invokes another.
    depth: int = 0


@dataclass(frozen=True)
class CommandDefinition:
    id: str
    handler: Callable[[Any, CommandExecutionContext], Any]
    title: str | None = None
    description: str | None = None
    #: Permission action checked before the handler runs. Omit for unrestricted commands.
    permission: str | None = None
    #: Builds the invocation that reverses this one. Defining it is what makes a command undoable.
    #:
    #: Expressing undo as *another command* rather than as a closure keeps the history
    #: serialisable and replayable, and means undo goes through the same permission and middleware
    #: path as the original action instead of quietly bypassing it.
    create_inverse: Callable[[Any, Any], CommandInvocation | None] | None = None


@dataclass(frozen=True)
class CommandInfo:
    id: str
    title: str | None
    description: str | None
    permission: str | None
    undoable: bool


CommandMiddleware = Callable[
    [CommandInvocation, Callable[[], Awaitable[Result[Any, KernelError]]]],
    Awaitable[Result[Any, KernelError]],
]

_HistoryMode = Literal["normal", "undo", "redo"]

# Depth and history mode are per-execution-context, not per-bus. Under asyncio two top-level
# commands can be in flight at once; a shared counter would let one see the other's depth and
# silently drop its own undo entry. A ContextVar is inherited by awaits within the same task --
# which is exactly "a command invoked from inside another command's handler" -- and copied for
# tasks spawned elsewhere.
_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "massingviser_command_depth", default=0
)
_mode: contextvars.ContextVar[_HistoryMode] = contextvars.ContextVar(
    "massingviser_command_mode", default="normal"
)


def _default_now() -> float:
    return time.monotonic() * 1000.0


class CommandBus:
    """The single path through which UI actions reach application logic.

    Routing everything through one bus is what makes cross-cutting behaviour possible without
    touching feature code: permissions, telemetry, undo history and audit are all implemented once,
    here, rather than being re-remembered at every button handler.

    ``execute`` returns a ``Result`` and never raises -- a plugin's handler blowing up is an
    ordinary, reportable outcome, not something that should unwind the caller.
    """

    __slots__ = (
        "_commands",
        "_middlewares",
        "_undo_stack",
        "_redo_stack",
        "_permissions",
        "_telemetry",
        "_history_limit",
        "_now",
    )

    def __init__(
        self,
        *,
        permissions: PermissionService | None = None,
        telemetry: TelemetrySink | None = None,
        history_limit: int = 100,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._commands: dict[str, CommandDefinition] = {}
        self._middlewares: list[CommandMiddleware] = []
        self._undo_stack: list[CommandInvocation] = []
        self._redo_stack: list[CommandInvocation] = []
        self._permissions = permissions
        self._telemetry = telemetry or NOOP_TELEMETRY
        self._history_limit = max(0, history_limit)
        self._now = now or _default_now

    def register(self, definition: CommandDefinition) -> Disposable:
        if definition.id in self._commands:
            raise KernelError(
                "COMMAND_DUPLICATE",
                f'Command "{definition.id}" is already registered.',
                {"commandId": definition.id},
            )
        self._commands[definition.id] = definition

        def _unregister() -> None:
            self._commands.pop(definition.id, None)
            # History entries naming a command that no longer exists would fail on undo with a
            # confusing "not found". Dropping them when the plugin unregisters keeps the stacks
            # honest.
            self._purge_history(definition.id)

        return to_disposable(_unregister)

    def use(self, middleware: CommandMiddleware) -> Disposable:
        self._middlewares.append(middleware)

        def _remove() -> None:
            if middleware in self._middlewares:
                self._middlewares.remove(middleware)

        return to_disposable(_remove)

    def has(self, command_id: str) -> bool:
        return command_id in self._commands

    def list(self) -> list[CommandInfo]:
        return [
            CommandInfo(
                id=d.id,
                title=d.title,
                description=d.description,
                permission=d.permission,
                undoable=callable(d.create_inverse),
            )
            for d in self._commands.values()
        ]

    async def execute(
        self,
        command_id: str,
        params: Any = None,
        *,
        signal: Any = None,
        record: bool = True,
    ) -> Result[Any, KernelError]:
        definition = self._commands.get(command_id)
        if definition is None:
            error = KernelError(
                "COMMAND_NOT_FOUND",
                f'No command registered as "{command_id}".',
                {"commandId": command_id},
            )
            self._telemetry.error(error)
            return err(error)

        invocation = CommandInvocation(command_id, params)
        started = self._now()
        depth = _depth.get()
        depth_token = _depth.set(depth + 1)

        try:

            async def core() -> Result[Any, KernelError]:
                if definition.permission and self._permissions is not None:
                    allowed = await self._permissions.require(
                        PermissionRequest(action=definition.permission)
                    )
                    if not allowed.ok:
                        return allowed
                context = CommandExecutionContext(
                    command_id=command_id,
                    identity=(
                        self._permissions.identity
                        if self._permissions is not None
                        else Identity(id="anonymous", roles=())
                    ),
                    signal=signal,
                    depth=depth,
                )
                outcome = await attempt_async(
                    lambda: definition.handler(params, context),
                    "COMMAND_FAILED",
                    f'Command "{command_id}" failed.',
                )
                if outcome.ok:
                    self._record_history(definition, params, outcome.value, depth, record)
                return outcome

            # Middleware is third-party too, so the whole composed chain is guarded rather than
            # just the handler -- a middleware that raises is a failed command, not a failed
            # application.
            chained = await attempt_async(
                self._compose(invocation, core),
                "COMMAND_FAILED",
                f'Command middleware for "{command_id}" failed.',
            )
            result: Result[Any, KernelError] = chained.value if chained.ok else chained

            self._telemetry.timing(
                "command.duration", self._now() - started, {"commandId": command_id}
            )
            self._telemetry.counter(
                "command.ok" if result.ok else "command.error", 1, {"commandId": command_id}
            )
            if not result.ok:
                self._telemetry.error(result.error, {"commandId": command_id})
            return result
        finally:
            _depth.reset(depth_token)

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def history_size(self) -> dict[str, int]:
        """Depth of the undo and redo stacks -- surfaced for diagnostics and UI affordances."""
        return {"undo": len(self._undo_stack), "redo": len(self._redo_stack)}

    async def undo(self) -> Result[Any, KernelError]:
        return await self._replay(self._undo_stack, "undo")

    async def redo(self) -> Result[Any, KernelError]:
        return await self._replay(self._redo_stack, "redo")

    def clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    # -- internals ---------------------------------------------------------------------------

    async def _replay(
        self, stack: list[CommandInvocation], mode: _HistoryMode
    ) -> Result[Any, KernelError]:
        if not stack:
            return err(KernelError("COMMAND_NOT_FOUND", f"Nothing to {mode}.", {"mode": mode}))
        entry = stack.pop()
        mode_token = _mode.set(mode)
        try:
            result = await self.execute(entry.command_id, entry.params)
            # Put it back if the reversal did not take, otherwise the action is stranded: neither
            # applied nor undoable.
            if not result.ok:
                stack.append(entry)
            return result
        finally:
            _mode.reset(mode_token)

    def _record_history(
        self,
        definition: CommandDefinition,
        params: Any,
        result: Any,
        depth: int,
        record: bool,
    ) -> None:
        # Only top-level commands enter the history. A composite command that internally executes
        # sub-commands must be undone as one step, so its children are deliberately not recorded --
        # otherwise a single user action would need several undos to reverse.
        if not record or depth != 0 or definition.create_inverse is None:
            return

        try:
            inverse = definition.create_inverse(params, result)
        except Exception as thrown:  # noqa: BLE001
            self._telemetry.error(
                KernelError(
                    "COMMAND_FAILED",
                    f'create_inverse for "{definition.id}" raised.',
                    {"commandId": definition.id},
                    cause=thrown,
                )
            )
            return
        if inverse is None:
            return

        mode = _mode.get()
        if mode == "undo":
            self._push(self._redo_stack, inverse)
        elif mode == "redo":
            self._push(self._undo_stack, inverse)
        else:
            self._push(self._undo_stack, inverse)
            # A fresh action invalidates the redo branch -- replaying it would apply an inverse
            # computed against a state that no longer exists.
            self._redo_stack.clear()

    def _push(self, stack: list[CommandInvocation], invocation: CommandInvocation) -> None:
        if self._history_limit == 0:
            return
        stack.append(invocation)
        while len(stack) > self._history_limit:
            stack.pop(0)

    def _purge_history(self, command_id: str) -> None:
        for stack in (self._undo_stack, self._redo_stack):
            stack[:] = [entry for entry in stack if entry.command_id != command_id]

    def _compose(
        self,
        invocation: CommandInvocation,
        core: Callable[[], Awaitable[Result[Any, KernelError]]],
    ) -> Callable[[], Awaitable[Result[Any, KernelError]]]:
        next_ = core
        for middleware in reversed(self._middlewares):
            downstream = next_

            # Bind both per-iteration values as defaults; a bare closure would capture the loop
            # variable and every layer would run the last middleware.
            def make(mw: CommandMiddleware = middleware, nxt=downstream):
                return lambda: mw(invocation, nxt)

            next_ = make()
        return next_
