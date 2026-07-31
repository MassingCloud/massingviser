"""``massingviser.plugins.shell`` -- the bookkeeping half of an application shell.

``massingifc``'s ``ui-shell`` ships layout, notifications, progress, a command palette and a status
bar, and leaves *rendering* to the host. That split survives here unchanged, and it is the reason
MassingViser can have a viser front end without the shell state knowing what viser is: the viewer
reads these records and draws them, and a desktop shell would read the same records.

Nothing in this module imports a rendering library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ...kernel import (
    CapabilityToken,
    CommandDefinition,
    KernelError,
    PluginContext,
    Result,
    create_capability_token,
    err,
    ok,
)
from ...schema import Id, IsoTimestamp
from ...sdk import (
    Clock,
    IdFactory,
    RecordStore,
    SequentialIdFactory,
    SystemClock,
    create_record_store,
    define_plugin,
)

PLUGIN_ID = "massingviser.ui-shell"
PLUGIN_VERSION = "0.1.0"

Severity = Literal["info", "success", "warning", "error"]
Placement = Literal["left", "right", "bottom", "modal"]


@dataclass(frozen=True)
class PanelState:
    id: Id
    title: str
    placement: Placement = "left"
    open: bool = False
    order: int = 0
    #: Host-interpreted size hint. The shell records the number; what a "unit" is is the host's
    #: business.
    size: float | None = None


@dataclass(frozen=True)
class Notification:
    id: Id
    message: str
    severity: Severity = "info"
    created_at: IsoTimestamp = ""
    #: Seconds after which a host may auto-dismiss. ``None`` means it stays until acknowledged --
    #: which is the right default for an error.
    ttl: float | None = None
    dismissed: bool = False
    #: Command a host may offer as the notification's action.
    action_command: str | None = None


@dataclass(frozen=True)
class ProgressTask:
    id: Id
    label: str
    #: 0..1, or ``None`` for indeterminate. Distinguished because a spinner and a bar say
    #: different things, and faking a percentage for work of unknown length is a lie.
    fraction: float | None = None
    started_at: IsoTimestamp = ""
    finished_at: IsoTimestamp | None = None
    cancelled: bool = False


@dataclass(frozen=True)
class PaletteEntry:
    command_id: str
    title: str
    subtitle: str | None = None
    #: Whether the command declares a permission. Shown so a user is not offered an action that
    #: will be refused.
    permission: str | None = None


@dataclass(frozen=True)
class StatusItem:
    id: Id
    text: str
    severity: Severity = "info"
    order: int = 0
    tooltip: str | None = None


@runtime_checkable
class ShellService(Protocol):
    # layout
    def panels(self, placement: Placement | None = None) -> tuple[PanelState, ...]: ...
    def sync_panels(self) -> tuple[PanelState, ...]: ...
    def set_panel_open(self, panel_id: Id, open: bool) -> Result[PanelState, KernelError]: ...
    def toggle_panel(self, panel_id: Id) -> Result[PanelState, KernelError]: ...

    # notifications
    def notify(self, message: str, **options: Any) -> Notification: ...
    def dismiss(self, notification_id: Id) -> Result[None, KernelError]: ...
    def notifications(self, *, include_dismissed: bool = False) -> tuple[Notification, ...]: ...

    # progress
    def begin(self, label: str, fraction: float | None = None) -> ProgressTask: ...
    def report(self, task_id: Id, fraction: float) -> Result[ProgressTask, KernelError]: ...
    def finish(
        self, task_id: Id, *, cancelled: bool = False
    ) -> Result[ProgressTask, KernelError]: ...
    def running(self) -> tuple[ProgressTask, ...]: ...

    # palette and status bar
    def palette(self, query: str = "") -> tuple[PaletteEntry, ...]: ...
    def set_status(self, item: StatusItem) -> None: ...
    def status(self) -> tuple[StatusItem, ...]: ...


ShellToken: CapabilityToken[ShellService] = create_capability_token("shell.service")


class SHELL_COMMANDS:
    toggle_panel = "shell.panel.toggle"
    notify = "shell.notify"
    dismiss = "shell.notification.dismiss"


class SHELL_EVENTS:
    layout_changed = "shell.layout.changed"
    notified = "shell.notified"
    progress_changed = "shell.progress.changed"
    status_changed = "shell.status.changed"


class ShellServiceImpl:
    __slots__ = ("_context", "_clock", "_ids", "_panels", "_notifications", "_tasks", "_status")

    def __init__(
        self,
        context: PluginContext,
        clock: Clock,
        ids: IdFactory,
        panels: RecordStore[PanelState],
        notifications: RecordStore[Notification],
        tasks: RecordStore[ProgressTask],
        status: RecordStore[StatusItem],
    ) -> None:
        self._context = context
        self._clock = clock
        self._ids = ids
        self._panels = panels
        self._notifications = notifications
        self._tasks = tasks
        self._status = status

    # -- layout -------------------------------------------------------------------------------

    def sync_panels(self) -> tuple[PanelState, ...]:
        """Mirror the kernel's UI registry into shell state.

        The registry says which panels *exist*; this says which are *open*. Keeping them apart is
        what lets a user's layout survive a plugin being deactivated and reactivated.
        """
        known = {panel.id for panel in self._panels.all()}
        for contribution in self._context.ui.by_point("panel"):
            if contribution.id in known:
                continue
            self._panels.add(
                PanelState(
                    id=contribution.id,
                    title=contribution.title or contribution.id,
                    placement=contribution.placement or "left",  # type: ignore[arg-type]
                    order=contribution.order,
                )
            )
        return self.panels()

    def panels(self, placement: Placement | None = None) -> tuple[PanelState, ...]:
        return tuple(
            sorted(
                (
                    panel
                    for panel in self._panels.all()
                    if placement is None or panel.placement == placement
                ),
                key=lambda panel: (panel.order, panel.id),
            )
        )

    def set_panel_open(self, panel_id: Id, open: bool) -> Result[PanelState, KernelError]:
        updated = self._panels.update(panel_id, {"open": open})
        if updated is None:
            return err(
                KernelError("COMMAND_FAILED", f'No panel "{panel_id}".', {"panelId": panel_id})
            )
        self._context.events.emit(SHELL_EVENTS.layout_changed, {"panel": updated})
        return ok(updated)

    def toggle_panel(self, panel_id: Id) -> Result[PanelState, KernelError]:
        panel = self._panels.get(panel_id)
        if panel is None:
            return err(
                KernelError("COMMAND_FAILED", f'No panel "{panel_id}".', {"panelId": panel_id})
            )
        return self.set_panel_open(panel_id, not panel.open)

    # -- notifications ------------------------------------------------------------------------

    def notify(self, message: str, **options: Any) -> Notification:
        severity: Severity = options.get("severity", "info")
        record = Notification(
            id=self._ids.next("note"),
            message=message,
            severity=severity,
            created_at=self._clock.iso(),
            # An error with a timeout is an error nobody reads. Only non-errors expire by default.
            ttl=options.get("ttl", None if severity == "error" else 6.0),
            action_command=options.get("action_command"),
        )
        self._notifications.add(record)
        self._context.events.emit(SHELL_EVENTS.notified, {"notification": record})
        return record

    def dismiss(self, notification_id: Id) -> Result[None, KernelError]:
        updated = self._notifications.update(notification_id, {"dismissed": True})
        return (
            ok(None)
            if updated
            else err(KernelError("COMMAND_FAILED", f'No notification "{notification_id}".', {}))
        )

    def notifications(self, *, include_dismissed: bool = False) -> tuple[Notification, ...]:
        return self._notifications.query(lambda note: include_dismissed or not note.dismissed)

    # -- progress -----------------------------------------------------------------------------

    def begin(self, label: str, fraction: float | None = None) -> ProgressTask:
        task = ProgressTask(
            id=self._ids.next("task"),
            label=label,
            fraction=fraction,
            started_at=self._clock.iso(),
        )
        self._tasks.add(task)
        self._context.events.emit(SHELL_EVENTS.progress_changed, {"task": task})
        return task

    def report(self, task_id: Id, fraction: float) -> Result[ProgressTask, KernelError]:
        # Clamped rather than rejected: a caller reporting 1.02 because of rounding should not have
        # its long-running job fail on the progress bar.
        clamped = max(0.0, min(1.0, fraction))
        updated = self._tasks.update(task_id, {"fraction": clamped})
        if updated is None:
            return err(KernelError("COMMAND_FAILED", f'No task "{task_id}".', {}))
        self._context.events.emit(SHELL_EVENTS.progress_changed, {"task": updated})
        return ok(updated)

    def finish(self, task_id: Id, *, cancelled: bool = False) -> Result[ProgressTask, KernelError]:
        updated = self._tasks.update(
            task_id,
            {
                "finished_at": self._clock.iso(),
                "cancelled": cancelled,
                **({} if cancelled else {"fraction": 1.0}),
            },
        )
        if updated is None:
            return err(KernelError("COMMAND_FAILED", f'No task "{task_id}".', {}))
        self._context.events.emit(SHELL_EVENTS.progress_changed, {"task": updated})
        return ok(updated)

    def running(self) -> tuple[ProgressTask, ...]:
        return self._tasks.query(lambda task: task.finished_at is None)

    # -- palette and status -------------------------------------------------------------------

    def palette(self, query: str = "") -> tuple[PaletteEntry, ...]:
        """Every registered command, filtered.

        Built from the command bus rather than from a curated list, so a newly installed plugin's
        actions are reachable the moment it activates.
        """
        needle = query.strip().lower()
        entries = [
            PaletteEntry(
                command_id=info.id,
                title=info.title or info.id,
                subtitle=info.description,
                permission=info.permission,
            )
            for info in self._context.commands.list()
        ]
        if needle:
            entries = [
                entry
                for entry in entries
                if needle in entry.title.lower() or needle in entry.command_id.lower()
            ]
        return tuple(sorted(entries, key=lambda entry: entry.title))

    def set_status(self, item: StatusItem) -> None:
        self._status.remove_where(lambda existing: existing.id == item.id)
        self._status.add(item)
        self._context.events.emit(SHELL_EVENTS.status_changed, {"item": item})

    def status(self) -> tuple[StatusItem, ...]:
        return tuple(sorted(self._status.all(), key=lambda item: (item.order, item.id)))


def create_shell_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        service = ShellServiceImpl(
            context,
            resolved_clock,
            resolved_ids,
            create_record_store(context.state, "panels"),
            create_record_store(context.state, "notifications"),
            create_record_store(context.state, "tasks"),
            create_record_store(context.state, "status"),
        )
        context.capabilities.provide(ShellToken, service, version=PLUGIN_VERSION)

        def toggle(params: Mapping[str, Any], _ctx: Any) -> Any:
            service.sync_panels()
            result = service.toggle_panel(params["panel_id"])
            if not result.ok:
                raise result.error
            return result.value

        def notify(params: Mapping[str, Any], _ctx: Any) -> Any:
            return service.notify(
                params["message"], **{k: v for k, v in params.items() if k != "message"}
            )

        def dismiss(params: Mapping[str, Any], _ctx: Any) -> Any:
            result = service.dismiss(params["notification_id"])
            if not result.ok:
                raise result.error
            return None

        for command in (
            CommandDefinition(id=SHELL_COMMANDS.toggle_panel, title="Toggle panel", handler=toggle),
            CommandDefinition(id=SHELL_COMMANDS.notify, title="Notify", handler=notify),
            CommandDefinition(
                id=SHELL_COMMANDS.dismiss, title="Dismiss notification", handler=dismiss
            ),
        ):
            context.commands.register(command)

        context.logger.info("Shell bookkeeping ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="UI shell",
        description="Headless shell state: layout, notifications, progress, palette, status bar.",
        activate=activate,
    )


shell_plugin = create_shell_plugin()
