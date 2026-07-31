"""UI shell -- the bookkeeping half: layout, notifications, progress, palette, status bar."""

from .plugin import (
    PLUGIN_ID,
    SHELL_COMMANDS,
    SHELL_EVENTS,
    Notification,
    PaletteEntry,
    PanelState,
    Placement,
    ProgressTask,
    Severity,
    ShellService,
    ShellToken,
    StatusItem,
    create_shell_plugin,
    shell_plugin,
)

__all__ = [
    "PLUGIN_ID",
    "SHELL_COMMANDS",
    "SHELL_EVENTS",
    "Notification",
    "PaletteEntry",
    "PanelState",
    "Placement",
    "ProgressTask",
    "Severity",
    "ShellService",
    "ShellToken",
    "StatusItem",
    "create_shell_plugin",
    "shell_plugin",
]
