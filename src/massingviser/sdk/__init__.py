"""``massingviser.sdk`` -- what a plugin author imports.

Everything registered through a ``PluginContext`` is released automatically when the plugin
deactivates: commands, panels, event subscriptions, capabilities, state slices and services.
"""

from .define_plugin import define_plugin
from .record_store import Identified, RecordStore, RecordStoreHost, create_record_store
from .runtime import (
    DEFAULT_CLOCK,
    DEFAULT_IDS,
    Clock,
    FixedClock,
    IdFactory,
    SequentialIdFactory,
    SystemClock,
    UuidIdFactory,
)
from .testing import TestHarness, create_test_harness

__all__ = [
    "DEFAULT_CLOCK",
    "DEFAULT_IDS",
    "Clock",
    "FixedClock",
    "IdFactory",
    "Identified",
    "RecordStore",
    "RecordStoreHost",
    "SequentialIdFactory",
    "SystemClock",
    "TestHarness",
    "UuidIdFactory",
    "create_record_store",
    "create_test_harness",
    "define_plugin",
]
