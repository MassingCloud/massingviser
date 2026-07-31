"""Durable storage adapters.

The key-encoding tests are the ones that matter most. An encoding that is not exactly reversible
breaks the ``::backup::`` marker the persistence engine relies on, and the symptom is not an error
-- it is `list_backups()` quietly returning nothing while backups accumulate in `keys()` as though
they were records.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from massingviser.kernel import PersistenceEngine
from massingviser.storage import (
    FileSystemStorageAdapter,
    KeyEscapeError,
    SqliteStorageAdapter,
    decode_key,
    encode_key,
    resolve_key_path,
)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture()
def adapters(root: Path):
    """Both adapters, so every behavioural test runs against both."""
    sqlite_adapter = SqliteStorageAdapter(":memory:")
    try:
        yield [FileSystemStorageAdapter(root), sqlite_adapter]
    finally:
        sqlite_adapter.close()


# ---------------------------------------------------------------------------------------------
# Key encoding
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "project",
        "container:p1:manifest",
        "container:p1:entry:models/tower.frag",
        "p1::backup::2026-07-27T03:47:24.810Z-0",
        "markup:settings",
        "..cache",
        ".hidden",
        "with space",
        "unicode:Tour Eiffel — café",
        "emoji:🏗",
        'awkward<>:"|?*chars',
        "trailing.",
        "nul",
        "CON",
        "com1",
        "%",
        "%25",
        "",
    ],
)
def test_key_encoding_round_trips_exactly(key):
    """The property everything else depends on."""
    assert decode_key(encode_key(key)) == key


@pytest.mark.parametrize(
    "key", ["../../etc/passwd", "a/../../b", "..\\..\\windows", "x:y", "/etc/passwd"]
)
def test_encoding_never_emits_a_path_separator(key):
    """Traversal is structurally impossible, not merely checked for."""
    encoded = encode_key(key)
    assert "/" not in encoded
    assert "\\" not in encoded
    assert os.sep not in encoded


def test_distinct_keys_stay_distinct():
    # A sanitiser that maps every illegal character to "_" collides these onto one file and
    # silently loses three of the four.
    assert len({encode_key(k) for k in ["a<b", "a>b", "a|b", "a_b"]}) == 4


@pytest.mark.parametrize("reserved", ["con", "PRN", "aux", "nul", "com1", "lpt9"])
def test_windows_reserved_device_names_are_escaped(reserved):
    """`nul.json` is not a file Windows will create, and this project runs on Windows."""
    assert encode_key(reserved) != reserved
    assert decode_key(encode_key(reserved)) == reserved


def test_a_name_that_merely_contains_a_device_name_is_untouched():
    assert encode_key("console") == "console"


def test_a_trailing_dot_is_escaped_because_windows_strips_it():
    assert encode_key("report.") == "report%2E"
    assert encode_key("report.") != encode_key("report")


@pytest.mark.parametrize(
    "key", ["../outside", "../../etc/passwd", "a/../../b", "..\\..\\windows", "/etc/passwd"]
)
def test_paths_are_contained_inside_the_root(key, tmp_path):
    resolved = resolve_key_path(tmp_path, key, ".json")
    assert resolved.startswith(os.path.abspath(tmp_path) + os.sep)


def test_a_legitimate_key_beginning_with_dots_is_allowed(tmp_path):
    # The `relative().startswith("..")` idiom rejects this: its relative path is the ordinary
    # filename `..cache.json`.
    assert resolve_key_path(tmp_path, "..cache", ".json")
    assert resolve_key_path(tmp_path, "..config:v1", ".json")


def test_a_sibling_directory_sharing_the_prefix_is_refused(tmp_path):
    root = tmp_path / "data"
    resolved = resolve_key_path(root, "x", ".json")
    assert not resolved.startswith(os.path.abspath(tmp_path / "data-evil"))


def test_the_containment_guard_still_fires_if_an_extension_escapes(tmp_path):
    """Defence in depth: unreachable given the encoding, but the guard must work."""
    with pytest.raises(KeyEscapeError):
        resolve_key_path(tmp_path, "x", f"{os.sep}..{os.sep}..{os.sep}evil.json")


# ---------------------------------------------------------------------------------------------
# Round trip, both adapters
# ---------------------------------------------------------------------------------------------


async def test_values_round_trip(adapters):
    for adapter in adapters:
        await adapter.put("project", {"name": "Tower", "storeys": 12})
        assert await adapter.get("project") == {"name": "Tower", "storeys": 12}


async def test_a_missing_key_is_absent_not_an_error(adapters):
    for adapter in adapters:
        assert await adapter.get("never-written") is None


async def test_a_key_containing_a_slash_round_trips(adapters):
    for adapter in adapters:
        await adapter.put("container:p1:entry:models/tower", {"a": 1})
        assert await adapter.keys() == ["container:p1:entry:models/tower"]


async def test_binary_payloads_round_trip(adapters):
    """Containers hold model payloads, and JSON cannot represent bytes."""
    payload = bytes([0, 1, 2, 250, 255])
    for adapter in adapters:
        await adapter.put("model", {"name": "Tower", "payload": payload})
        loaded = await adapter.get("model")
        assert isinstance(loaded["payload"], bytes)
        assert loaded["payload"] == payload


async def test_delete_tolerates_a_missing_key(adapters):
    for adapter in adapters:
        await adapter.put("gone", {"x": 1})
        await adapter.delete("gone")
        assert await adapter.get("gone") is None
        await adapter.delete("gone")  # must not raise


async def test_keys_are_sorted_and_prefix_filtered(adapters):
    for adapter in adapters:
        await adapter.put("markup:b", {})
        await adapter.put("markup:a", {})
        await adapter.put("massing:a", {})
        assert await adapter.keys() == ["markup:a", "markup:b", "massing:a"]
        assert await adapter.keys("markup") == ["markup:a", "markup:b"]


async def test_a_prefix_containing_sql_wildcards_is_not_interpreted():
    """`LIKE 'a%_%'` would match far too much; the range scan does not interpret anything."""
    adapter = SqliteStorageAdapter(":memory:")
    try:
        await adapter.put("a%b", {})
        await adapter.put("axb", {})
        await adapter.put("a%c", {})
        assert await adapter.keys("a%") == ["a%b", "a%c"]
    finally:
        adapter.close()


async def test_writes_are_atomic_and_leave_no_temporary_files(root):
    adapter = FileSystemStorageAdapter(root)
    await adapter.put("project", {"v": 1})
    await adapter.put("project", {"v": 2})
    assert await adapter.get("project") == {"v": 2}
    assert [p.suffix for p in root.iterdir()] == [".json"]
    assert await adapter.keys() == ["project"]


async def test_concurrent_writes_to_one_key_resolve_to_a_last_write(root):
    """On Windows a replace onto a file another replace is touching fails outright."""
    import asyncio

    adapter = FileSystemStorageAdapter(root)
    await asyncio.gather(*(adapter.put("hot", {"n": n}) for n in range(20)))
    value = await adapter.get("hot")
    assert value is not None and 0 <= value["n"] < 20  # one of them won; none raised


async def test_keys_on_an_unwritten_root_is_empty_not_an_error(root):
    adapter = FileSystemStorageAdapter(root)
    assert await adapter.keys() == []
    assert await adapter.exists() is False


# ---------------------------------------------------------------------------------------------
# The reason the encoding has to be reversible
# ---------------------------------------------------------------------------------------------


async def test_backups_survive_a_real_adapter(root):
    """The `::backup::` marker must come back byte-identical, or pruning silently stops working."""
    for adapter in (FileSystemStorageAdapter(root), SqliteStorageAdapter(":memory:")):
        engine = PersistenceEngine(adapter=adapter, max_backups=3)
        await engine.save("project", "massingifc.project", {"v": 1})
        for version in range(2, 7):
            await engine.save("project", "massingifc.project", {"v": version}, backup=True)

        backups = await engine.list_backups("project")
        assert len(backups) == 3  # pruned to the limit, which requires listing to work
        # Backups must not leak into the record listing.
        assert await engine.keys() == ["project"]

        rolled = await engine.rollback("project", backups[-1].id)
        assert rolled.ok


async def test_a_project_survives_a_restart(root):
    """The whole point of the package."""
    from massingviser import build_kernel
    from massingviser.plugins.massing import MASSING_COMMANDS, MassingToken

    from massingviser.app import filesystem_storage

    # Records are frozen dataclasses, so the adapter needs the schema codec. Wiring that is the
    # composition root's job -- a bare adapter handles values and bytes, nothing richer.
    kernel = build_kernel(storage=filesystem_storage(root))
    await kernel.start()
    profile = (
        await kernel.commands.execute(
            MASSING_COMMANDS.sketch_profile,
            {"points": [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)]},
        )
    ).value
    mass = (
        await kernel.commands.execute(
            MASSING_COMMANDS.create_mass,
            {"name": "Block A", "profile_id": profile, "story_count": 11},
        )
    ).value
    saved = await kernel.persistence.save(
        "session", "massingifc.session", kernel.state.snapshot()
    )
    assert saved.ok
    await kernel.stop()

    # A different process would see exactly this: a fresh kernel over the same directory.
    revived = build_kernel(storage=filesystem_storage(root))
    document = (await revived.persistence.load("session")).value
    assert document is not None
    revived.state.restore(document.data)
    await revived.start()

    restored = revived.capabilities.get(MassingToken).get(mass.id)
    assert restored is not None and restored.story_count == 11
    await revived.stop()
