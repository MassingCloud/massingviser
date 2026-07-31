"""SQLite-backed storage.

``massingifc`` ships an IndexedDB adapter for the browser. There is no IndexedDB in Python, but the
properties that adapter was chosen for carry over exactly: durable transactions, native binary
storage, and bounded prefix queries that do not scan the whole store. SQLite is the direct
equivalent, and it is in the standard library, so this stays dependency-free like everything else
below the viewer.

Choose this over the filesystem adapter when a project has many small records: one file per key
costs an inode and a syscall each, and ``keys()`` becomes a directory scan. SQLite indexes the key
column, so a prefix query is a range scan.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .filesystem import compose_default, compose_object_hook

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
) WITHOUT ROWID;
"""


def _prefix_upper_bound(prefix: str) -> str:
    """The exclusive upper bound for a ``LIKE``-free prefix range scan.

    ``key >= prefix AND key < bound`` uses the primary-key index; ``LIKE 'prefix%'`` does not when
    the prefix contains a wildcard, and escaping ``%`` and ``_`` in every caller's key is a rule
    somebody eventually forgets. Incrementing the last code point gives the same answer without
    the escaping problem.
    """
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


class SqliteStorageAdapter:
    """A ``StorageAdapter`` over a single SQLite file (or ``":memory:"``)."""

    __slots__ = ("_path", "_connection", "_lock", "_default", "_object_hook")

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        default: Any = None,
        object_hook: Any = None,
    ) -> None:
        self._path = str(path)
        self._default = compose_default(default)
        self._object_hook = compose_object_hook(object_hook)
        # `check_same_thread=False` because every call is marshalled onto a worker thread by
        # `asyncio.to_thread`, which does not guarantee the same one twice. The lock below is what
        # actually serialises access -- SQLite connections are not safe to use concurrently.
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        # WAL keeps a reader from blocking the writer. Harmless (and ignored) for `:memory:`.
        if self._path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")  # durability over throughput
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        self._lock = asyncio.Lock()

    def _serialise(self, value: Any) -> bytes:
        return json.dumps(value, default=self._default).encode("utf-8")

    def _deserialise(self, blob: bytes) -> Any:
        return json.loads(blob.decode("utf-8"), object_hook=self._object_hook)

    async def get(self, key: str) -> Any | None:
        def _read() -> Any | None:
            row = self._connection.execute(
                "SELECT value FROM records WHERE key = ?", (key,)
            ).fetchone()
            return None if row is None else self._deserialise(row[0])

        async with self._lock:
            return await asyncio.to_thread(_read)

    async def put(self, key: str, value: Any) -> None:
        blob = self._serialise(value)

        def _write() -> None:
            # One statement, one implicit transaction: a crash leaves either the old row or the new
            # one, never a half-written value.
            with self._connection:
                self._connection.execute(
                    "INSERT INTO records (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, blob),
                )

        async with self._lock:
            await asyncio.to_thread(_write)

    async def delete(self, key: str) -> None:
        def _remove() -> None:
            with self._connection:
                self._connection.execute("DELETE FROM records WHERE key = ?", (key,))

        async with self._lock:
            await asyncio.to_thread(_remove)

    async def keys(self, prefix: str = "") -> list[str]:
        def _list() -> list[str]:
            if not prefix:
                rows = self._connection.execute("SELECT key FROM records ORDER BY key").fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT key FROM records WHERE key >= ? AND key < ? ORDER BY key",
                    (prefix, _prefix_upper_bound(prefix)),
                ).fetchall()
            return [row[0] for row in rows]

        async with self._lock:
            return await asyncio.to_thread(_list)

    async def count(self) -> int:
        def _count() -> int:
            return self._connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]

        async with self._lock:
            return await asyncio.to_thread(_count)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteStorageAdapter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
