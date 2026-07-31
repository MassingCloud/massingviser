"""Filesystem-backed storage.

``MemoryStorageAdapter`` makes the kernel testable; this makes it *persist*. Nothing in the platform
survives a restart without it, which leaves the whole versioning and migration apparatus
theoretical.

The key-to-filename mapping is **percent-encoded and exactly reversible**. The tempting alternative
-- map ``:`` to a path separator for a browsable tree, strip illegal characters -- is lossy in both
directions: ``keys()`` returns keys that were never written, and the ``::backup::`` marker
``PersistenceEngine`` relies on comes back as ``:backup:``, so ``list_backups()`` finds nothing,
pruning never runs, and backups leak into ``keys()`` as though they were records. Reversibility is
worth more than a readable directory listing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

#: Characters kept verbatim. Everything else is percent-encoded, so the mapping is total.
_SAFE = re.compile(r"[A-Za-z0-9._-]")

#: Windows refuses these as filenames, with or without an extension on older releases.
#:
#: Source: "Naming Files, Paths, and Namespaces" (Microsoft Learn). A key of ``nul`` is entirely
#: legal in this platform's own namespace, so it is encoded rather than rejected.
_RESERVED_DEVICE_NAME = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])$", re.IGNORECASE)

#: JSON cannot represent binary, and containers hold model payloads.
#:
#: Encoding as base64 under a tagged wrapper keeps a project a single readable JSON tree rather than
#: introducing a second on-disk format for blobs. It costs ~33% in size, which is the right trade
#: for a desktop project file and the wrong one for a streaming server -- hence a documented
#: boundary rather than a silent default.
BYTES_TAG = "$massingifc:bytes"


class KeyEscapeError(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(f'Storage key "{key}" resolves outside the storage root.')
        self.key = key


def _percent(byte: int) -> str:
    return f"%{byte:02X}"


def encode_key(key: str) -> str:
    """Encode a key as a single filename.

    Deliberately produces one flat file per key rather than a directory tree. Because no path
    separator can survive the encoding, directory traversal is *structurally impossible* rather than
    merely checked for -- the containment check below is then defence in depth rather than the only
    line. The cost is a flat directory and a key-length ceiling (a filename is capped at 255 bytes
    on most filesystems, and percent-encoding inflates non-ASCII); a key long enough to exceed that
    surfaces as an OS error rather than being silently truncated.
    """
    encoded_parts: list[str] = []
    for character in key:
        if _SAFE.fullmatch(character):
            encoded_parts.append(character)
            continue
        encoded_parts.extend(_percent(byte) for byte in character.encode("utf-8"))
    encoded = "".join(encoded_parts)

    # An empty key would produce an empty filename, which is not a filename.
    if encoded == "":
        return "%"

    # Windows strips a trailing dot or space, which would make two distinct keys collide.
    if encoded.endswith("."):
        encoded = f"{encoded[:-1]}%2E"

    # A reserved device name is only reserved as the *stem*, so encoding its first byte is enough.
    stem = encoded.split(".")[0]
    if _RESERVED_DEVICE_NAME.fullmatch(stem):
        encoded = _percent(ord(encoded[0])) + encoded[1:]
    return encoded


def decode_key(encoded: str) -> str:
    """Exact inverse of :func:`encode_key`."""
    if encoded == "%":
        return ""
    out = bytearray()
    index = 0
    while index < len(encoded):
        if encoded[index] == "%":
            out.append(int(encoded[index + 1 : index + 3], 16))
            index += 3
            continue
        # Every unencoded character is ASCII by construction, so one char is one byte.
        out.append(ord(encoded[index]))
        index += 1
    return out.decode("utf-8")


def resolve_key_path(root: str | os.PathLike[str], key: str, extension: str) -> str:
    """Map a storage key to a path beneath the root.

    Containment is checked by resolving and comparing against ``root + sep``, not by testing whether
    the relative path starts with ``..``. The latter is a common idiom but rejects a legitimate key
    such as ``..cache``, whose relative path is the ordinary filename ``..cache.json``. Comparing
    against ``root + sep`` also refuses the sibling-prefix case (``/data-evil`` against a ``/data``
    root).
    """
    root_path = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_path, f"{encode_key(key)}{extension}"))
    if not target.startswith(root_path + os.sep):
        raise KeyEscapeError(key)
    return target


def _encode_bytes(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {BYTES_TAG: base64.b64encode(bytes(value)).decode("ascii")}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _decode_bytes(mapping: dict[str, Any]) -> Any:
    tagged = mapping.get(BYTES_TAG)
    if isinstance(tagged, str) and len(mapping) == 1:
        return base64.b64decode(tagged)
    return mapping


def compose_default(extra: Any) -> Any:
    """Chain a caller's encoder behind the built-in bytes handling.

    The adapters stay generic -- they know about bytes and nothing else. Anything richer (records,
    for instance) is wired in by the composition root, which is the only layer entitled to know
    about both storage and the schema.
    """
    if extra is None:
        return _encode_bytes

    def _default(value: Any) -> Any:
        try:
            return _encode_bytes(value)
        except TypeError:
            return extra(value)

    return _default


def compose_object_hook(extra: Any) -> Any:
    if extra is None:
        return _decode_bytes

    def _hook(mapping: dict[str, Any]) -> Any:
        decoded = _decode_bytes(mapping)
        return decoded if decoded is not mapping else extra(mapping)

    return _hook


class FileSystemStorageAdapter:
    """A ``StorageAdapter`` over a directory of JSON files."""

    __slots__ = ("_root", "_extension", "_locks", "_default", "_object_hook")

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        extension: str = ".json",
        default: Any = None,
        object_hook: Any = None,
    ) -> None:
        self._root = os.path.abspath(root)
        self._extension = extension
        self._default = compose_default(default)
        self._object_hook = compose_object_hook(object_hook)
        # One lock per destination path.
        #
        # Each individual write is atomic, but two of them are not safe to overlap: on Windows a
        # replace onto a destination another replace is touching fails outright. Serialising per key
        # turns concurrent `put`s on one key into a deterministic last-write-wins instead of a
        # spurious failure. In-process only -- two processes writing one file still race, which is
        # why a container is owned by one host.
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def root(self) -> str:
        return self._root

    def _lock_for(self, path: str) -> asyncio.Lock:
        lock = self._locks.get(path)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[path] = lock
        return lock

    async def get(self, key: str) -> Any | None:
        path = resolve_key_path(self._root, key, self._extension)

        def _read() -> Any | None:
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle, object_hook=self._object_hook)
            except FileNotFoundError:
                # A missing key is a normal outcome -- a first run -- not a failure. A corrupt file
                # is not, and its parse error propagates: conflating the two hides data loss behind
                # an empty state.
                return None

        return await asyncio.to_thread(_read)

    async def put(self, key: str, value: Any) -> None:
        """Write atomically: to a temporary file, then replace.

        A crash or a full disk part-way through a direct write leaves a truncated JSON file, which
        is indistinguishable from corruption on the next load. ``os.replace`` is atomic on both
        POSIX and NTFS, so a reader sees either the old file or the whole new one.
        """
        path = resolve_key_path(self._root, key, self._extension)

        def _write() -> None:
            os.makedirs(self._root, exist_ok=True)
            # A random suffix, not pid+timestamp: two writes within the same millisecond in one
            # process would otherwise pick the same temporary name.
            temporary = f"{path}.{uuid.uuid4().hex}.tmp"
            try:
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, default=self._default)
                # `os.replace`, not `os.rename`: rename onto an existing file raises on Windows.
                os.replace(temporary, path)
            except BaseException:
                try:
                    os.remove(temporary)
                except OSError:
                    pass
                raise

        async with self._lock_for(path):
            await asyncio.to_thread(_write)

    async def delete(self, key: str) -> None:
        path = resolve_key_path(self._root, key, self._extension)

        def _remove() -> None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass  # deleting what is not there is not a failure

        await asyncio.to_thread(_remove)

    async def keys(self, prefix: str = "") -> list[str]:
        def _list() -> list[str]:
            try:
                names = os.listdir(self._root)
            except FileNotFoundError:
                return []  # nothing written yet
            found = []
            for name in names:
                # A crash between write and replace can leave `key.json.<uuid>.tmp` behind. It ends
                # in `.tmp`, so this one check excludes it -- and, unlike a substring test, still
                # lists a legitimate key such as `a.tmp`, whose file is `a.tmp.json`.
                if not name.endswith(self._extension):
                    continue
                if not os.path.isfile(os.path.join(self._root, name)):
                    continue
                found.append(decode_key(name[: -len(self._extension)]))
            return sorted(key for key in found if key.startswith(prefix))

        return await asyncio.to_thread(_list)

    async def exists(self) -> bool:
        """Whether the root exists yet. Useful for a host deciding between open and create."""
        return await asyncio.to_thread(os.path.isdir, self._root)
