"""``massingviser.storage`` -- durable ``StorageAdapter`` implementations.

The kernel ships only ``MemoryStorageAdapter``, which makes it testable and nothing more. These
make it persist. Both satisfy the same narrow four-method port, so a host swaps one for the other
without anything above noticing -- and both the per-document engine and the container service pick
up the change at once, because the kernel hands one adapter to both.

Neither imports anything outside the standard library.
"""

from .filesystem import (
    BYTES_TAG,
    FileSystemStorageAdapter,
    KeyEscapeError,
    decode_key,
    encode_key,
    resolve_key_path,
)
from .sqlite import SqliteStorageAdapter

__all__ = [
    "BYTES_TAG",
    "FileSystemStorageAdapter",
    "KeyEscapeError",
    "SqliteStorageAdapter",
    "decode_key",
    "encode_key",
    "resolve_key_path",
]
