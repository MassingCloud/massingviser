from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass, replace
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from ..kernel import Slice

T = TypeVar("T")


@runtime_checkable
class Identified(Protocol):
    id: str


@runtime_checkable
class RecordStoreHost(Protocol):
    def define_slice(self, name: str, initial: Any) -> Slice[Any]: ...


def _patched(record: Any, changes: dict[str, Any]) -> Any:
    """Shallow-merge ``changes`` onto ``record``, preserving its id.

    Frozen dataclasses are the record shape throughout the platform, so this goes through
    ``dataclasses.replace`` -- which validates field names rather than silently attaching a
    misspelled attribute the way a dict update would.
    """
    if is_dataclass(record):
        known = {f.name for f in dataclass_fields(record)}
        unknown = set(changes) - known
        if unknown:
            raise TypeError(
                f"{type(record).__name__} has no field(s) {sorted(unknown)}; "
                "a typo here would silently never apply."
            )
        merged = {key: value for key, value in changes.items() if key != "id"}
        return replace(record, **merged)
    merged = {**record, **changes, "id": record["id"]}
    return merged


class RecordStore(Generic[T]):
    """A keyed collection held in a kernel state slice.

    Nearly every capability family owns one or more collections of records with the same access
    pattern: get by id, list, filter, insert, patch, delete. Writing that nine times would be nine
    chances to get the immutability wrong -- and the state store's change detection depends on
    writes producing new objects, so getting it wrong means subscribers silently stop updating.

    Reads are O(1) via an index rebuilt on write, because list-scanning showed up first on hot
    paths like resolving markup anchors across thousands of records.
    """

    __slots__ = ("slice", "_index", "_indexed_for")

    def __init__(self, slice_: Slice[tuple[T, ...]]) -> None:
        self.slice = slice_
        self._index: dict[str, T] = {}
        self._indexed_for: Any = None

    def _ensure_index(self) -> dict[str, T]:
        current = self.slice.get()
        # Rebuilt lazily and only when the tuple identity changed -- restoring a persisted project
        # replaces the slice wholesale without going through `add`, so tracking writes is not
        # enough.
        if self._indexed_for is not current:
            self._index = {record.id: record for record in current}
            self._indexed_for = current
        return self._index

    def all(self) -> tuple[T, ...]:
        return self.slice.get()

    def get(self, id: str) -> T | None:
        return self._ensure_index().get(id)

    def has(self, id: str) -> bool:
        return id in self._ensure_index()

    def add(self, record: T) -> T:
        self.slice.update(lambda current: (*current, record))
        return record

    def add_many(self, records: Sequence[T]) -> Sequence[T]:
        if not records:
            return records
        self.slice.update(lambda current: (*current, *records))
        return records

    def update(self, id: str, changes: dict[str, Any]) -> T | None:
        """Shallow-merge ``changes``. Returns ``None`` if the id is unknown."""
        if id not in self._ensure_index():
            return None
        updated: list[T] = []

        def _apply(current: tuple[T, ...]) -> tuple[T, ...]:
            out = []
            for record in current:
                if record.id != id:
                    out.append(record)
                    continue
                patched = _patched(record, changes)
                updated.append(patched)
                out.append(patched)
            return tuple(out)

        self.slice.update(_apply)
        return updated[0] if updated else None

    def replace(self, record: T) -> T | None:
        """Replace wholesale. Returns ``None`` if the id is unknown."""
        if record.id not in self._ensure_index():
            return None
        self.slice.update(
            lambda current: tuple(record if item.id == record.id else item for item in current)
        )
        return record

    def remove(self, id: str) -> bool:
        if id not in self._ensure_index():
            return False
        self.slice.update(lambda current: tuple(r for r in current if r.id != id))
        return True

    def remove_where(self, predicate: Callable[[T], bool]) -> int:
        current = self.slice.get()
        kept = tuple(record for record in current if not predicate(record))
        removed = len(current) - len(kept)
        if removed > 0:
            self.slice.set(kept)
        return removed

    def query(self, predicate: Callable[[T], bool] | None = None) -> tuple[T, ...]:
        if predicate is None:
            return self.slice.get()
        return tuple(record for record in self.slice.get() if predicate(record))

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        for record in self.slice.get():
            if predicate(record):
                return record
        return None

    def count(self) -> int:
        return len(self.slice.get())

    def clear(self) -> None:
        if self.slice.get():
            self.slice.set(())


def create_record_store(
    host: RecordStoreHost, name: str, initial: Sequence[T] = ()
) -> RecordStore[T]:
    return RecordStore(host.define_slice(name, tuple(initial)))
