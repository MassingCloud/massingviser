"""Content-addressed objects.

The model is Speckle's, because Speckle solved this problem properly: decompose a model into atomic
objects, hash each one by its content, and store children **by reference** rather than by value.
The identity of an object *is* its content, which buys three things at once:

- **Deduplication.** Two versions that share a wall store that wall once. So do two different
  models that happen to contain the same object.
- **Diffing without traversal.** "What changed between v3 and v7" is a set difference over ids, not
  a walk of two trees.
- **Integrity.** An id that does not match its content is corruption, and it is detectable.

The id is ``sha256(canonical json)[:32]`` -- 128 bits, hex. Truncated because 32 characters is what
travels comfortably in a URL, a filename and a log line, and 128 bits of SHA-256 is far past the
point where collision is a practical concern for a project's object count.

Nothing here imports anything outside the standard library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

#: Truncation length of the hex digest. 128 bits.
ID_LENGTH = 32

#: Marker for a child stored separately and referenced.
REFERENCE_TYPE = "reference"

#: Property-name prefix that detaches a member, following Speckle's ``@`` convention. A member
#: named ``@geometry`` is stored as its own object; ``geometry`` is stored inline.
DETACH_PREFIX = "@"

#: Default items per chunk when a detached list is split. Large enough that a wall's vertex buffer
#: is a handful of objects, small enough that one changed vertex does not re-upload a megabyte.
DEFAULT_CHUNK_SIZE = 5000


class VcsError(Exception):
    pass


@dataclass(frozen=True)
class Reference:
    """A pointer to an object held separately."""

    referenced_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"referencedId": self.referenced_id, "type": REFERENCE_TYPE}

    @staticmethod
    def is_reference(value: Any) -> bool:
        return isinstance(value, Mapping) and value.get("type") == REFERENCE_TYPE


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace.

    Determinism is the whole contract. Two processes on two machines must produce byte-identical
    text for equal content, or the id stops being an identity and becomes a coincidence.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_encode)


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "speckle_type": type(value).__name__,
            **{f.name: getattr(value, f.name) for f in fields(value)},
        }
    if isinstance(value, (set, frozenset)):
        # Sets have no order, so they are sorted before hashing -- otherwise the same content
        # hashes differently depending on insertion history.
        return sorted(value, key=repr)
    if isinstance(value, bytes):
        import base64

        return {"$bytes": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"{type(value).__name__} is not serialisable into a content-addressed object")


def compute_id(payload: Mapping[str, Any]) -> str:
    """The object's identity: a truncated SHA-256 over its canonical form.

    The ``id`` member itself is excluded, because an object cannot contain its own hash.
    """
    without_id = {key: value for key, value in payload.items() if key != "id"}
    digest = hashlib.sha256(canonical_json(without_id).encode("utf-8")).hexdigest()
    return digest[:ID_LENGTH]


@dataclass
class SerialisedObject:
    id: str
    payload: dict[str, Any]
    #: Every detached descendant, and how far below this object it sits.
    #:
    #: The closure table is what makes "fetch everything this version needs" one lookup instead of
    #: a recursive walk -- which matters when a version is 400,000 objects and the walk is over a
    #: network.
    closure: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(canonical_json(self.payload))


class Serialiser:
    """Decomposes a tree into atomic, content-addressed objects.

    Traversal is depth-first and bottom-up: a child is hashed before its parent, because the
    parent's hash includes the child's id. That ordering is why an unchanged subtree keeps its id
    no matter what happens above it.
    """

    __slots__ = ("_objects", "_chunk_size")

    def __init__(self, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self._objects: dict[str, SerialisedObject] = {}
        self._chunk_size = max(1, chunk_size)

    @property
    def objects(self) -> dict[str, SerialisedObject]:
        return self._objects

    def serialise(self, value: Any) -> SerialisedObject:
        """Decompose a value into objects and return the root."""
        payload, closure = self._walk(value, depth=0)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return self._emit(payload, closure)

    # -- internals ---------------------------------------------------------------------------

    def _emit(self, payload: dict[str, Any], closure: dict[str, int]) -> SerialisedObject:
        object_id = compute_id(payload)
        stored = dict(payload)
        stored["id"] = object_id
        record = SerialisedObject(id=object_id, payload=stored, closure=dict(closure))
        # Identical content is emitted once. Re-emitting would be harmless but wasteful, and the
        # closure of the first is already correct.
        self._objects.setdefault(object_id, record)
        return record

    def _merge(self, into: dict[str, int], other: Mapping[str, int]) -> None:
        for object_id, depth in other.items():
            # Keep the *shallowest* depth: an object reachable by two paths is as close as its
            # nearest one, which is what a fetch planner wants.
            if object_id not in into or depth < into[object_id]:
                into[object_id] = depth

    def _walk(self, value: Any, *, depth: int) -> tuple[Any, dict[str, int]]:
        closure: dict[str, int] = {}

        if is_dataclass(value) and not isinstance(value, type):
            value = {
                "speckle_type": type(value).__name__,
                **{f.name: getattr(value, f.name) for f in fields(value)},
            }

        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, child in value.items():
                if key.startswith(DETACH_PREFIX):
                    out[key.lstrip(DETACH_PREFIX)] = self._detach(child, closure, depth)
                else:
                    resolved, inner = self._walk(child, depth=depth)
                    self._merge(closure, inner)
                    out[key] = resolved
            return out, closure

        if isinstance(value, (list, tuple)):
            resolved_items = []
            for item in value:
                resolved, inner = self._walk(item, depth=depth)
                self._merge(closure, inner)
                resolved_items.append(resolved)
            return resolved_items, closure

        return value, closure

    def _detach(self, child: Any, closure: dict[str, int], depth: int) -> Any:
        if isinstance(child, (list, tuple)) and len(child) > self._chunk_size:
            # Chunked *and* detached. A 200k-vertex buffer that changes in one place re-uploads one
            # chunk, not the buffer.
            references: list[dict[str, Any]] = []
            for start in range(0, len(child), self._chunk_size):
                window = list(child[start : start + self._chunk_size])
                payload, inner = self._walk({"speckle_type": "DataChunk", "data": window}, depth=depth + 1)
                record = self._emit(payload, inner)
                self._merge(closure, {record.id: 1})
                self._merge(closure, {k: v + 1 for k, v in record.closure.items()})
                references.append(Reference(record.id).to_dict())
            return references

        if isinstance(child, (list, tuple)):
            return [self._detach(item, closure, depth) for item in child]

        payload, inner = self._walk(child, depth=depth + 1)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        record = self._emit(payload, inner)
        # Depths are **relative to the object being built**, not absolute in the tree. The detached
        # child sits exactly one below; its own closure is already relative to it, so each of its
        # descendants sits one further. Folding the ambient depth in here as well double-counts,
        # and the closure then claims a grandchild is deeper than it is.
        self._merge(closure, {record.id: 1})
        self._merge(closure, {k: v + 1 for k, v in record.closure.items()})
        return Reference(record.id).to_dict()


def serialise(value: Any, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[SerialisedObject, dict[str, SerialisedObject]]:
    """Decompose a value. Returns the root object and every object produced, keyed by id."""
    serialiser = Serialiser(chunk_size=chunk_size)
    root = serialiser.serialise(value)
    return root, serialiser.objects


def deserialise(root_id: str, objects: Mapping[str, Mapping[str, Any]]) -> Any:
    """Rebuild a tree, resolving references.

    A missing referenced object raises rather than yielding a partial tree: a half-resolved model
    that looks whole is worse than one that refuses to load.
    """
    seen: set[str] = set()

    def resolve(value: Any) -> Any:
        if Reference.is_reference(value):
            referenced = value["referencedId"]
            if referenced not in objects:
                raise VcsError(
                    f'Object "{referenced}" is referenced but not present. The store is '
                    "incomplete; fetching the closure of the root would have caught this."
                )
            if referenced in seen:
                raise VcsError(f'Cycle detected through object "{referenced}".')
            seen.add(referenced)
            try:
                return resolve(dict(objects[referenced]))
            finally:
                seen.discard(referenced)
        if isinstance(value, Mapping):
            resolved = {key: resolve(item) for key, item in value.items() if key != "id"}
            if resolved.get("speckle_type") == "DataChunk":
                return resolved.get("data", [])
            return resolved
        if isinstance(value, list):
            out: list[Any] = []
            for item in value:
                resolved_item = resolve(item)
                # A chunked list flattens back into one list on the way out.
                if isinstance(resolved_item, list) and Reference.is_reference(item):
                    out.extend(resolved_item)
                else:
                    out.append(resolved_item)
            return out
        return value

    if root_id not in objects:
        raise VcsError(f'Root object "{root_id}" is not in the store.')
    return resolve(dict(objects[root_id]))


def verify(object_id: str, payload: Mapping[str, Any]) -> bool:
    """Whether an object's content still hashes to its id.

    Content addressing makes corruption detectable, so it is worth being able to detect it.
    """
    return compute_id(payload) == object_id
