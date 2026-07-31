"""Commits, branches and merges over the content-addressed object store.

This is Git's shape, applied to a model rather than to text. A commit names a root object and its
parents; a branch is a moving name for a commit; history is a DAG, not a line.

The part that differs from Git is the diff. Git diffs text hunks; here the objects are already
content-addressed, so a diff between two versions is a **set difference over ids** -- which is why
comparing two 400,000-element models is instant rather than a tree walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ..kernel import KernelError, Result, StorageAdapter, err, ok
from .objects import (
    ID_LENGTH,
    SerialisedObject,
    VcsError,
    canonical_json,
    compute_id,
    deserialise,
    serialise,
    verify,
)

#: Branch a repository starts on.
DEFAULT_BRANCH = "main"


@dataclass(frozen=True)
class Commit:
    id: str
    #: The object this version points at.
    root_id: str
    message: str
    author: str
    committed_at: str
    #: Zero for the first commit, one normally, two for a merge.
    parents: tuple[str, ...] = ()
    branch: str = DEFAULT_BRANCH

    def to_dict(self) -> dict[str, Any]:
        return {
            "rootId": self.root_id,
            "message": self.message,
            "author": self.author,
            "committedAt": self.committed_at,
            "parents": list(self.parents),
            "branch": self.branch,
        }


@dataclass(frozen=True)
class Branch:
    name: str
    head: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class Tag:
    name: str
    commit_id: str
    message: str | None = None


@dataclass(frozen=True)
class Diff:
    """What changed between two versions, as sets of object ids."""

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    #: Present in both, so unchanged by definition -- content addressing means an object that
    #: appears in both versions is byte-identical in both.
    unchanged: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed

    @property
    def churn(self) -> int:
        return len(self.added) + len(self.removed)


@dataclass(frozen=True)
class MergeConflict:
    #: The path, as a dotted key trail into the object tree.
    path: str
    base: Any
    ours: Any
    theirs: Any


@dataclass(frozen=True)
class MergeResult:
    ok: bool
    commit: Commit | None = None
    conflicts: tuple[MergeConflict, ...] = ()
    #: Set when one side was already an ancestor of the other and no merge commit was needed.
    fast_forward: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


_OBJECT_PREFIX = "vcs:object:"
_COMMIT_PREFIX = "vcs:commit:"
_BRANCH_PREFIX = "vcs:branch:"
_TAG_PREFIX = "vcs:tag:"


class ObjectStore:
    """Immutable, content-addressed storage over any ``StorageAdapter``.

    Writes are idempotent by construction: the key *is* the hash, so storing an object twice is a
    no-op rather than a duplicate. That is what makes pushing a version cheap when most of it is
    already there.
    """

    __slots__ = ("_adapter",)

    def __init__(self, adapter: StorageAdapter) -> None:
        self._adapter = adapter

    async def put(self, record: SerialisedObject) -> None:
        await self._adapter.put(
            f"{_OBJECT_PREFIX}{record.id}",
            {"payload": record.payload, "closure": record.closure},
        )

    async def put_many(self, records: Iterable[SerialisedObject]) -> int:
        written = 0
        for record in records:
            if not await self.has(record.id):
                await self.put(record)
                written += 1
        return written

    async def get(self, object_id: str) -> SerialisedObject | None:
        stored = await self._adapter.get(f"{_OBJECT_PREFIX}{object_id}")
        if stored is None:
            return None
        return SerialisedObject(
            id=object_id, payload=stored["payload"], closure=dict(stored.get("closure", {}))
        )

    async def has(self, object_id: str) -> bool:
        return await self._adapter.get(f"{_OBJECT_PREFIX}{object_id}") is not None

    async def missing(self, ids: Iterable[str]) -> tuple[str, ...]:
        """Which of these are not held.

        The whole point of a push protocol: ask what is missing, send only that.
        """
        absent = []
        for object_id in ids:
            if not await self.has(object_id):
                absent.append(object_id)
        return tuple(absent)

    async def closure_of(self, object_id: str) -> dict[str, int]:
        record = await self.get(object_id)
        return dict(record.closure) if record else {}

    async def collect(self, root_id: str) -> dict[str, Mapping[str, Any]]:
        """Everything needed to rebuild a root, in one closure lookup rather than a walk."""
        root = await self.get(root_id)
        if root is None:
            raise VcsError(f'Root object "{root_id}" is not in the store.')
        objects: dict[str, Mapping[str, Any]] = {root_id: root.payload}
        for object_id in root.closure:
            child = await self.get(object_id)
            if child is None:
                raise VcsError(
                    f'Object "{object_id}" is in the closure of "{root_id}" but not in the store.'
                )
            objects[object_id] = child.payload
        return objects

    async def verify_all(self, root_id: str) -> tuple[str, ...]:
        """Ids whose content no longer hashes to them. Empty means the subtree is intact."""
        objects = await self.collect(root_id)
        return tuple(
            object_id
            for object_id, payload in objects.items()
            if not verify(object_id, payload)
        )


class Repository:
    """Versioned model storage: commit, branch, diff, merge.

    Deliberately not a Git wrapper. Git is line-oriented and its object model would have to be
    fought to hold a building; the *ideas* port cleanly, the implementation does not.
    """

    __slots__ = ("_adapter", "objects", "_now")

    def __init__(self, adapter: StorageAdapter, *, now: Any = None) -> None:
        self._adapter = adapter
        self.objects = ObjectStore(adapter)
        self._now = now or _utc_now

    # -- branches ----------------------------------------------------------------------------

    async def branches(self) -> tuple[Branch, ...]:
        keys = await self._adapter.keys(_BRANCH_PREFIX)
        found = []
        for key in keys:
            stored = await self._adapter.get(key)
            if stored is not None:
                found.append(Branch(**stored))
        return tuple(sorted(found, key=lambda branch: branch.name))

    async def branch(self, name: str) -> Branch | None:
        stored = await self._adapter.get(f"{_BRANCH_PREFIX}{name}")
        return Branch(**stored) if stored else None

    async def create_branch(
        self, name: str, *, from_commit: str | None = None, description: str | None = None
    ) -> Result[Branch, KernelError]:
        if await self.branch(name) is not None:
            return err(KernelError("COMMAND_FAILED", f'Branch "{name}" already exists.', {}))
        head = from_commit
        if head is None:
            current = await self.branch(DEFAULT_BRANCH)
            head = current.head if current else None
        if from_commit is not None and await self.commit(from_commit) is None:
            return err(KernelError("COMMAND_FAILED", f'No commit "{from_commit}".', {}))
        record = Branch(name=name, head=head, description=description)
        await self._adapter.put(f"{_BRANCH_PREFIX}{name}", record.__dict__)
        return ok(record)

    async def _set_head(self, name: str, commit_id: str) -> None:
        existing = await self.branch(name)
        record = (
            replace(existing, head=commit_id) if existing else Branch(name=name, head=commit_id)
        )
        await self._adapter.put(f"{_BRANCH_PREFIX}{name}", record.__dict__)

    # -- commits -----------------------------------------------------------------------------

    async def commit(self, commit_id: str) -> Commit | None:
        stored = await self._adapter.get(f"{_COMMIT_PREFIX}{commit_id}")
        if stored is None:
            return None
        return Commit(
            id=commit_id,
            root_id=stored["rootId"],
            message=stored["message"],
            author=stored["author"],
            committed_at=stored["committedAt"],
            parents=tuple(stored.get("parents", ())),
            branch=stored.get("branch", DEFAULT_BRANCH),
        )

    async def head(self, branch: str = DEFAULT_BRANCH) -> Commit | None:
        record = await self.branch(branch)
        return await self.commit(record.head) if record and record.head else None

    async def save(
        self,
        value: Any,
        *,
        message: str,
        author: str,
        branch: str = DEFAULT_BRANCH,
        parents: Sequence[str] | None = None,
        chunk_size: int | None = None,
    ) -> Result[Commit, KernelError]:
        """Decompose a value, store what is new, and record a commit.

        Only objects the store does not already hold are written. An edit to one storey of a
        forty-storey tower writes that storey and the handful of parents above it -- everything
        else is already there under the same id.
        """
        root, produced = serialise(
            value, **({"chunk_size": chunk_size} if chunk_size else {})
        )
        await self.objects.put_many(produced.values())

        if parents is None:
            current = await self.head(branch)
            parents = (current.id,) if current else ()

        payload = {
            "rootId": root.id,
            "message": message,
            "author": author,
            "committedAt": _iso(self._now()),
            "parents": list(parents),
            "branch": branch,
        }
        commit_id = compute_id(payload)
        record = Commit(
            id=commit_id,
            root_id=root.id,
            message=message,
            author=author,
            committed_at=payload["committedAt"],
            parents=tuple(parents),
            branch=branch,
        )
        await self._adapter.put(f"{_COMMIT_PREFIX}{commit_id}", payload)
        await self._set_head(branch, commit_id)
        return ok(record)

    async def load(self, commit_id: str) -> Result[Any, KernelError]:
        record = await self.commit(commit_id)
        if record is None:
            return err(KernelError("COMMAND_FAILED", f'No commit "{commit_id}".', {}))
        try:
            objects = await self.objects.collect(record.root_id)
            return ok(deserialise(record.root_id, objects))
        except VcsError as thrown:
            return err(KernelError("STORAGE_FAILED", str(thrown), {"commitId": commit_id}))

    async def log(self, branch: str = DEFAULT_BRANCH, *, limit: int = 50) -> tuple[Commit, ...]:
        """First-parent history, newest first.

        First-parent rather than a full topological walk: after a merge, the first parent is the
        branch you were on, which is the history a person asking "what happened here" means.
        """
        current = await self.head(branch)
        out: list[Commit] = []
        seen: set[str] = set()
        while current is not None and len(out) < limit:
            if current.id in seen:
                break  # a cycle cannot happen through hashing, but a corrupt store is a thing
            seen.add(current.id)
            out.append(current)
            current = await self.commit(current.parents[0]) if current.parents else None
        return tuple(out)

    async def ancestors(self, commit_id: str) -> set[str]:
        """Every commit reachable from this one, itself included."""
        seen: set[str] = set()
        stack = [commit_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            record = await self.commit(current)
            if record is not None:
                stack.extend(record.parents)
        return seen

    async def merge_base(self, left: str, right: str) -> str | None:
        """The nearest common ancestor -- the third point of a three-way merge."""
        left_ancestors = await self.ancestors(left)
        # Breadth-first from the right so the *nearest* shared commit wins, not any shared one.
        frontier = [right]
        seen: set[str] = set()
        while frontier:
            nxt: list[str] = []
            for commit_id in frontier:
                if commit_id in seen:
                    continue
                seen.add(commit_id)
                if commit_id in left_ancestors:
                    return commit_id
                record = await self.commit(commit_id)
                if record is not None:
                    nxt.extend(record.parents)
            frontier = nxt
        return None

    # -- diff --------------------------------------------------------------------------------

    async def diff(self, from_commit: str, to_commit: str) -> Result[Diff, KernelError]:
        """Set difference over object ids.

        This is the payoff of content addressing. No tree walk, no field comparison: an object
        present in both versions is byte-identical in both, because its id is its content.
        """
        left = await self.commit(from_commit)
        right = await self.commit(to_commit)
        if left is None:
            return err(KernelError("COMMAND_FAILED", f'No commit "{from_commit}".', {}))
        if right is None:
            return err(KernelError("COMMAND_FAILED", f'No commit "{to_commit}".', {}))

        before = set(await self.objects.closure_of(left.root_id)) | {left.root_id}
        after = set(await self.objects.closure_of(right.root_id)) | {right.root_id}
        return ok(
            Diff(
                added=tuple(sorted(after - before)),
                removed=tuple(sorted(before - after)),
                unchanged=tuple(sorted(before & after)),
            )
        )

    # -- merge -------------------------------------------------------------------------------

    async def merge(
        self,
        *,
        ours: str,
        theirs: str,
        author: str,
        message: str | None = None,
        branch: str = DEFAULT_BRANCH,
    ) -> Result[MergeResult, KernelError]:
        """Three-way merge, refusing where both sides changed the same thing differently.

        Conflicts are *reported*, never resolved by preference. "Ours wins" is a policy a person
        chooses knowing what they are discarding, not a default a tool applies silently.
        """
        our_commit = await self.commit(ours)
        their_commit = await self.commit(theirs)
        if our_commit is None or their_commit is None:
            return err(KernelError("COMMAND_FAILED", "Both sides of a merge must exist.", {}))

        base_id = await self.merge_base(ours, theirs)
        if base_id == theirs:
            # Already contains it.
            return ok(MergeResult(ok=True, commit=our_commit, fast_forward=True))
        if base_id == ours:
            await self._set_head(branch, theirs)
            return ok(MergeResult(ok=True, commit=their_commit, fast_forward=True))

        base_commit = await self.commit(base_id) if base_id else None
        base_tree = (
            deserialise(base_commit.root_id, await self.objects.collect(base_commit.root_id))
            if base_commit
            else {}
        )
        our_tree = deserialise(our_commit.root_id, await self.objects.collect(our_commit.root_id))
        their_tree = deserialise(
            their_commit.root_id, await self.objects.collect(their_commit.root_id)
        )

        merged, conflicts = _merge_trees(base_tree, our_tree, their_tree, path="")
        if conflicts:
            return ok(MergeResult(ok=False, conflicts=tuple(conflicts)))

        saved = await self.save(
            merged,
            message=message or f"Merge {theirs[:8]} into {ours[:8]}",
            author=author,
            branch=branch,
            parents=(ours, theirs),
        )
        if not saved.ok:
            return err(saved.error)
        return ok(MergeResult(ok=True, commit=saved.value))

    # -- tags --------------------------------------------------------------------------------

    async def tag(
        self, name: str, commit_id: str, message: str | None = None
    ) -> Result[Tag, KernelError]:
        if await self.commit(commit_id) is None:
            return err(KernelError("COMMAND_FAILED", f'No commit "{commit_id}".', {}))
        existing = await self._adapter.get(f"{_TAG_PREFIX}{name}")
        if existing is not None:
            # Tags name issued states -- a drawing pack, a tender. Moving one silently rewrites
            # what somebody was handed.
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Tag "{name}" already points at {existing["commit_id"][:8]}. Tags are '
                    "immutable; issue a new one rather than moving this.",
                    {},
                )
            )
        record = Tag(name=name, commit_id=commit_id, message=message)
        await self._adapter.put(f"{_TAG_PREFIX}{name}", record.__dict__)
        return ok(record)

    async def tags(self) -> tuple[Tag, ...]:
        keys = await self._adapter.keys(_TAG_PREFIX)
        found = []
        for key in keys:
            stored = await self._adapter.get(key)
            if stored is not None:
                found.append(Tag(**stored))
        return tuple(sorted(found, key=lambda tag: tag.name))


def _merge_trees(
    base: Any, ours: Any, theirs: Any, *, path: str
) -> tuple[Any, list[MergeConflict]]:
    """Recursive three-way merge of two dicts against their common ancestor."""
    conflicts: list[MergeConflict] = []

    if not isinstance(ours, Mapping) or not isinstance(theirs, Mapping):
        if ours == theirs:
            return ours, conflicts
        if ours == base:
            return theirs, conflicts  # only they changed it
        if theirs == base:
            return ours, conflicts  # only we changed it
        conflicts.append(MergeConflict(path=path or "<root>", base=base, ours=ours, theirs=theirs))
        return ours, conflicts

    base_map = base if isinstance(base, Mapping) else {}
    merged: dict[str, Any] = {}

    for key in sorted(set(ours) | set(theirs)):
        child_path = f"{path}.{key}" if path else key
        in_ours = key in ours
        in_theirs = key in theirs
        base_value = base_map.get(key)

        if in_ours and not in_theirs:
            # They deleted it. If we did not change it, honour the deletion.
            if key in base_map and ours[key] == base_value:
                continue
            if key not in base_map:
                merged[key] = ours[key]
                continue
            conflicts.append(
                MergeConflict(path=child_path, base=base_value, ours=ours[key], theirs=None)
            )
            merged[key] = ours[key]
            continue

        if in_theirs and not in_ours:
            if key in base_map and theirs[key] == base_value:
                continue
            if key not in base_map:
                merged[key] = theirs[key]
                continue
            conflicts.append(
                MergeConflict(path=child_path, base=base_value, ours=None, theirs=theirs[key])
            )
            continue

        child, child_conflicts = _merge_trees(
            base_value, ours[key], theirs[key], path=child_path
        )
        merged[key] = child
        conflicts.extend(child_conflicts)

    return merged, conflicts
