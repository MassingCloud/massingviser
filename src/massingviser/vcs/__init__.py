"""``massingviser.vcs`` -- version control for models.

Git's shape, Speckle's object model, no dependencies. A model is decomposed into atomic objects
identified by the hash of their content; a commit names a root object and its parents; a branch is
a moving name for a commit.

    repo = Repository(FileSystemStorageAdapter("project.mass"))
    await repo.save(scheme, message="Add north slab", author="ada")
    await repo.diff(before.id, after.id)      # a set difference over ids, not a tree walk

Three properties fall out of content addressing, and they are the reason for it:

- two versions sharing a wall store that wall **once**;
- an object present in both versions is **byte-identical** in both, so diffing is a set operation;
- an id that does not match its content is **detectable** corruption.
"""

from .history import (
    DEFAULT_BRANCH,
    Branch,
    Commit,
    Diff,
    MergeConflict,
    MergeResult,
    ObjectStore,
    Repository,
    Tag,
)
from .objects import (
    DEFAULT_CHUNK_SIZE,
    DETACH_PREFIX,
    ID_LENGTH,
    REFERENCE_TYPE,
    Reference,
    SerialisedObject,
    Serialiser,
    VcsError,
    canonical_json,
    compute_id,
    deserialise,
    serialise,
    verify,
)

__all__ = [
    "DEFAULT_BRANCH",
    "DEFAULT_CHUNK_SIZE",
    "DETACH_PREFIX",
    "ID_LENGTH",
    "REFERENCE_TYPE",
    "Branch",
    "Commit",
    "Diff",
    "MergeConflict",
    "MergeResult",
    "ObjectStore",
    "Reference",
    "Repository",
    "SerialisedObject",
    "Serialiser",
    "Tag",
    "VcsError",
    "canonical_json",
    "compute_id",
    "deserialise",
    "serialise",
    "verify",
]
