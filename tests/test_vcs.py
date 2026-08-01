"""Version control.

Content addressing is the whole design, so these tests pin the three properties that follow from
it: identical content dedupes, an object present in two versions is unchanged *by definition*, and
a mismatched hash is detectable corruption.
"""

from __future__ import annotations

import copy

import pytest

from massingviser.kernel import MemoryStorageAdapter
from massingviser.vcs import (
    DEFAULT_BRANCH,
    Reference,
    Repository,
    VcsError,
    canonical_json,
    compute_id,
    deserialise,
    serialise,
    verify,
)
from massingviser.vcs.history import _merge_trees

SCHEME = {
    "name": "Tower",
    "@masses": [
        {"name": "A", "storeys": 10, "@profile": {"points": [[0, 0], [30, 0], [30, 20], [0, 20]]}},
        {"name": "B", "storeys": 6, "@profile": {"points": [[0, 0], [30, 0], [30, 20], [0, 20]]}},
    ],
}


# ---------------------------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------------------------


def test_ids_are_deterministic_regardless_of_key_order():
    """Two processes on two machines must agree, or the id is a coincidence not an identity."""
    assert compute_id({"a": 1, "b": 2}) == compute_id({"b": 2, "a": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_an_id_is_128_bits_of_hex():
    identifier = compute_id({"anything": True})
    assert len(identifier) == 32
    assert all(character in "0123456789abcdef" for character in identifier)


def test_the_id_member_is_excluded_from_its_own_hash():
    """An object cannot contain its own hash."""
    payload = {"name": "wall"}
    identifier = compute_id(payload)
    assert compute_id({**payload, "id": identifier}) == identifier


def test_identical_content_is_stored_once():
    """Two masses sharing a footprint store that footprint once."""
    _, objects = serialise(SCHEME)
    # root + two masses + one shared profile.
    assert len(objects) == 4


def test_changing_one_branch_leaves_the_others_untouched():
    """An unchanged subtree keeps its id no matter what happens above it."""
    root_a, before = serialise(SCHEME)
    changed = copy.deepcopy(SCHEME)
    changed["@masses"][0]["storeys"] = 14
    root_b, after = serialise(changed)

    assert root_a.id != root_b.id
    shared = set(before) & set(after)
    # The untouched mass and the shared profile survive; the root and one mass are new.
    assert len(shared) == 2


def test_detached_members_become_references():
    root, objects = serialise({"name": "x", "@child": {"deep": 1}})
    assert Reference.is_reference(root.payload["child"])
    referenced = root.payload["child"]["referencedId"]
    assert referenced in objects


def test_an_undetached_member_stays_inline():
    root, objects = serialise({"name": "x", "child": {"deep": 1}})
    assert root.payload["child"] == {"deep": 1}
    assert len(objects) == 1


def test_the_closure_records_every_descendant_and_its_depth():
    """One lookup instead of a recursive walk -- which matters over a network."""
    root, _ = serialise(SCHEME)
    depths = set(root.closure.values())
    assert len(root.closure) == 3  # two masses and the shared profile
    assert depths == {1, 2}


def test_a_long_detached_list_is_chunked():
    root, objects = serialise({"@vertices": list(range(25))}, chunk_size=10)
    references = root.payload["vertices"]
    assert len(references) == 3  # 10 + 10 + 5
    assert all(Reference.is_reference(item) for item in references)
    assert len(objects) == 4  # three chunks plus the root


def test_a_chunked_list_flattens_back_on_the_way_out():
    root, objects = serialise({"@vertices": list(range(25))}, chunk_size=10)
    rebuilt = deserialise(root.id, {i: o.payload for i, o in objects.items()})
    assert rebuilt["vertices"] == list(range(25))


def test_a_tree_round_trips():
    root, objects = serialise(SCHEME)
    rebuilt = deserialise(root.id, {i: o.payload for i, o in objects.items()})
    assert rebuilt["name"] == "Tower"
    assert [mass["name"] for mass in rebuilt["masses"]] == ["A", "B"]
    assert rebuilt["masses"][0]["profile"]["points"][1] == [30, 0]


def test_a_missing_referenced_object_refuses_rather_than_half_loading():
    """A half-resolved model that looks whole is worse than one that refuses."""
    root, objects = serialise(SCHEME)
    partial = {i: o.payload for i, o in objects.items()}
    victim = next(i for i in partial if i != root.id)
    del partial[victim]
    with pytest.raises(VcsError, match="referenced but not present"):
        deserialise(root.id, partial)


def test_corruption_is_detectable():
    root, _ = serialise({"name": "wall"})
    assert verify(root.id, root.payload)
    assert not verify(root.id, {**root.payload, "name": "tampered"})


def test_sets_hash_independently_of_insertion_order():
    assert compute_id({"tags": {"a", "b"}}) == compute_id({"tags": {"b", "a"}})


# ---------------------------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def repo():
    return Repository(MemoryStorageAdapter())


async def test_a_commit_records_its_root_and_moves_the_branch(repo):
    commit = (await repo.save(SCHEME, message="initial", author="ada")).value
    assert commit.parents == ()
    assert commit.branch == DEFAULT_BRANCH
    assert (await repo.head()).id == commit.id


async def test_a_second_commit_names_the_first_as_parent(repo):
    first = (await repo.save(SCHEME, message="initial", author="ada")).value
    second = (await repo.save({**SCHEME, "name": "Tower II"}, message="rename", author="ada")).value
    assert second.parents == (first.id,)
    assert [c.message for c in await repo.log()] == ["rename", "initial"]


async def test_only_new_objects_are_written(repo):
    """An edit to one storey writes that storey and its parents, not the model."""
    await repo.save(SCHEME, message="initial", author="ada")
    changed = copy.deepcopy(SCHEME)
    changed["@masses"][0]["storeys"] = 14

    root, produced = serialise(changed)
    written = await repo.objects.put_many(produced.values())
    assert written == 2  # the changed mass and the new root; the rest is already held


async def test_diff_is_a_set_difference_not_a_tree_walk(repo):
    first = (await repo.save(SCHEME, message="initial", author="ada")).value
    changed = copy.deepcopy(SCHEME)
    changed["@masses"][0]["storeys"] = 14
    second = (await repo.save(changed, message="taller", author="ada")).value

    diff = (await repo.diff(first.id, second.id)).value
    assert diff.churn == 4  # two objects gone, two arrived
    # Anything in both versions is byte-identical in both -- that is what the id means.
    assert set(diff.unchanged) & set(diff.added) == set()


async def test_diffing_a_version_against_itself_is_empty(repo):
    commit = (await repo.save(SCHEME, message="initial", author="ada")).value
    assert (await repo.diff(commit.id, commit.id)).value.is_empty


async def test_loading_reconstructs_the_committed_state(repo):
    first = (await repo.save(SCHEME, message="initial", author="ada")).value
    changed = copy.deepcopy(SCHEME)
    changed["@masses"][0]["storeys"] = 14
    await repo.save(changed, message="taller", author="ada")

    # Checking out an old version gives what was committed, not what is current.
    restored = (await repo.load(first.id)).value
    assert restored["masses"][0]["storeys"] == 10


async def test_a_branch_starts_at_the_current_head_and_diverges(repo):
    base = (await repo.save(SCHEME, message="initial", author="ada")).value
    assert (await repo.create_branch("option-b")).ok
    assert (await repo.branch("option-b")).head == base.id

    variant = copy.deepcopy(SCHEME)
    variant["@masses"][0]["storeys"] = 30
    await repo.save(variant, message="tall option", author="ada", branch="option-b")

    assert (await repo.head()).id == base.id  # main did not move
    assert (await repo.head("option-b")).message == "tall option"


async def test_a_duplicate_branch_name_is_refused(repo):
    await repo.save(SCHEME, message="initial", author="ada")
    await repo.create_branch("option-b")
    assert not (await repo.create_branch("option-b")).ok


async def test_the_merge_base_is_the_nearest_common_ancestor(repo):
    base = (await repo.save(SCHEME, message="initial", author="ada")).value
    await repo.create_branch("option-b")

    ours = (await repo.save({**SCHEME, "client": "A"}, message="ours", author="ada")).value
    theirs = (
        await repo.save(
            {**SCHEME, "phase": "RIBA 2"}, message="theirs", author="bob", branch="option-b"
        )
    ).value

    assert await repo.merge_base(ours.id, theirs.id) == base.id


async def test_a_merge_of_disjoint_edits_succeeds(repo):
    await repo.save(SCHEME, message="initial", author="ada")
    await repo.create_branch("option-b")

    ours = (await repo.save({**SCHEME, "client": "A"}, message="ours", author="ada")).value
    theirs = (
        await repo.save(
            {**SCHEME, "phase": "RIBA 2"}, message="theirs", author="bob", branch="option-b"
        )
    ).value

    merged = (await repo.merge(ours=ours.id, theirs=theirs.id, author="ada")).value
    assert merged.ok and merged.commit is not None
    assert merged.commit.parents == (ours.id, theirs.id)

    state = (await repo.load(merged.commit.id)).value
    assert state["client"] == "A" and state["phase"] == "RIBA 2"


async def test_a_merge_where_both_sides_changed_the_same_thing_reports_a_conflict(repo):
    """ "Ours wins" is a decision a person makes knowing what they discard, not a default."""
    await repo.save(SCHEME, message="initial", author="ada")
    await repo.create_branch("option-b")

    ours = (await repo.save({**SCHEME, "name": "Tower North"}, message="ours", author="ada")).value
    theirs = (
        await repo.save(
            {**SCHEME, "name": "Tower South"}, message="theirs", author="bob", branch="option-b"
        )
    ).value

    result = (await repo.merge(ours=ours.id, theirs=theirs.id, author="ada")).value
    assert not result.ok
    conflict = result.conflicts[0]
    assert conflict.path == "name"
    assert (conflict.ours, conflict.theirs) == ("Tower North", "Tower South")


async def test_merging_an_ancestor_fast_forwards(repo):
    first = (await repo.save(SCHEME, message="initial", author="ada")).value
    second = (await repo.save({**SCHEME, "name": "II"}, message="next", author="ada")).value

    result = (await repo.merge(ours=second.id, theirs=first.id, author="ada")).value
    assert result.ok and result.fast_forward


async def test_a_tag_cannot_be_moved(repo):
    """Tags name issued states -- a drawing pack, a tender."""
    first = (await repo.save(SCHEME, message="initial", author="ada")).value
    second = (await repo.save({**SCHEME, "name": "II"}, message="next", author="ada")).value

    assert (await repo.tag("RIBA-2-issue", first.id)).ok
    moved = await repo.tag("RIBA-2-issue", second.id)
    assert not moved.ok and "immutable" in moved.error.message
    assert [tag.commit_id for tag in await repo.tags()] == [first.id]


async def test_the_store_reports_what_it_is_missing(repo):
    """The whole point of a push protocol: ask what is missing, send only that."""
    root, objects = serialise(SCHEME)
    assert len(await repo.objects.missing(objects)) == len(objects)
    await repo.objects.put_many(objects.values())
    assert await repo.objects.missing(objects) == ()


async def test_a_version_verifies_end_to_end(repo):
    commit = (await repo.save(SCHEME, message="initial", author="ada")).value
    assert await repo.objects.verify_all(commit.root_id) == ()


# ---------------------------------------------------------------------------------------------
# What "absent from the base" means
#
# `base_map.get(key)` returned None both for "the base did not have this key" and for "the base
# held null". That let `ours == base` succeed when neither side had inherited anything, so where
# both sides created the same record and one left a field null, the other side's value was taken
# silently -- a conflict resolved by preference, which is the one thing this module says it never
# does.
# ---------------------------------------------------------------------------------------------


def test_both_sides_adding_a_key_conflicts_even_when_one_writes_null():
    merged, conflicts = _merge_trees(
        {}, {"r1": {"note": None}}, {"r1": {"note": "theirs"}}, path=""
    )
    assert len(conflicts) == 1
    assert conflicts[0].path == "r1.note"
    assert conflicts[0].ours is None and conflicts[0].theirs == "theirs"
    # Nothing was inherited, so there is no base value to report.
    assert conflicts[0].base is None


def test_the_same_holds_in_the_other_direction():
    """Neither side may win by default; the asymmetry would just move the silent loss."""
    _, conflicts = _merge_trees({}, {"r1": {"note": "ours"}}, {"r1": {"note": None}}, path="")
    assert len(conflicts) == 1


def test_a_base_that_really_held_null_still_merges_cleanly():
    """The fix must not turn 'only they changed it' into a conflict."""
    merged, conflicts = _merge_trees(
        {"r1": {"note": None}}, {"r1": {"note": None}}, {"r1": {"note": "theirs"}}, path=""
    )
    assert conflicts == []
    assert merged == {"r1": {"note": "theirs"}}


def test_both_sides_adding_the_same_value_is_not_a_conflict():
    merged, conflicts = _merge_trees({}, {"r1": {"n": 1}}, {"r1": {"n": 1}}, path="")
    assert conflicts == []
    assert merged == {"r1": {"n": 1}}


def test_disjoint_edits_still_merge():
    merged, conflicts = _merge_trees({"a": 1, "b": 2}, {"a": 9, "b": 2}, {"a": 1, "b": 8}, path="")
    assert conflicts == []
    assert merged == {"a": 9, "b": 8}


def test_a_key_only_one_side_adds_is_kept_without_a_conflict():
    merged, conflicts = _merge_trees({}, {"a": 1}, {}, path="")
    assert conflicts == [] and merged == {"a": 1}
    merged, conflicts = _merge_trees({}, {}, {"b": 2}, path="")
    assert conflicts == [] and merged == {"b": 2}
