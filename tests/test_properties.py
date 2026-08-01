"""Randomised invariant checks over the pure surfaces.

Every other test file states a specific claim and pins it with a specific case. These state a claim
that must hold for *any* input and then throw a lot of structured randomness at it. The two find
different things: hand-picked cases find the failure you thought of, and this finds the one you did
not -- an escape character nobody considered, a mesh with one degenerate face, a tree where both
sides deleted the same key.

**Fixed seeds.** A suite that generates fresh inputs every run goes red on a commit that changed
nothing, and a test that fails for reasons unrelated to the change is a test people learn to
re-run rather than read. The seeds here are constant, so a failure is reproducible from the name
alone; widen the search locally with

    MASSINGVISER_FUZZ_ROUNDS=5000 pytest tests/test_properties.py

These properties were validated by injecting faults and confirming each was caught -- a suite that
always passes is worth nothing until you know it *can* fail.
"""

from __future__ import annotations

import json
import math
import os
import random
import string

import pytest

from massingviser.kernel.semver import compare_semver, parse_semver, satisfies
from massingviser.plugins.estimating.math import evaluate_expression
from massingviser.plugins.icdd.rdf import Graph, Iri, Literal
from massingviser.plugins.icdd.syntaxes import from_jsonld, from_turtle, to_jsonld, to_turtle
from massingviser.schema import Money
from massingviser.storage.filesystem import decode_key, encode_key
from massingviser.vcs.history import _merge_trees

#: Enough to exercise the shapes without slowing the build. CI runs this on every push.
ROUNDS = int(os.environ.get("MASSINGVISER_FUZZ_ROUNDS", "150"))

#: Three unrelated streams, so a property that only fails for one shape of input still shows up.
SEEDS = (1, 42, 99991)

#: Characters that have historically broken *something* somewhere: path separators, Windows
#: reserved punctuation, percent-encoding, whitespace and non-ASCII.
_ALPHABET = string.printable[:95] + 'éßπ中\t\n\\/:*?"<>|%'


def _text(rng: random.Random, limit: int = 24) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, limit)))


def _json(rng: random.Random, depth: int = 0):
    kind = rng.randint(0, 6 if depth < 3 else 3)
    if kind == 0:
        return rng.randint(-(10**6), 10**6)
    if kind == 1:
        return _text(rng, 6)
    if kind == 2:
        return rng.choice([True, False, None])
    if kind == 3:
        return round(rng.uniform(-1e4, 1e4), 6)
    if kind == 4:
        return [_json(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {_text(rng, 4): _json(rng, depth + 1) for _ in range(rng.randint(0, 3))}


# ---------------------------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_any_key_survives_encoding_to_a_filename_and_back(seed):
    """A key that does not round-trip is a record saved under one name and looked up under another."""
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        key = _text(rng, 40)
        assert decode_key(encode_key(key)) == key, f"key {key!r}"


# ---------------------------------------------------------------------------------------------
# Semver
# ---------------------------------------------------------------------------------------------


def _version(rng: random.Random) -> str:
    return f"{rng.randint(0, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"


@pytest.mark.parametrize("seed", SEEDS)
def test_version_comparison_is_antisymmetric(seed):
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        a, b = _version(rng), _version(rng)
        left = compare_semver(parse_semver(a), parse_semver(b))
        right = compare_semver(parse_semver(b), parse_semver(a))
        assert (left > 0) == (right < 0), f"{a} vs {b}"
        assert (left == 0) == (right == 0), f"{a} vs {b}"


@pytest.mark.parametrize("seed", SEEDS)
def test_a_caret_range_never_matches_something_older(seed):
    """The failure this prevents: resolving a dependency to a build that predates the requirement."""
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        older, newer = _version(rng), _version(rng)
        if compare_semver(parse_semver(older), parse_semver(newer)) >= 0:
            continue
        assert satisfies(older, f"^{newer}") is False, f"^{newer} matched {older}"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_version_satisfies_itself(seed):
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        version = _version(rng)
        assert satisfies(version, version)
        assert satisfies(version, f">={version}")
        assert satisfies(version, f"^{version}")


# ---------------------------------------------------------------------------------------------
# The takeoff evaluator
# ---------------------------------------------------------------------------------------------

_EXPRESSION_ATOMS = [
    "1",
    "2.5",
    "0",
    "A",
    "B",
    "(",
    ")",
    "+",
    "-",
    "*",
    "/",
    "%",
    "^",
    ",",
    "min",
    "max",
    "abs",
    "sqrt",
    "round",
    " ",
]


@pytest.mark.parametrize("seed", SEEDS)
def test_no_expression_can_make_the_evaluator_raise(seed):
    """It is handed formulas out of saved projects and cost libraries -- untrusted input.

    The contract is a Result either way. Anything that escapes as an exception takes down whatever
    was running a takeoff, which is a batch job over every element in a model.
    """
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        expression = "".join(rng.choice(_EXPRESSION_ATOMS) for _ in range(rng.randint(1, 14)))
        variables = {"A": rng.uniform(-100, 100), "B": rng.uniform(-100, 100)}
        result = evaluate_expression(expression, variables)
        if result.ok:
            assert math.isfinite(result.value), f"{expression!r} produced {result.value}"


# ---------------------------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_money_addition_is_exactly_reversible(seed):
    """The property floats do not have, and the reason this type exists."""
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        a, b = rng.randint(-(10**9), 10**9), rng.randint(-(10**9), 10**9)
        assert (Money(a, "GBP") + Money(b, "GBP")) - Money(b, "GBP") == Money(a, "GBP")


@pytest.mark.parametrize("seed", SEEDS)
def test_scaling_money_always_lands_on_a_whole_minor_unit(seed):
    """A fractional penny would reintroduce exactly the drift minor units exist to prevent."""
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        amount = rng.randint(-(10**9), 10**9)
        scaled = Money(amount, "GBP").scaled(rng.uniform(-1000, 1000))
        assert isinstance(scaled.amount_minor, int)
        assert not isinstance(scaled.amount_minor, bool)


# ---------------------------------------------------------------------------------------------
# Geometry payloads
# ---------------------------------------------------------------------------------------------


def _meshes(rng: random.Random, count: int):
    numpy = pytest.importorskip("numpy")
    from massingviser.geometry.payload import MeshInput

    built = []
    for index in range(count):
        vertex_count = rng.randint(3, 40)
        vertices = numpy.array(
            [[rng.uniform(-1e3, 1e3) for _ in range(3)] for _ in range(vertex_count)]
        )
        faces = numpy.array(
            [[rng.randrange(vertex_count) for _ in range(3)] for _ in range(rng.randint(1, 30))]
        )
        built.append(MeshInput(f"E{index}", vertices, faces))
    return built


@pytest.mark.parametrize("seed", SEEDS)
def test_any_mesh_batch_decodes_to_what_was_encoded(seed):
    """The wire contract, over arbitrary shapes rather than the two cubes in the fixtures."""
    from massingviser.geometry.payload import decode_mesh_batch, encode_mesh_batch

    rng = random.Random(seed)
    for _ in range(max(ROUNDS // 5, 20)):
        meshes = _meshes(rng, rng.randint(1, 4))
        decoded = decode_mesh_batch(encode_mesh_batch(meshes).data)
        assert len(decoded) == len(meshes)
        for original, back in zip(meshes, decoded, strict=True):
            assert len(back.vertices) == len(original.vertices)
            assert len(back.faces) == len(original.faces)
            # No index may reach past its own mesh -- that reads a neighbour's vertices and draws
            # something plausible and wrong.
            if len(back.faces):
                assert back.faces.max() < len(back.vertices)


@pytest.mark.parametrize("seed", SEEDS)
def test_chunking_never_loses_reorders_or_duplicates_a_mesh(seed):
    from massingviser.geometry.payload import chunk_meshes

    rng = random.Random(seed)
    for _ in range(max(ROUNDS // 5, 20)):
        meshes = _meshes(rng, rng.randint(1, 6))
        chunks = chunk_meshes(meshes, chunk_vertices=rng.randint(1, 200))
        flattened = [mesh.global_id for chunk in chunks for mesh in chunk]
        assert flattened == [mesh.global_id for mesh in meshes]


# ---------------------------------------------------------------------------------------------
# Three-way merge
# ---------------------------------------------------------------------------------------------


def _tree(rng: random.Random, depth: int = 0):
    return {
        f"k{index}": (_tree(rng, depth + 1) if depth < 2 and rng.random() < 0.4 else _json(rng))
        for index in range(rng.randint(0, 4))
    }


def _mutated(rng: random.Random, tree):
    copy = json.loads(json.dumps(tree))
    for _ in range(rng.randint(0, 3)):
        if not isinstance(copy, dict) or not copy:
            break
        key = rng.choice(list(copy))
        action = rng.randint(0, 2)
        if action == 0:
            del copy[key]
        elif action == 1:
            copy[key] = _json(rng)
        else:
            copy[f"new{rng.randint(0, 9)}"] = _json(rng)
    return copy


@pytest.mark.parametrize("seed", SEEDS)
def test_a_side_merged_against_itself_never_conflicts(seed):
    """If both people made the same edit there is nothing to decide between."""
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        base = _tree(rng)
        ours = _mutated(rng, base)
        merged, conflicts = _merge_trees(base, ours, json.loads(json.dumps(ours)), path="")
        assert conflicts == []
        assert merged == ours


@pytest.mark.parametrize("seed", SEEDS)
def test_conflict_detection_does_not_depend_on_which_side_is_ours(seed):
    """Whether a merge is refused must not turn on the order the two branches were named."""
    rng = random.Random(seed)
    for _ in range(ROUNDS):
        base = _tree(rng)
        ours, theirs = _mutated(rng, base), _mutated(rng, base)
        _, forward = _merge_trees(base, ours, theirs, path="")
        _, backward = _merge_trees(base, theirs, ours, path="")
        assert len(forward) == len(backward)
        assert {c.path for c in forward} == {c.path for c in backward}


# ---------------------------------------------------------------------------------------------
# RDF syntaxes
# ---------------------------------------------------------------------------------------------


def _graph(rng: random.Random) -> Graph:
    graph = Graph()
    for _ in range(rng.randint(1, 6)):
        subject = f"urn:s:{rng.randint(0, 5)}"
        predicate = f"urn:p:{rng.randint(0, 5)}"
        if rng.random() < 0.5:
            graph.add(subject, predicate, Iri(f"urn:o:{rng.randint(0, 5)}"))
        else:
            graph.add(subject, predicate, Literal(_text(rng, 10)))
    return graph


def _triples(graph: Graph):
    return sorted((t.subject, t.predicate, repr(t.object)) for t in graph)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("dump", "load"), [(to_turtle, from_turtle), (to_jsonld, from_jsonld)])
def test_a_graph_survives_a_round_trip_through_either_syntax(seed, dump, load):
    """Including literals full of quotes, backslashes, newlines and non-ASCII."""
    rng = random.Random(seed)
    for _ in range(max(ROUNDS // 3, 30)):
        graph = _graph(rng)
        assert _triples(load(dump(graph))) == _triples(graph)
