"""SHACL, less the parts that would mean writing a query engine.

The previous position here was that a SHACL engine is a large dependency for failures that are
almost always mundane. That is still true of a *complete* engine -- SPARQL-based constraints alone
would mean an entire query language, and `sh:sparql` is still refused rather than approximated.

What is implemented is everything that reduces to walking a graph and comparing values:
cardinality, datatype, class, node kind, value ranges, string tests and enumerations; the logical
constraints `sh:node`, `sh:not`, `sh:and`, `sh:or`, `sh:xone` and qualified value shapes; and
property paths beyond a bare predicate -- inverse, sequence and alternative.

The logical constraints are why shapes are addressable by IRI rather than being a flat list: each
runs *another* shape against a value, and a shape referenced by `sh:node` is frequently never typed
`sh:NodeShape` at all, because SHACL infers shape-hood from use.

So that is what this validates, and the one rule that makes it honest:

    **A constraint this engine does not implement is reported, never ignored.**

That distinction is the whole difference between a subset and a lie. An engine that skips
``sh:sparql`` and returns "conforms" has told you the graph satisfies a shape it never checked. Here
an unsupported constraint produces an ``unsupported`` entry in the report, and ``report.complete``
is ``False`` -- so "conforms" means "conforms as far as anything was checked", and you can see how
far that was.

Shapes are read from the same ``Graph`` the rest of this package uses, so they can come from
RDF/XML, Turtle or JSON-LD without this module knowing which.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .ontology import NS
from .rdf import Graph, Iri, Literal

SH = "http://www.w3.org/ns/shacl#"
XSD = "http://www.w3.org/2001/XMLSchema#"

#: Constraint parameters this engine evaluates. Anything else on a shape is reported.
SUPPORTED = frozenset(
    {
        "minCount",
        "maxCount",
        "datatype",
        "class",
        "nodeKind",
        "in",
        "hasValue",
        "pattern",
        "flags",
        "minLength",
        "maxLength",
        "languageIn",
        "minInclusive",
        "maxInclusive",
        "minExclusive",
        "maxExclusive",
        # Logical constraints. Each takes a shape (or a list of them) and is evaluated by running
        # that shape against the value, which is why shapes are addressable by IRI.
        "node",
        "not",
        "and",
        "or",
        "xone",
        "qualifiedValueShape",
        "qualifiedMinCount",
        "qualifiedMaxCount",
        # Path forms beyond a bare predicate.
        "inversePath",
        "alternativePath",
        # Structural, not constraints in themselves.
        "path",
        "property",
        "targetClass",
        "targetNode",
        "targetSubjectsOf",
        "targetObjectsOf",
        "severity",
        "message",
        "name",
        "description",
        "deactivated",
        "closed",
        "ignoredProperties",
        "order",
        "group",
    }
)

#: Severity IRIs, shortest name last so a report reads in plain words.
_SEVERITY = {
    f"{SH}Violation": "violation",
    f"{SH}Warning": "warning",
    f"{SH}Info": "info",
}


@dataclass(frozen=True)
class Result:
    """One constraint that did not hold."""

    focus: str
    path: str | None
    constraint: str
    severity: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        where = f"{self.focus} {self.path}" if self.path else self.focus
        return f"[{self.severity}] {where}: {self.message}"


@dataclass(frozen=True)
class ShaclReport:
    results: tuple[Result, ...] = ()
    #: Constraint parameters present on the shapes that this engine cannot evaluate.
    unsupported: tuple[str, ...] = ()
    shapes_run: int = 0
    focus_nodes: int = 0

    @property
    def conforms(self) -> bool:
        """No violations. Warnings and info do not stop a graph conforming, per the spec."""
        return not any(result.severity == "violation" for result in self.results)

    @property
    def complete(self) -> bool:
        """Whether every constraint on every shape was actually evaluated."""
        return not self.unsupported

    @property
    def violations(self) -> tuple[Result, ...]:
        return tuple(r for r in self.results if r.severity == "violation")


@dataclass(frozen=True)
class Path:
    """How to get from a focus node to the values a property shape constrains.

    A bare predicate covers most shapes. The rest of SHACL's path language is a small grammar:
    follow a predicate backwards, follow several in sequence, or accept any of several. Each is a
    different question about the graph, and answering the wrong one silently constrains the wrong
    values -- so an unrecognised form is refused at parse time rather than approximated.
    """

    kind: str = "predicate"
    #: For `predicate` and `inverse`.
    predicate: str | None = None
    #: For `sequence` and `alternative`.
    steps: tuple[Path, ...] = ()

    def describe(self) -> str:
        if self.kind == "predicate":
            return self.predicate or "?"
        if self.kind == "inverse":
            return f"^{self.predicate}"
        joiner = "/" if self.kind == "sequence" else "|"
        return "(" + joiner.join(step.describe() for step in self.steps) + ")"

    def reach(self, graph: Graph, focus: str) -> list[Any]:
        """Every value this path leads to from one focus node."""
        if self.kind == "predicate":
            return graph.objects(focus, self.predicate or "")
        if self.kind == "inverse":
            # Backwards: every subject that points *at* this node through the predicate.
            return [
                Iri(triple.subject)
                for triple in graph
                if triple.predicate == self.predicate
                and isinstance(triple.object, Iri)
                and triple.object.value == focus
            ]
        if self.kind == "sequence":
            current: list[Any] = [Iri(focus)]
            for step in self.steps:
                nxt: list[Any] = []
                for value in current:
                    if isinstance(value, Iri):
                        nxt.extend(step.reach(graph, value.value))
                current = nxt
            return current
        if self.kind == "alternative":
            found: list[Any] = []
            for step in self.steps:
                for value in step.reach(graph, focus):
                    if value not in found:
                        found.append(value)
            return found
        return []


@dataclass(frozen=True)
class PropertyShape:
    path: Path
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    severity: str = "violation"
    message: str | None = None


@dataclass(frozen=True)
class NodeShape:
    id: str
    target_classes: tuple[str, ...] = ()
    target_nodes: tuple[str, ...] = ()
    target_subjects_of: tuple[str, ...] = ()
    target_objects_of: tuple[str, ...] = ()
    properties: tuple[PropertyShape, ...] = ()
    deactivated: bool = False
    #: Constraint parameters on the *node shape itself* that this engine cannot evaluate, plus a
    #: marker for any property shape whose path it could not express. Held here because scanning
    #: only property parameters misses `sh:sparql`, `sh:and`, `sh:not` and every other node-level
    #: component -- and missing them makes `complete` claim a shape was checked when it was not.
    unsupported: tuple[str, ...] = ()


def _local(iri: str) -> str:
    return iri[len(SH) :] if iri.startswith(SH) else iri


def _values(graph: Graph, subject: str, predicate: str) -> list[Any]:
    return graph.objects(subject, predicate)


def _plain(value: Any) -> Any:
    if isinstance(value, Literal):
        return value.value
    if isinstance(value, Iri):
        return value.value
    return value


def _parse_path(value: Any, graph: Graph) -> Path | None:
    """Read a property path, or ``None`` for a form this engine does not implement."""
    if isinstance(value, Iri):
        # Either a bare predicate, or a node carrying one of the path constructs.
        inverse = graph.value(value.value, f"{SH}inversePath")
        if isinstance(inverse, Iri):
            inner = _parse_path(inverse, graph)
            return Path("inverse", predicate=inner.predicate) if inner else None

        alternative = graph.value(value.value, f"{SH}alternativePath")
        if alternative is not None:
            members = _rdf_list(alternative, graph)
            steps = [_parse_path(member, graph) for member in members or ()]
            if members and all(steps):
                return Path("alternative", steps=tuple(s for s in steps if s))
            return None

        sequence = _rdf_list(value, graph)
        if sequence is not None and len(sequence) > 1:
            steps = [_parse_path(member, graph) for member in sequence]
            if all(steps):
                return Path("sequence", steps=tuple(s for s in steps if s))
            return None

        # A node carrying a path construct this engine does not implement -- `zeroOrMorePath`,
        # `oneOrMorePath`, `zeroOrOnePath`. Falling through to "bare predicate" would treat the
        # *construct node's own IRI* as a predicate, which matches nothing and reports every focus
        # node as missing a value it was never asked for: a wrong answer dressed as a finding.
        for triple in graph:
            if (
                triple.subject == value.value
                and triple.predicate.startswith(SH)
                and triple.predicate.endswith("Path")
            ):
                return None

        return Path("predicate", predicate=value.value)
    return None


def _shape_ids(graph: Graph) -> list[str]:
    """Every subject that is a shape, declared or implied.

    A shape referenced by `sh:node` or `sh:not` is frequently not typed `sh:NodeShape` at all --
    SHACL infers shape-hood from use. Collecting only declared ones would leave every logical
    constraint pointing at nothing and quietly passing.
    """
    declared = list(graph.subjects_of_type(f"{SH}NodeShape"))
    referenced: list[str] = []
    for predicate in ("node", "not", "qualifiedValueShape"):
        for triple in graph:
            if triple.predicate == f"{SH}{predicate}" and isinstance(triple.object, Iri):
                referenced.append(triple.object.value)
    for predicate in ("and", "or", "xone"):
        for triple in graph:
            if triple.predicate != f"{SH}{predicate}":
                continue
            for member in _rdf_list(triple.object, graph) or ():
                if isinstance(member, Iri):
                    referenced.append(member.value)
    seen: list[str] = []
    for subject in declared + referenced:
        if subject not in seen:
            seen.append(subject)
    return seen


def parse_shapes(graph: Graph) -> tuple[NodeShape, ...]:
    """Read node shapes and their property shapes out of a shapes graph."""
    shapes: list[NodeShape] = []
    for subject in _shape_ids(graph):
        properties: list[PropertyShape] = []
        # Node-level parameters, scanned so `sh:sparql`, `sh:and`, `sh:not`, `sh:node` and the rest
        # are reported rather than passing unnoticed.
        unsupported: list[str] = [
            _local(triple.predicate)
            for triple in graph
            if triple.subject == subject
            and triple.predicate.startswith(SH)
            and _local(triple.predicate) not in SUPPORTED
        ]
        for entry in _values(graph, subject, f"{SH}property"):
            if not isinstance(entry, Iri):
                unsupported.append("property (inline shape)")
                continue
            node = entry.value
            parsed_path = _parse_path(graph.value(node, f"{SH}path"), graph)
            if parsed_path is None:
                # A path form this engine does not implement -- `zeroOrMorePath` and friends.
                # Recorded, because skipping one silently would leave the report claiming a
                # constraint held when it was never evaluated.
                unsupported.append("path (unsupported form)")
                continue
            parameters: dict[str, list[Any]] = {}
            for triple in graph:
                if triple.subject != node or not triple.predicate.startswith(SH):
                    continue
                parameters.setdefault(_local(triple.predicate), []).append(triple.object)
            # `sh:in` takes an rdf:List, and that list lives in the *shapes* graph -- so it is
            # expanded here, where that graph is in hand, rather than at evaluation time where
            # only the data graph is. Left unexpanded, the list head becomes the sole permitted
            # value and every real value is reported as not permitted.
            if "in" in parameters:
                parameters["in"] = _permitted_values(parameters["in"], graph)
            severity_value = graph.value(node, f"{SH}severity")
            properties.append(
                PropertyShape(
                    path=parsed_path,
                    parameters=parameters,
                    severity=_SEVERITY.get(
                        severity_value.value if isinstance(severity_value, Iri) else "",
                        "violation",
                    ),
                    message=graph.literal(node, f"{SH}message"),
                )
            )
        shapes.append(
            NodeShape(
                id=subject,
                target_classes=tuple(
                    v.value
                    for v in _values(graph, subject, f"{SH}targetClass")
                    if isinstance(v, Iri)
                ),
                target_nodes=tuple(
                    v.value
                    for v in _values(graph, subject, f"{SH}targetNode")
                    if isinstance(v, Iri)
                ),
                target_subjects_of=tuple(
                    v.value
                    for v in _values(graph, subject, f"{SH}targetSubjectsOf")
                    if isinstance(v, Iri)
                ),
                target_objects_of=tuple(
                    v.value
                    for v in _values(graph, subject, f"{SH}targetObjectsOf")
                    if isinstance(v, Iri)
                ),
                properties=tuple(properties),
                deactivated=(graph.literal(subject, f"{SH}deactivated") or "").lower() == "true",
                unsupported=tuple(sorted(set(unsupported))),
            )
        )
    return tuple(shapes)


def _focus_nodes(data: Graph, shape: NodeShape) -> list[str]:
    found: list[str] = []
    for target in shape.target_classes:
        found.extend(data.subjects_of_type(target))
    found.extend(shape.target_nodes)
    for predicate in shape.target_subjects_of:
        found.extend(t.subject for t in data if t.predicate == predicate)
    for predicate in shape.target_objects_of:
        found.extend(
            t.object.value for t in data if t.predicate == predicate and isinstance(t.object, Iri)
        )
    # Stable and unique: a node targeted twice is one focus node, and report order must not depend
    # on which rule reached it first.
    return sorted(dict.fromkeys(found))


def _numeric(value: Any) -> float | None:
    try:
        return float(_plain(value))
    except (TypeError, ValueError):
        return None


def _check(
    shape: PropertyShape,
    focus: str,
    values: Sequence[Any],
    data: Graph,
    index: dict[str, NodeShape] | None = None,
    depth: int = 0,
) -> list[tuple[str, str]]:
    """Evaluate one property shape. Returns ``(constraint, message)`` for each failure."""
    failures: list[tuple[str, str]] = []
    parameters = shape.parameters

    def first(name: str) -> Any | None:
        entries = parameters.get(name)
        return entries[0] if entries else None

    count = len(values)
    minimum = first("minCount")
    if minimum is not None and count < int(_plain(minimum)):
        failures.append(("minCount", f"expected at least {_plain(minimum)}, found {count}"))
    maximum = first("maxCount")
    if maximum is not None and count > int(_plain(maximum)):
        failures.append(("maxCount", f"expected at most {_plain(maximum)}, found {count}"))

    datatype = first("datatype")
    if datatype is not None and isinstance(datatype, Iri):
        for value in values:
            if not isinstance(value, Literal):
                failures.append(("datatype", f"{_plain(value)!r} is not a literal"))
            elif (value.datatype or f"{XSD}string") != datatype.value:
                failures.append(
                    (
                        "datatype",
                        f"{value.value!r} is {value.datatype or 'plain'}, "
                        f"expected {_local(datatype.value)}",
                    )
                )

    expected_class = first("class")
    if expected_class is not None and isinstance(expected_class, Iri):
        for value in values:
            if not isinstance(value, Iri):
                failures.append(("class", f"{_plain(value)!r} is a literal, expected a node"))
            elif expected_class.value not in {
                t.object.value
                for t in data
                if t.subject == value.value
                and t.predicate == f"{NS.rdf}type"
                and isinstance(t.object, Iri)
            }:
                failures.append(("class", f"{value.value} is not a {expected_class.value}"))

    kind = first("nodeKind")
    if kind is not None and isinstance(kind, Iri):
        wants_iri = kind.value.endswith("IRI")
        wants_literal = kind.value.endswith("Literal")
        for value in values:
            if wants_iri and not isinstance(value, Iri):
                failures.append(("nodeKind", f"{_plain(value)!r} is not an IRI"))
            if wants_literal and not isinstance(value, Literal):
                failures.append(("nodeKind", f"{_plain(value)!r} is not a literal"))

    allowed = parameters.get("in")
    if allowed:
        # Already expanded at parse time, where the shapes graph was in hand.
        permitted = {_plain(entry) for entry in allowed}
        for value in values:
            if _plain(value) not in permitted:
                failures.append(("in", f"{_plain(value)!r} is not one of {sorted(permitted)}"))

    required = first("hasValue")
    if required is not None and _plain(required) not in {_plain(v) for v in values}:
        failures.append(("hasValue", f"{_plain(required)!r} is required and absent"))

    pattern = first("pattern")
    if pattern is not None:
        flags = re.IGNORECASE if "i" in str(_plain(first("flags")) or "") else 0
        compiled = re.compile(str(_plain(pattern)), flags)
        for value in values:
            if not compiled.search(str(_plain(value))):
                failures.append(
                    ("pattern", f"{_plain(value)!r} does not match {_plain(pattern)!r}")
                )

    for name, test, describe in (
        ("minLength", lambda text, n: len(text) >= n, "shorter than"),
        ("maxLength", lambda text, n: len(text) <= n, "longer than"),
    ):
        bound = first(name)
        if bound is not None:
            limit = int(_plain(bound))
            for value in values:
                if not test(str(_plain(value)), limit):
                    failures.append((name, f"{_plain(value)!r} is {describe} {limit}"))

    for name, compare, describe in (
        ("minInclusive", lambda a, b: a >= b, "below"),
        ("maxInclusive", lambda a, b: a <= b, "above"),
        ("minExclusive", lambda a, b: a > b, "not above"),
        ("maxExclusive", lambda a, b: a < b, "not below"),
    ):
        bound = first(name)
        if bound is None:
            continue
        limit = _numeric(bound)
        for value in values:
            actual = _numeric(value)
            if actual is None:
                failures.append((name, f"{_plain(value)!r} is not a number"))
            elif limit is not None and not compare(actual, limit):
                failures.append((name, f"{actual} is {describe} {limit}"))

    # -- logical constraints -------------------------------------------------------------------
    #
    # Each runs another shape against each value. They are evaluated last so a value that already
    # failed a cheap cardinality or datatype test is not also reported against a nested shape,
    # which turns one mistake into a page of findings.
    shapes = index or {}

    def holds(value: Any, shape_iri: str) -> bool | None:
        """Whether a value conforms to a named shape; ``None`` when it cannot be judged."""
        target = shapes.get(shape_iri)
        if target is None or not isinstance(value, Iri):
            return None
        return _shape_holds(data, target, value.value, shapes)

    node = first("node")
    if node is not None and isinstance(node, Iri):
        for value in values:
            verdict = holds(value, node.value)
            if verdict is False:
                failures.append(("node", f"{_plain(value)!r} does not conform to {node.value}"))

    negated = first("not")
    if negated is not None and isinstance(negated, Iri):
        for value in values:
            verdict = holds(value, negated.value)
            if verdict is True:
                failures.append(
                    ("not", f"{_plain(value)!r} conforms to {negated.value} and must not")
                )

    for name, wanted in (("and", "all"), ("or", "any"), ("xone", "exactly one")):
        entry = first(name)
        if entry is None:
            continue
        members = [m.value for m in (_rdf_list(entry, data) or []) if isinstance(m, Iri)]
        if not members:
            # The list lives in the shapes graph; when it is not reachable from the data graph
            # there is nothing to run, and claiming the constraint passed would be a lie.
            continue
        for value in values:
            verdicts = [holds(value, member) for member in members]
            judged = [v for v in verdicts if v is not None]
            if not judged:
                continue
            count = sum(1 for v in judged if v)
            satisfied = (
                count == len(judged)
                if name == "and"
                else count >= 1
                if name == "or"
                else count == 1
            )
            if not satisfied:
                failures.append(
                    (name, f"{_plain(value)!r} satisfies {count} of {len(judged)}, needs {wanted}")
                )

    qualified = first("qualifiedValueShape")
    if qualified is not None and isinstance(qualified, Iri):
        conforming = sum(1 for value in values if holds(value, qualified.value) is True)
        minimum = first("qualifiedMinCount")
        maximum = first("qualifiedMaxCount")
        if minimum is not None and conforming < int(_plain(minimum)):
            failures.append(
                (
                    "qualifiedMinCount",
                    f"{conforming} value(s) conform to {qualified.value}, "
                    f"expected at least {_plain(minimum)}",
                )
            )
        if maximum is not None and conforming > int(_plain(maximum)):
            failures.append(
                (
                    "qualifiedMaxCount",
                    f"{conforming} value(s) conform to {qualified.value}, "
                    f"expected at most {_plain(maximum)}",
                )
            )

    languages = parameters.get("languageIn")
    if languages:
        permitted = {str(_plain(entry)) for entry in languages}
        for value in values:
            tag = getattr(value, "language", None)
            if tag not in permitted:
                failures.append(
                    ("languageIn", f"language {tag!r} is not one of {sorted(permitted)}")
                )

    return failures


def _rdf_list(head: Any, graph: Graph) -> list[Any] | None:
    """Expand an ``rdf:List``, or ``None`` if this is not one.

    ``sh:in`` takes a list, not a repeated predicate. Treating the list *head* as the permitted
    value makes every real value fail the constraint -- a shape that permits "internal" would
    report "internal" as not permitted, which reads like a data problem and is a reader problem.
    """
    if not isinstance(head, Iri):
        return None
    values: list[Any] = []
    node: str | None = head.value
    seen: set[str] = set()
    while node and node != f"{NS.rdf}nil":
        if node in seen:  # a cyclic list is malformed; stop rather than spin
            return values
        seen.add(node)
        first = graph.value(node, f"{NS.rdf}first")
        if first is None:
            return values or None
        values.append(first)
        rest = graph.value(node, f"{NS.rdf}rest")
        node = rest.value if isinstance(rest, Iri) else None
    return values


def _permitted_values(entries: Sequence[Any], graph: Graph) -> list[Any]:
    """The values ``sh:in`` allows, whether written as an rdf:List or a repeated predicate."""
    permitted: list[Any] = []
    for entry in entries:
        expanded = _rdf_list(entry, graph)
        permitted.extend([entry] if expanded is None else expanded)
    return permitted


def unsupported_parameters(shapes: Iterable[NodeShape]) -> tuple[str, ...]:
    """Constraint parameters present on the shapes that this engine does not evaluate."""
    found: set[str] = set()
    for shape in shapes:
        found.update(shape.unsupported)
        for property_shape in shape.properties:
            found.update(name for name in property_shape.parameters if name not in SUPPORTED)
    return tuple(sorted(found))


def _shape_holds(data: Graph, shape: NodeShape, focus: str, index: dict[str, NodeShape]) -> bool:
    """Whether one node satisfies one shape, with nothing reported.

    The building block every logical constraint needs. `sh:not` cares only *whether* the inner
    shape held, and reporting its internals would fill a report with failures that are the point
    rather than the problem.
    """
    return not _evaluate(data, shape, focus, index)


def _evaluate(
    data: Graph, shape: NodeShape, focus: str, index: dict[str, NodeShape], depth: int = 0
) -> list[Result]:
    """Run every property shape of one node shape against one focus node."""
    if depth > 12:
        # A shapes graph can reference itself. Bounded rather than recursive-until-death, because
        # a cyclic shape is a broken file and must not take the process with it.
        return []
    found: list[Result] = []
    for property_shape in shape.properties:
        values = property_shape.path.reach(data, focus)
        for constraint, detail in _check(property_shape, focus, values, data, index, depth):
            found.append(
                Result(
                    focus=focus,
                    path=property_shape.path.describe(),
                    constraint=constraint,
                    severity=property_shape.severity,
                    message=property_shape.message or detail,
                )
            )
    return found


def validate(data: Graph, shapes_graph: Graph) -> ShaclReport:
    """Run a shapes graph against a data graph."""
    shapes = parse_shapes(shapes_graph)
    index = {shape.id: shape for shape in shapes}
    results: list[Result] = []
    focus_count = 0

    for shape in shapes:
        if shape.deactivated:
            continue
        for focus in _focus_nodes(data, shape):
            focus_count += 1
            results.extend(_evaluate(data, shape, focus, index))

    return ShaclReport(
        results=tuple(results),
        unsupported=unsupported_parameters(shapes),
        shapes_run=len([s for s in shapes if not s.deactivated]),
        focus_nodes=focus_count,
    )
