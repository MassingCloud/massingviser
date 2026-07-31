"""SHACL Core, enough of it to run the shapes ISO 21597 publishes.

The previous position here was that a SHACL engine is a large dependency for failures that are
almost always mundane. That is still true of a *complete* engine -- SPARQL-based constraints alone
would mean an entire query language. What is tractable, and what the published container shapes
actually use, is **SHACL Core's constraint components**: cardinality, datatype, class, node kind,
value ranges, string tests, and logical combinations over property paths.

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
class PropertyShape:
    path: str
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


def parse_shapes(graph: Graph) -> tuple[NodeShape, ...]:
    """Read node shapes and their property shapes out of a shapes graph."""
    shapes: list[NodeShape] = []
    for subject in graph.subjects_of_type(f"{SH}NodeShape"):
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
            path = graph.value(node, f"{SH}path")
            if not isinstance(path, Iri):
                # Property paths beyond a single predicate (sequences, alternatives, inverse) are
                # a language of their own. Recorded, because skipping one silently would leave the
                # report claiming a constraint held when it was never evaluated.
                unsupported.append("path (not a single predicate)")
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
                    path=path.value,
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
    shape: PropertyShape, focus: str, values: Sequence[Any], data: Graph
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


def validate(data: Graph, shapes_graph: Graph) -> ShaclReport:
    """Run a shapes graph against a data graph."""
    shapes = parse_shapes(shapes_graph)
    results: list[Result] = []
    focus_count = 0

    for shape in shapes:
        if shape.deactivated:
            continue
        for focus in _focus_nodes(data, shape):
            focus_count += 1
            for property_shape in shape.properties:
                values = data.objects(focus, property_shape.path)
                for constraint, detail in _check(property_shape, focus, values, data):
                    results.append(
                        Result(
                            focus=focus,
                            path=property_shape.path,
                            constraint=constraint,
                            severity=property_shape.severity,
                            message=property_shape.message or detail,
                        )
                    )

    return ShaclReport(
        results=tuple(results),
        unsupported=unsupported_parameters(shapes),
        shapes_run=len([s for s in shapes if not s.deactivated]),
        focus_nodes=focus_count,
    )
