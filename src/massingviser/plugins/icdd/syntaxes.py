"""Turtle and JSON-LD, alongside the RDF/XML the standard mandates.

ISO 21597 specifies RDF/XML for the container index, and that stays the format written by default.
These two exist because RDF/XML is the format nobody reads: a reviewer asked to check a linkset by
eye can read Turtle, and a web client consuming a container wants JSON. Both are exact round trips
of the same triple set, so nothing is lost by looking at a container through either.

**The Turtle reader is a deliberate subset.** It handles IRIs, prefixed names, the ``a`` keyword,
plain and typed and language-tagged literals, and the ``;`` and ``,`` continuations. It does *not*
handle blank nodes, collections, or nested bracket syntax -- and it **refuses** them rather than
skipping them. A parser that silently drops a construct it does not understand turns a linkset that
asserts something into a linkset that asserts less, and nothing about the file says so.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .ontology import NS
from .rdf import Graph, Iri, Literal, RdfError

#: Constructs a real Turtle document may contain that this reader will not guess at.
_UNSUPPORTED = {
    "[": "blank node",
    "]": "blank node",
    "(": "collection",
    ")": "collection",
}

_PREFIX_LINE = re.compile(r"^@?prefix\s+([A-Za-z][\w.-]*)?:\s*<([^>]*)>\s*\.?$", re.IGNORECASE)
_BASE_LINE = re.compile(r"^@?base\s+<([^>]*)>\s*\.?$", re.IGNORECASE)

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _shorten(iri: str, prefixes: Mapping[str, str]) -> str:
    for prefix, namespace in prefixes.items():
        if iri.startswith(namespace):
            local = iri[len(namespace) :]
            # Only when the remainder is a legal local name; otherwise the full IRI is written.
            if local and re.fullmatch(r"[A-Za-z_][\w.-]*", local):
                return f"{prefix}:{local}"
    return f"<{iri}>"


def _turtle_literal(value: Literal) -> str:
    escaped = (
        str(value.value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    text = f'"{escaped}"'
    datatype = getattr(value, "datatype", None)
    language = getattr(value, "language", None)
    if language:
        return f"{text}@{language}"
    if datatype:
        return f"{text}^^<{datatype}>"
    return text


def to_turtle(graph: Graph, *, prefixes: Mapping[str, str] | None = None) -> str:
    """Serialise, grouping by subject so the output reads as objects rather than as a triple dump."""
    bindings = dict(prefixes or {})
    lines = [f"@prefix {prefix}: <{namespace}> ." for prefix, namespace in bindings.items()]
    if lines:
        lines.append("")

    grouped: dict[str, list[tuple[str, object]]] = {}
    for triple in graph:
        grouped.setdefault(triple.subject, []).append((triple.predicate, triple.object))

    for subject, pairs in grouped.items():
        statements = []
        for predicate, object_ in pairs:
            # `a` is Turtle's own shorthand for rdf:type and is what makes it readable.
            name = "a" if predicate == f"{NS.rdf}type" else _shorten(predicate, bindings)
            if isinstance(object_, Iri):
                rendered = _shorten(object_.value, bindings)
            elif isinstance(object_, Literal):
                rendered = _turtle_literal(object_)
            else:
                rendered = _turtle_literal(Literal(str(object_)))
            statements.append(f"    {name} {rendered}")
        lines.append(_shorten(subject, bindings))
        lines.append(" ;\n".join(statements) + " .")
        lines.append("")
    return "\n".join(lines)


def _tokenise(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r\n":
            index += 1
            continue
        if char == "#":
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == "<":
            end = text.find(">", index)
            if end == -1:
                raise RdfError("An IRI is opened and never closed.")
            tokens.append(text[index : end + 1])
            index = end + 1
            continue
        if char == '"':
            index += 1
            buffer = []
            while index < length and text[index] != '"':
                if text[index] == "\\" and index + 1 < length:
                    escape = text[index + 1]
                    buffer.append(
                        {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(escape, escape)
                    )
                    index += 2
                    continue
                buffer.append(text[index])
                index += 1
            if index >= length:
                raise RdfError("A literal is opened and never closed.")
            index += 1
            suffix = ""
            if text.startswith("^^", index):
                end = text.find(">", index)
                if end == -1:
                    raise RdfError("A datatype is opened and never closed.")
                suffix = text[index : end + 1]
                index = end + 1
            elif index < length and text[index] == "@":
                start = index
                index += 1
                while index < length and (text[index].isalnum() or text[index] == "-"):
                    index += 1
                suffix = text[start:index]
            tokens.append('"' + "".join(buffer) + '"' + suffix)
            continue
        if char in ".;,":
            tokens.append(char)
            index += 1
            continue
        start = index
        while index < length and text[index] not in " \t\r\n.;,#":
            index += 1
        tokens.append(text[start:index])
    return tokens


def _expand(token: str, prefixes: Mapping[str, str]) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if ":" in token:
        prefix, _, local = token.partition(":")
        if prefix in prefixes:
            return f"{prefixes[prefix]}{local}"
        raise RdfError(f'Unknown prefix "{prefix}:".')
    raise RdfError(f'"{token}" is neither an IRI nor a prefixed name.')


def _object(token: str, prefixes: Mapping[str, str]) -> object:
    if token.startswith('"'):
        closing = token.rindex('"')
        value = token[1:closing]
        rest = token[closing + 1 :]
        if rest.startswith("^^"):
            return Literal(value, datatype=rest[3:-1])
        if rest.startswith("@"):
            return Literal(value, language=rest[1:])
        return Literal(value)
    if _NUMERIC.fullmatch(token):
        return Literal(token)
    if token in ("true", "false"):
        return Literal(token)
    return Iri(_expand(token, prefixes))


def from_turtle(source: str | bytes) -> Graph:
    """Read the subset this module writes, and refuse the rest by name."""
    text = source.decode("utf-8") if isinstance(source, (bytes, bytearray)) else source

    prefixes: dict[str, str] = {}
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = _PREFIX_LINE.match(stripped)
        if match:
            prefixes[match.group(1) or ""] = match.group(2)
            continue
        if _BASE_LINE.match(stripped):
            continue
        body.append(line)

    joined = "\n".join(body)
    for symbol, name in _UNSUPPORTED.items():
        if symbol in joined:
            raise RdfError(
                f"This reader does not support {name} syntax, and refuses it rather than "
                "dropping what it cannot represent."
            )

    graph = Graph()
    tokens = _tokenise(joined)
    index = 0
    subject: str | None = None
    predicate: str | None = None

    while index < len(tokens):
        token = tokens[index]
        if token == ".":
            subject = predicate = None
            index += 1
            continue
        if token == ";":
            predicate = None
            index += 1
            continue
        if token == ",":
            index += 1
            continue
        if subject is None:
            subject = _expand(token, prefixes)
            index += 1
            continue
        if predicate is None:
            predicate = f"{NS.rdf}type" if token == "a" else _expand(token, prefixes)
            index += 1
            continue
        graph.add(subject, predicate, _object(token, prefixes))
        index += 1
    return graph


def to_jsonld(graph: Graph, *, prefixes: Mapping[str, str] | None = None) -> str:
    """A JSON-LD document: one node object per subject, under a context of the prefixes."""
    bindings = dict(prefixes or {})
    nodes: dict[str, dict[str, object]] = {}

    for triple in graph:
        node = nodes.setdefault(triple.subject, {"@id": triple.subject})
        if triple.predicate == f"{NS.rdf}type" and isinstance(triple.object, Iri):
            node.setdefault("@type", []).append(triple.object.value)  # type: ignore[union-attr]
            continue
        if isinstance(triple.object, Iri):
            entry: object = {"@id": triple.object.value}
        elif isinstance(triple.object, Literal):
            datatype = getattr(triple.object, "datatype", None)
            language = getattr(triple.object, "language", None)
            if datatype:
                entry = {"@value": triple.object.value, "@type": datatype}
            elif language:
                entry = {"@value": triple.object.value, "@language": language}
            else:
                entry = {"@value": triple.object.value}
        else:
            entry = {"@value": str(triple.object)}
        node.setdefault(triple.predicate, []).append(entry)  # type: ignore[union-attr]

    document: dict[str, object] = {"@graph": list(nodes.values())}
    if bindings:
        document["@context"] = dict(bindings)
    return json.dumps(document, indent=2)


def from_jsonld(source: str | bytes) -> Graph:
    text = source.decode("utf-8") if isinstance(source, (bytes, bytearray)) else source
    try:
        document = json.loads(text)
    except ValueError as thrown:
        raise RdfError(f"Not valid JSON: {thrown}") from thrown

    context = document.get("@context", {}) if isinstance(document, dict) else {}
    nodes = document.get("@graph", document) if isinstance(document, dict) else document
    if isinstance(nodes, dict):
        nodes = [nodes]
    if not isinstance(nodes, list):
        raise RdfError("Expected a @graph array or a single node object.")

    def resolve(term: str) -> str:
        prefix, separator, local = term.partition(":")
        if separator and prefix in context and not local.startswith("//"):
            return f"{context[prefix]}{local}"
        return term

    graph = Graph()
    for node in nodes:
        if not isinstance(node, dict):
            raise RdfError("A node in @graph is not an object.")
        subject = node.get("@id")
        if not subject:
            # No blank nodes, for the same reason the Turtle reader refuses them: there is nothing
            # honest to key the resulting triples on.
            raise RdfError("A node object without @id cannot be represented.")
        subject = resolve(str(subject))
        for kind in (
            node.get("@type", [])
            if isinstance(node.get("@type"), list)
            else ([node["@type"]] if "@type" in node else [])
        ):
            graph.add(subject, f"{NS.rdf}type", Iri(resolve(str(kind))))
        for key, values in node.items():
            if key.startswith("@"):
                continue
            predicate = resolve(key)
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, dict) and "@id" in value:
                    graph.add(subject, predicate, Iri(resolve(str(value["@id"]))))
                elif isinstance(value, dict):
                    graph.add(
                        subject,
                        predicate,
                        Literal(
                            str(value.get("@value", "")),
                            datatype=value.get("@type"),
                            language=value.get("@language"),
                        ),
                    )
                else:
                    graph.add(subject, predicate, Literal(str(value)))
    return graph


#: What a container's graphs can be written as. ``rdfxml`` is what ISO 21597 mandates and stays the
#: default; the other two exist to be read by people and by web clients respectively.
SYNTAXES = ("rdfxml", "turtle", "jsonld")


def dump(graph: Graph, syntax: str = "rdfxml", *, prefixes: Mapping[str, str] | None = None) -> str:
    """Serialise a graph in any supported syntax."""
    if syntax == "turtle":
        return to_turtle(graph, prefixes=prefixes)
    if syntax == "jsonld":
        return to_jsonld(graph, prefixes=prefixes)
    if syntax == "rdfxml":
        from .rdf import serialise

        return serialise(graph, prefixes=prefixes)
    raise RdfError(f'Unknown syntax "{syntax}". Known: {", ".join(SYNTAXES)}.')


def load(source: str | bytes, syntax: str = "rdfxml") -> Graph:
    """Read a graph from any supported syntax."""
    if syntax == "turtle":
        return from_turtle(source)
    if syntax == "jsonld":
        return from_jsonld(source)
    if syntax == "rdfxml":
        from .rdf import parse

        return parse(source)
    raise RdfError(f'Unknown syntax "{syntax}". Known: {", ".join(SYNTAXES)}.')
