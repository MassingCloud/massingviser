"""A small RDF/XML reader and writer.

Enough of RDF/XML to write and read ICDD containers, and no more. A general RDF library would be a
substantial dependency for a format whose ICDD usage is a few dozen triple shapes -- and taking one
would put a parser the platform does not control on the path of every container it opens.

Two refusals are deliberate:

- **DTDs are rejected**, not ignored. A DOCTYPE in an untrusted container is the entry point for
  entity expansion and external-entity attacks, and a container is by definition a file somebody
  else sent you.
- **``rdf:parseType`` is rejected** rather than mis-parsed. ``Collection``, ``Resource`` and
  ``Literal`` each change the meaning of the element's children; silently treating one as ordinary
  markup produces a graph that is confidently wrong.

Turtle and JSON-LD are out of scope, which is stated rather than discovered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence
from xml.etree import ElementTree

from .ontology import DEFAULT_PREFIXES, NS


class RdfError(Exception):
    """Raised for input this reader refuses rather than guesses at."""


@dataclass(frozen=True)
class Literal:
    value: str
    datatype: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class Iri:
    value: str


Node = "Iri | Literal"


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: Any  # Iri | Literal


class Graph:
    """An unindexed triple set with the two lookups ICDD actually needs."""

    __slots__ = ("_triples",)

    def __init__(self, triples: Iterable[Triple] = ()) -> None:
        self._triples: list[Triple] = list(triples)

    def __len__(self) -> int:
        return len(self._triples)

    def __iter__(self) -> Iterator[Triple]:
        return iter(self._triples)

    def add(self, subject: str, predicate: str, object: Any) -> None:
        self._triples.append(Triple(subject, predicate, object))

    def add_literal(self, subject: str, predicate: str, value: Any) -> None:
        if value is None:
            return  # an absent value is an absent triple, not an empty one
        self.add(subject, predicate, Literal(str(value)))

    def objects(self, subject: str, predicate: str) -> list[Any]:
        return [t.object for t in self._triples if t.subject == subject and t.predicate == predicate]

    def value(self, subject: str, predicate: str) -> Any | None:
        found = self.objects(subject, predicate)
        return found[0] if found else None

    def literal(self, subject: str, predicate: str) -> str | None:
        found = self.value(subject, predicate)
        return found.value if isinstance(found, Literal) else None

    def subjects_of_type(self, type_iri: str) -> list[str]:
        return [
            t.subject
            for t in self._triples
            if t.predicate == f"{NS.rdf}type"
            and isinstance(t.object, Iri)
            and t.object.value == type_iri
        ]

    def type_of(self, subject: str) -> str | None:
        found = self.value(subject, f"{NS.rdf}type")
        return found.value if isinstance(found, Iri) else None

    def subjects(self) -> list[str]:
        seen: dict[str, None] = {}
        for triple in self._triples:
            seen.setdefault(triple.subject, None)
        return list(seen)


_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def _escape(text: str) -> str:
    return "".join(_ESCAPES.get(character, character) for character in text)


def _split(iri: str) -> tuple[str, str]:
    """Split an IRI into namespace and local name at the last ``#`` or ``/``."""
    index = max(iri.rfind("#"), iri.rfind("/"))
    if index < 0:
        raise RdfError(f'Cannot split IRI "{iri}" into a namespace and a local name.')
    return iri[: index + 1], iri[index + 1 :]


def serialise(graph: Graph, *, prefixes: Mapping[str, str] | None = None) -> str:
    """Write a graph as RDF/XML, one ``rdf:Description`` per subject."""
    bindings = dict(prefixes or DEFAULT_PREFIXES)
    reverse = {namespace: prefix for prefix, namespace in bindings.items()}

    def qname(iri: str) -> str:
        namespace, local = _split(iri)
        prefix = reverse.get(namespace)
        if prefix is None:
            # Bind on demand rather than emitting a raw IRI, which RDF/XML cannot express as an
            # element name.
            prefix = f"ns{len(bindings)}"
            bindings[prefix] = namespace
            reverse[namespace] = prefix
        return f"{prefix}:{local}"

    body: list[str] = []
    for subject in graph.subjects():
        triples = [t for t in graph if t.subject == subject]
        type_iri = next(
            (
                t.object.value
                for t in triples
                if t.predicate == f"{NS.rdf}type" and isinstance(t.object, Iri)
            ),
            None,
        )
        element = qname(type_iri) if type_iri else "rdf:Description"
        body.append(f'  <{element} rdf:about="{_escape(subject)}">')
        for triple in triples:
            if triple.predicate == f"{NS.rdf}type":
                continue
            name = qname(triple.predicate)
            if isinstance(triple.object, Iri):
                body.append(f'    <{name} rdf:resource="{_escape(triple.object.value)}"/>')
            else:
                attributes = ""
                if triple.object.datatype:
                    attributes += f' rdf:datatype="{_escape(triple.object.datatype)}"'
                if triple.object.language:
                    attributes += f' xml:lang="{_escape(triple.object.language)}"'
                body.append(f"    <{name}{attributes}>{_escape(triple.object.value)}</{name}>")
        body.append(f"  </{element}>")

    declarations = "\n    ".join(
        f'xmlns:{prefix}="{_escape(namespace)}"' for prefix, namespace in bindings.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<rdf:RDF\n    {declarations}>\n" + "\n".join(body) + "\n</rdf:RDF>\n"
    )


_DOCTYPE = re.compile(rb"<!DOCTYPE", re.IGNORECASE)


def parse(source: str | bytes) -> Graph:
    """Read RDF/XML into a graph, refusing the constructs this reader will not honour."""
    raw = source.encode("utf-8") if isinstance(source, str) else source
    if _DOCTYPE.search(raw):
        raise RdfError(
            "This RDF/XML declares a DTD. A container is a file somebody else sent you, and a "
            "DOCTYPE is the entry point for entity expansion; it is refused rather than ignored."
        )

    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as thrown:
        raise RdfError(f"Malformed RDF/XML: {thrown}") from thrown

    graph = Graph()
    rdf_about = f"{{{NS.rdf}}}about"
    rdf_resource = f"{{{NS.rdf}}}resource"
    rdf_datatype = f"{{{NS.rdf}}}datatype"
    rdf_parse_type = f"{{{NS.rdf}}}parseType"
    rdf_type = f"{NS.rdf}type"

    def expand(tag: str) -> str:
        # ElementTree gives "{namespace}local"; ICDD IRIs are namespace + local with no separator.
        if tag.startswith("{"):
            namespace, local = tag[1:].split("}", 1)
            return f"{namespace}{local}"
        return tag

    for node in root:
        if rdf_parse_type in node.attrib:
            raise RdfError(
                f'rdf:parseType="{node.attrib[rdf_parse_type]}" changes what the children mean; '
                "this reader refuses it rather than parsing it as ordinary markup."
            )
        subject = node.attrib.get(rdf_about)
        if subject is None:
            continue  # blank nodes are not used by ICDD containers

        node_type = expand(node.tag)
        if node_type != f"{NS.rdf}Description":
            graph.add(subject, rdf_type, Iri(node_type))

        for child in node:
            if rdf_parse_type in child.attrib:
                raise RdfError(
                    f'rdf:parseType="{child.attrib[rdf_parse_type]}" is not supported.'
                )
            predicate = expand(child.tag)
            resource = child.attrib.get(rdf_resource)
            if resource is not None:
                graph.add(subject, predicate, Iri(resource))
                continue
            text = (child.text or "").strip()
            graph.add(
                subject,
                predicate,
                Literal(
                    text,
                    datatype=child.attrib.get(rdf_datatype),
                    language=child.attrib.get("{http://www.w3.org/XML/1998/namespace}lang"),
                ),
            )
    return graph
