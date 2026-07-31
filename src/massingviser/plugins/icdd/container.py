"""ICDD containers: assembly, linking and structural validation.

ZIP is a **port**, not a dependency. ``ContainerArchive`` abstracts entry access so the host
supplies compression -- ``zipfile``, streaming from object storage, or nothing at all in a test.
Taking a ZIP dependency here would force that choice on every deployment.

Validation is **structural, not SHACL**. A full SHACL engine is a large dependency for failures
that are almost always mundane: a payload file that never got written, a link to a renamed
document. Those are caught here cheaply, with the offending file named, and the parsed graphs are
exposed so a host that wants the published SHACL shapes can run them itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, runtime_checkable

from .ontology import (
    CONTAINER_LAYOUT,
    CT,
    LINK_TYPES,
    LS,
    NS,
    LinkTypeDescriptor,
)
from .rdf import Graph, Iri, RdfError, parse, serialise

DocumentKind = Literal["internal", "external", "folder"]
PartyKind = Literal["Person", "Organisation"]
IdentifierKind = Literal["string", "uri", "query"]


@runtime_checkable
class ContainerArchive(Protocol):
    """Entry access. The host decides what the bytes are stored in."""

    def read(self, path: str) -> bytes | None: ...
    def write(self, path: str, data: bytes) -> None: ...
    def entries(self) -> Sequence[str]: ...


class MemoryArchive:
    """An in-memory archive, which is also exactly what a test wants."""

    __slots__ = ("_entries",)

    def __init__(self, entries: Mapping[str, bytes] | None = None) -> None:
        self._entries: dict[str, bytes] = dict(entries or {})

    def read(self, path: str) -> bytes | None:
        return self._entries.get(path)

    def write(self, path: str, data: bytes) -> None:
        self._entries[path] = data

    def entries(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


@dataclass(frozen=True)
class Party:
    id: str
    name: str
    kind: PartyKind = "Organisation"


@dataclass(frozen=True)
class Document:
    id: str
    name: str
    kind: DocumentKind = "internal"
    #: Required for internal documents -- it is the path inside the payload folder.
    filename: str | None = None
    #: Required for folder documents.
    foldername: str | None = None
    #: Required for external documents.
    url: str | None = None
    filetype: str | None = None
    format: str | None = None
    description: str | None = None
    version_id: str | None = None
    version_description: str | None = None
    creation_date: str | None = None
    created_by: str | None = None
    prior_version: str | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None

    @property
    def payload_path(self) -> str | None:
        if self.kind == "internal" and self.filename:
            return f"{CONTAINER_LAYOUT.payload_folder}/{self.filename}"
        return None


@dataclass(frozen=True)
class Identifier:
    """Addresses something *inside* a document, not the document as a whole."""

    kind: IdentifierKind
    value: str
    #: Which field the value is matched against, e.g. ``GlobalId``.
    field: str | None = None
    query_language: str | None = None


@dataclass(frozen=True)
class LinkElement:
    document_id: str
    identifier: Identifier | None = None


@dataclass(frozen=True)
class Link:
    type: str
    #: For a directed link these are the from/to sides. For a symmetric one they are simply the
    #: two ends, and direction carries no meaning.
    from_elements: tuple[LinkElement, ...] = ()
    to_elements: tuple[LinkElement, ...] = ()
    id: str | None = None

    @property
    def descriptor(self) -> LinkTypeDescriptor | None:
        return LINK_TYPES.get(self.type)


@dataclass(frozen=True)
class Linkset:
    id: str
    name: str
    filename: str
    links: tuple[Link, ...] = ()


@dataclass(frozen=True)
class ContainerDescription:
    id: str
    name: str
    conformance_indicator: str = "ICDD-Part1-Container"
    description: str | None = None
    created_by: str | None = None
    creation_date: str | None = None
    version_id: str | None = None


@dataclass(frozen=True)
class Container:
    description: ContainerDescription
    parties: tuple[Party, ...] = ()
    documents: tuple[Document, ...] = ()
    linksets: tuple[Linkset, ...] = ()


def invert_link(link: Link) -> Link:
    """Reverse a link's meaning.

    Swaps the **class and the endpoints together**. Swapping only the endpoints would leave
    ``HasPart`` asserting that a whole is part of its own component -- a statement that still
    validates and is exactly backwards.
    """
    descriptor = link.descriptor
    if descriptor is None:
        raise ValueError(f'Unknown link type "{link.type}".')
    if not descriptor.directed:
        # Symmetric links have no inverse class; swapping the ends says the same thing.
        return replace(link, from_elements=link.to_elements, to_elements=link.from_elements)
    assert descriptor.inverse is not None
    return replace(
        link,
        type=descriptor.inverse,
        from_elements=link.to_elements,
        to_elements=link.from_elements,
    )


# ---------------------------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------------------------


def _subject(container_id: str, kind: str, local: str) -> str:
    return f"urn:icdd:{container_id}:{kind}:{local}"


_DOCUMENT_CLASS = {
    "internal": CT.InternalDocument,
    "external": CT.ExternalDocument,
    "folder": CT.FolderDocument,
}


def build_index_graph(container: Container) -> Graph:
    graph = Graph()
    container_id = container.description.id
    subject = _subject(container_id, "container", container.description.id)

    graph.add(subject, f"{NS.rdf}type", Iri(CT.ContainerDescription))
    graph.add_literal(subject, CT.name, container.description.name)
    graph.add_literal(subject, CT.description, container.description.description)
    graph.add_literal(subject, CT.conformanceIndicator, container.description.conformance_indicator)
    graph.add_literal(subject, CT.creationDate, container.description.creation_date)
    graph.add_literal(subject, CT.versionID, container.description.version_id)

    for party in container.parties:
        party_subject = _subject(container_id, "party", party.id)
        graph.add(
            party_subject,
            f"{NS.rdf}type",
            Iri(CT.Person if party.kind == "Person" else CT.Organisation),
        )
        graph.add_literal(party_subject, CT.name, party.name)
        if container.description.created_by == party.id:
            graph.add(subject, CT.createdBy, Iri(party_subject))

    for document in container.documents:
        document_subject = _subject(container_id, "document", document.id)
        graph.add(document_subject, f"{NS.rdf}type", Iri(_DOCUMENT_CLASS[document.kind]))
        graph.add(subject, CT.containsDocument, Iri(document_subject))
        graph.add(document_subject, CT.belongsToContainer, Iri(subject))
        graph.add_literal(document_subject, CT.name, document.name)
        graph.add_literal(document_subject, CT.description, document.description)
        graph.add_literal(document_subject, CT.filename, document.filename)
        graph.add_literal(document_subject, CT.foldername, document.foldername)
        graph.add_literal(document_subject, CT.url, document.url)
        graph.add_literal(document_subject, CT.filetype, document.filetype)
        graph.add_literal(document_subject, CT.format, document.format)
        graph.add_literal(document_subject, CT.versionID, document.version_id)
        graph.add_literal(document_subject, CT.versionDescription, document.version_description)
        graph.add_literal(document_subject, CT.creationDate, document.creation_date)
        graph.add_literal(document_subject, CT.checksum, document.checksum)
        graph.add_literal(document_subject, CT.checksumAlgorithm, document.checksum_algorithm)
        if document.created_by:
            graph.add(
                document_subject,
                CT.createdBy,
                Iri(_subject(container_id, "party", document.created_by)),
            )
        if document.prior_version:
            graph.add(
                document_subject,
                CT.priorVersion,
                Iri(_subject(container_id, "document", document.prior_version)),
            )

    for linkset in container.linksets:
        linkset_subject = _subject(container_id, "linkset", linkset.id)
        graph.add(linkset_subject, f"{NS.rdf}type", Iri(CT.Linkset))
        graph.add(subject, CT.containsLinkset, Iri(linkset_subject))
        graph.add_literal(linkset_subject, CT.name, linkset.name)
        graph.add_literal(linkset_subject, CT.filename, linkset.filename)

    return graph


def build_linkset_graph(container_id: str, linkset: Linkset) -> Graph:
    graph = Graph()

    for index, link in enumerate(linkset.links):
        descriptor = link.descriptor
        if descriptor is None:
            raise ValueError(f'Unknown link type "{link.type}".')
        link_subject = _subject(container_id, "link", link.id or f"{linkset.id}-{index}")
        graph.add(link_subject, f"{NS.rdf}type", Iri(descriptor.iri))

        def emit(
            element: LinkElement,
            predicate: str,
            position: int,
            *,
            # Bound per iteration rather than captured. Every call happens inside this iteration,
            # so the capture is harmless -- but if it ever escaped, every link element in the file
            # would be minted under the last link's subject and the graph would be silently wrong.
            link_subject: str = link_subject,
        ) -> None:
            element_subject = f"{link_subject}:el{position}"
            graph.add(link_subject, predicate, Iri(element_subject))
            graph.add(element_subject, f"{NS.rdf}type", Iri(LS.LinkElement))
            graph.add(
                element_subject,
                LS.hasDocument,
                Iri(_subject(container_id, "document", element.document_id)),
            )
            if element.identifier is None:
                return
            identifier_subject = f"{element_subject}:id"
            identifier = element.identifier
            identifier_class = {
                "string": LS.StringBasedIdentifier,
                "uri": LS.URIBasedIdentifier,
                "query": LS.QueryBasedIdentifier,
            }[identifier.kind]
            graph.add(element_subject, LS.hasIdentifier, Iri(identifier_subject))
            graph.add(identifier_subject, f"{NS.rdf}type", Iri(identifier_class))
            if identifier.kind == "uri":
                graph.add_literal(identifier_subject, LS.uri, identifier.value)
            elif identifier.kind == "query":
                graph.add_literal(identifier_subject, LS.queryExpression, identifier.value)
                graph.add_literal(identifier_subject, LS.queryLanguage, identifier.query_language)
            else:
                graph.add_literal(identifier_subject, LS.identifier, identifier.value)
                graph.add_literal(identifier_subject, LS.identifierField, identifier.field)

        position = 0
        if descriptor.directed:
            # Direction-aware serialisation: a directed link uses hasFromLinkElement and
            # hasToLinkElement, which is what carries the asymmetry into the graph.
            for element in link.from_elements:
                emit(element, LS.hasFromLinkElement, position)
                position += 1
            for element in link.to_elements:
                emit(element, LS.hasToLinkElement, position)
                position += 1
        else:
            for element in (*link.from_elements, *link.to_elements):
                emit(element, LS.hasLinkElement, position)
                position += 1

    return graph


def write_container(archive: ContainerArchive, container: Container) -> None:
    """Write the canonical ISO 21597-1 layout into an archive."""
    archive.write(
        CONTAINER_LAYOUT.index,
        serialise(build_index_graph(container)).encode("utf-8"),
    )
    for linkset in container.linksets:
        archive.write(
            f"{CONTAINER_LAYOUT.triples_folder}/{linkset.filename}",
            serialise(build_linkset_graph(container.description.id, linkset)).encode("utf-8"),
        )


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    message: str
    #: The offending file or subject, named so the report is actionable.
    subject: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    issues: tuple[ValidationIssue, ...] = ()
    #: Parsed graphs, exposed so a host can run its own SPARQL or the published SHACL shapes.
    index_graph: Graph | None = None
    linkset_graphs: Mapping[str, Graph] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def validate_container(archive: ContainerArchive, container: Container) -> ValidationReport:
    """Structural checks: missing payloads, undeclared files, dangling targets, direction misuse."""
    issues: list[ValidationIssue] = []
    entries = set(archive.entries())

    index_bytes = archive.read(CONTAINER_LAYOUT.index)
    index_graph: Graph | None = None
    if index_bytes is None:
        issues.append(
            ValidationIssue("error", "The container has no index.rdf.", CONTAINER_LAYOUT.index)
        )
    else:
        try:
            index_graph = parse(index_bytes)
        except RdfError as thrown:
            issues.append(ValidationIssue("error", str(thrown), CONTAINER_LAYOUT.index))

    known_documents = {document.id for document in container.documents}
    declared_paths: set[str] = {CONTAINER_LAYOUT.index}

    for document in container.documents:
        if document.kind == "internal":
            if not document.filename:
                issues.append(
                    ValidationIssue(
                        "error",
                        f'Internal document "{document.name}" has no filename.',
                        document.id,
                    )
                )
                continue
            path = document.payload_path
            assert path is not None
            declared_paths.add(path)
            if path not in entries:
                issues.append(
                    ValidationIssue(
                        "error", f'Payload "{path}" is declared but not present.', document.id
                    )
                )
        elif document.kind == "external" and not document.url:
            issues.append(
                ValidationIssue(
                    "error", f'External document "{document.name}" has no url.', document.id
                )
            )
        elif document.kind == "folder" and not document.foldername:
            issues.append(
                ValidationIssue(
                    "error", f'Folder document "{document.name}" has no foldername.', document.id
                )
            )
        if document.prior_version and document.prior_version not in known_documents:
            issues.append(
                ValidationIssue(
                    "error",
                    f'"{document.name}" names a prior version that is not in this container.',
                    document.id,
                )
            )

    linkset_graphs: dict[str, Graph] = {}
    for linkset in container.linksets:
        path = f"{CONTAINER_LAYOUT.triples_folder}/{linkset.filename}"
        declared_paths.add(path)
        payload = archive.read(path)
        if payload is None:
            issues.append(
                ValidationIssue(
                    "error", f'Linkset "{path}" is declared but not present.', linkset.id
                )
            )
        else:
            try:
                linkset_graphs[linkset.id] = parse(payload)
            except RdfError as thrown:
                issues.append(ValidationIssue("error", str(thrown), path))

        for index, link in enumerate(linkset.links):
            label = link.id or f"{linkset.id}[{index}]"
            descriptor = link.descriptor
            if descriptor is None:
                issues.append(ValidationIssue("error", f'Unknown link type "{link.type}".', label))
                continue

            elements = (*link.from_elements, *link.to_elements)
            if len(elements) < 2:
                issues.append(ValidationIssue("error", "A link needs at least two ends.", label))
            for element in elements:
                if element.document_id not in known_documents:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f'Link target document "{element.document_id}" is not in this '
                            "container.",
                            label,
                        )
                    )

            if descriptor.directed and (not link.from_elements or not link.to_elements):
                issues.append(
                    ValidationIssue(
                        "error",
                        f'"{link.type}" is directed but is missing a from or to end.',
                        label,
                    )
                )
            if not descriptor.directed and link.from_elements and link.to_elements:
                # A symmetric link given a direction is a statement its class cannot carry.
                issues.append(
                    ValidationIssue(
                        "warning",
                        f'"{link.type}" is symmetric, so its from/to split carries no meaning.',
                        label,
                    )
                )

    for entry in sorted(entries - declared_paths):
        if entry.startswith(CONTAINER_LAYOUT.ontology_folder):
            continue  # ontology copies are expected and are not payload
        issues.append(
            ValidationIssue("warning", f'"{entry}" is in the archive but not declared.', entry)
        )

    return ValidationReport(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        index_graph=index_graph,
        linkset_graphs=linkset_graphs,
    )
