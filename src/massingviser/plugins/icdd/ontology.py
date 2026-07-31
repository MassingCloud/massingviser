"""ISO 21597 vocabulary.

Every IRI here was taken from the published ontology documents themselves -- ``Container.rdf``,
``Linkset.rdf`` (Part 1) and ``ExtendedLinkset.rdf`` (Part 2) at ``standards.iso.org`` -- rather
than transcribed from prose. Conformance depends on these strings being exact: an ICDD container
whose index declares ``ct:InternalDocument`` with a mistyped namespace is not a container, it is a
zip file that looks like one.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


class NS:
    #: ISO 21597-1 Container ontology.
    ct = "https://standards.iso.org/iso/21597/-1/ed-1/en/Container#"
    #: ISO 21597-1 Linkset ontology.
    ls = "https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#"
    #: ISO 21597-2 Extended Linkset ontology.
    els = "https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#"
    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    rdfs = "http://www.w3.org/2000/01/rdf-schema#"
    owl = "http://www.w3.org/2002/07/owl#"
    xsd = "http://www.w3.org/2001/XMLSchema#"
    dcterms = "http://purl.org/dc/terms/"


class ONTOLOGY_IRI:
    """Ontology document IRIs, as imported by a container's index and linksets."""

    container = "https://standards.iso.org/iso/21597/-1/ed-1/en/Container"
    linkset = "https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset"
    extended_linkset = "https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset"


def _ct(local: str) -> str:
    return f"{NS.ct}{local}"


def _ls(local: str) -> str:
    return f"{NS.ls}{local}"


def _els(local: str) -> str:
    return f"{NS.els}{local}"


class CT:
    """ISO 21597-1 Container ontology classes and properties."""

    ContainerDescription = _ct("ContainerDescription")
    Document = _ct("Document")
    InternalDocument = _ct("InternalDocument")
    ExternalDocument = _ct("ExternalDocument")
    FolderDocument = _ct("FolderDocument")
    SecuredDocument = _ct("SecuredDocument")
    EncryptedDocument = _ct("EncryptedDocument")
    Linkset = _ct("Linkset")
    Party = _ct("Party")
    Person = _ct("Person")
    Organisation = _ct("Organisation")

    containsDocument = _ct("containsDocument")
    containsLinkset = _ct("containsLinkset")
    belongsToContainer = _ct("belongsToContainer")
    containedInContainer = _ct("containedInContainer")
    createdBy = _ct("createdBy")
    created = _ct("created")
    modifiedBy = _ct("modifiedBy")
    modified = _ct("modified")
    publishedBy = _ct("publishedBy")
    published = _ct("published")
    priorVersion = _ct("priorVersion")
    alternativeDocument = _ct("alternativeDocument")
    alternativeDocumentTo = _ct("alternativeDocumentTo")

    name = _ct("name")
    description = _ct("description")
    filename = _ct("filename")
    foldername = _ct("foldername")
    filetype = _ct("filetype")
    format = _ct("format")
    url = _ct("url")
    versionID = _ct("versionID")
    versionDescription = _ct("versionDescription")
    creationDate = _ct("creationDate")
    modificationDate = _ct("modificationDate")
    checksum = _ct("checksum")
    checksumAlgorithm = _ct("checksumAlgorithm")
    encryptionAlgorithm = _ct("encryptionAlgorithm")
    conformanceIndicator = _ct("conformanceIndicator")
    userID = _ct("userID")
    requested = _ct("requested")


class LS:
    """ISO 21597-1 Linkset ontology classes and properties."""

    Link = _ls("Link")
    BinaryLink = _ls("BinaryLink")
    DirectedLink = _ls("DirectedLink")
    DirectedBinaryLink = _ls("DirectedBinaryLink")
    Directed1toNLink = _ls("Directed1toNLink")
    LinkElement = _ls("LinkElement")
    Identifier = _ls("Identifier")
    StringBasedIdentifier = _ls("StringBasedIdentifier")
    URIBasedIdentifier = _ls("URIBasedIdentifier")
    QueryBasedIdentifier = _ls("QueryBasedIdentifier")

    hasLinkElement = _ls("hasLinkElement")
    hasFromLinkElement = _ls("hasFromLinkElement")
    hasToLinkElement = _ls("hasToLinkElement")
    hasDocument = _ls("hasDocument")
    hasIdentifier = _ls("hasIdentifier")

    identifier = _ls("identifier")
    identifierField = _ls("identifierField")
    uri = _ls("uri")
    queryLanguage = _ls("queryLanguage")
    queryExpression = _ls("queryExpression")


LinkFamily = Literal[
    "Identity",
    "Conflict",
    "Alternative",
    "Specialization",
    "Aggregation",
    "Membership",
    "Replacement",
    "Elaboration",
    "Control",
]


@dataclass(frozen=True)
class LinkTypeDescriptor:
    name: str
    iri: str
    #: The ISO 21597-2 semantic family this class belongs to.
    family: LinkFamily
    #: Directional links use from/to link elements; symmetric ones use plain link elements.
    directed: bool
    inverse: str | None = None


#: The nine ISO 21597-2 link types, as fifteen classes.
#:
#: Six of the nine are directional and appear as inverse pairs; three (identity, conflict,
#: alternative) are symmetric and have a single class each. Modelling the pairs explicitly rather
#: than as one class plus a direction flag is what the standard does, and it matters for reasoning
#: -- a consumer that only understands ``HasPart`` should not silently read ``IsPartOf`` backwards.
LINK_TYPES: Mapping[str, LinkTypeDescriptor] = MappingProxyType(
    {
        descriptor.name: descriptor
        for descriptor in (
            LinkTypeDescriptor("IsIdenticalTo", _els("IsIdenticalTo"), "Identity", False),
            LinkTypeDescriptor("ConflictsWith", _els("ConflictsWith"), "Conflict", False),
            LinkTypeDescriptor("IsAlternativeTo", _els("IsAlternativeTo"), "Alternative", False),
            LinkTypeDescriptor(
                "Specialises", _els("Specialises"), "Specialization", True, "IsSpecialisedAs"
            ),
            LinkTypeDescriptor(
                "IsSpecialisedAs", _els("IsSpecialisedAs"), "Specialization", True, "Specialises"
            ),
            LinkTypeDescriptor("HasPart", _els("HasPart"), "Aggregation", True, "IsPartOf"),
            LinkTypeDescriptor("IsPartOf", _els("IsPartOf"), "Aggregation", True, "HasPart"),
            LinkTypeDescriptor("HasMember", _els("HasMember"), "Membership", True, "IsMemberOf"),
            LinkTypeDescriptor("IsMemberOf", _els("IsMemberOf"), "Membership", True, "HasMember"),
            LinkTypeDescriptor(
                "Supersedes", _els("Supersedes"), "Replacement", True, "IsSupersededBy"
            ),
            LinkTypeDescriptor(
                "IsSupersededBy", _els("IsSupersededBy"), "Replacement", True, "Supersedes"
            ),
            LinkTypeDescriptor(
                "Elaborates", _els("Elaborates"), "Elaboration", True, "IsElaboratedBy"
            ),
            LinkTypeDescriptor(
                "IsElaboratedBy", _els("IsElaboratedBy"), "Elaboration", True, "Elaborates"
            ),
            LinkTypeDescriptor("Controls", _els("Controls"), "Control", True, "IsControlledBy"),
            LinkTypeDescriptor(
                "IsControlledBy", _els("IsControlledBy"), "Control", True, "Controls"
            ),
        )
    }
)


def link_type_by_iri(iri: str) -> LinkTypeDescriptor | None:
    return next(
        (descriptor for descriptor in LINK_TYPES.values() if descriptor.iri == iri), None
    )


class CONTAINER_LAYOUT:
    """Canonical container layout defined by ISO 21597-1."""

    index = "index.rdf"
    ontology_folder = "Ontology resources"
    payload_folder = "Payload documents"
    triples_folder = "Payload triples"


#: Default prefix bindings used when serialising container documents.
DEFAULT_PREFIXES: Mapping[str, str] = MappingProxyType(
    {
        "rdf": NS.rdf,
        "rdfs": NS.rdfs,
        "owl": NS.owl,
        "xsd": NS.xsd,
        "dcterms": NS.dcterms,
        "ct": NS.ct,
        "ls": NS.ls,
        "els": NS.els,
    }
)
