"""ISO 21597 containers.

Conformance rests on exact IRIs and on link direction meaning what it says, so those are what these
tests pin down. The rest is structural validation, which exists to catch the mundane failures --
a payload that never got written, a link to a renamed document -- cheaply and by name.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from massingviser.plugins.icdd import (
    CONTAINER_LAYOUT,
    CT,
    DEFAULT_PREFIXES,
    LINK_TYPES,
    LS,
    NS,
    ONTOLOGY_IRI,
    SYNTAXES,
    Container,
    ContainerDescription,
    Document,
    Graph,
    Identifier,
    Iri,
    Link,
    LinkElement,
    Linkset,
    MemoryArchive,
    Party,
    RdfError,
    build_index_graph,
    build_linkset_graph,
    compute_checksum,
    dump,
    from_jsonld,
    from_turtle,
    invert_link,
    link_type_by_iri,
    load,
    parse,
    parse_shapes,
    serialise,
    shacl_validate,
    to_jsonld,
    to_turtle,
    unsupported_parameters,
    validate_container,
    write_container,
)

# ---------------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------------


def test_the_namespaces_are_the_published_ones():
    """A container whose index declares a mistyped namespace is a zip file that looks like one."""
    assert NS.ct == "https://standards.iso.org/iso/21597/-1/ed-1/en/Container#"
    assert NS.ls == "https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#"
    assert NS.els == "https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#"
    assert ONTOLOGY_IRI.container.endswith("/Container")
    assert CT.InternalDocument == f"{NS.ct}InternalDocument"
    assert LS.hasFromLinkElement == f"{NS.ls}hasFromLinkElement"


def test_nine_families_appear_as_fifteen_classes():
    assert len(LINK_TYPES) == 15
    assert len({descriptor.family for descriptor in LINK_TYPES.values()}) == 9
    directed = [d for d in LINK_TYPES.values() if d.directed]
    symmetric = [d for d in LINK_TYPES.values() if not d.directed]
    assert len(directed) == 12 and len(symmetric) == 3


def test_every_directed_class_pairs_with_its_inverse_and_back_again():
    for descriptor in LINK_TYPES.values():
        if not descriptor.directed:
            assert descriptor.inverse is None
            continue
        inverse = LINK_TYPES[descriptor.inverse]
        assert inverse.inverse == descriptor.name
        assert inverse.family == descriptor.family


def test_a_class_can_be_found_by_its_iri():
    assert link_type_by_iri(f"{NS.els}HasPart").name == "HasPart"
    assert link_type_by_iri("https://example.invalid/Nope") is None


# ---------------------------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------------------------


def test_inverting_swaps_the_class_and_the_endpoints_together():
    """Swapping only the endpoints leaves `HasPart` saying a whole is part of its own component."""
    link = Link(
        type="HasPart",
        from_elements=(LinkElement("assembly"),),
        to_elements=(LinkElement("component"),),
    )
    inverted = invert_link(link)

    assert inverted.type == "IsPartOf"
    assert [e.document_id for e in inverted.from_elements] == ["component"]
    assert [e.document_id for e in inverted.to_elements] == ["assembly"]
    # And it round-trips.
    assert invert_link(inverted) == link


def test_inverting_a_symmetric_link_keeps_its_class():
    link = Link(
        type="ConflictsWith",
        from_elements=(LinkElement("a"),),
        to_elements=(LinkElement("b"),),
    )
    inverted = invert_link(link)
    assert inverted.type == "ConflictsWith"  # there is no inverse class to swap to
    assert [e.document_id for e in inverted.from_elements] == ["b"]


def test_inverting_an_unknown_type_raises():
    with pytest.raises(ValueError):
        invert_link(Link(type="Invented"))


# ---------------------------------------------------------------------------------------------
# RDF/XML
# ---------------------------------------------------------------------------------------------


def test_rdf_round_trips_through_serialise_and_parse():
    graph = Graph()
    graph.add("urn:x:1", f"{NS.rdf}type", Iri(CT.InternalDocument))
    graph.add_literal("urn:x:1", CT.name, "Bridge model")
    graph.add_literal("urn:x:1", CT.filename, "bridge.ifc")
    graph.add("urn:x:1", CT.belongsToContainer, Iri("urn:x:c"))

    reparsed = parse(serialise(graph))
    assert reparsed.type_of("urn:x:1") == CT.InternalDocument
    assert reparsed.literal("urn:x:1", CT.name) == "Bridge model"
    assert reparsed.value("urn:x:1", CT.belongsToContainer) == Iri("urn:x:c")


def test_markup_in_a_literal_survives_escaping():
    graph = Graph()
    graph.add_literal("urn:x:1", CT.description, 'A <wall> & a "door"')
    assert parse(serialise(graph)).literal("urn:x:1", CT.description) == 'A <wall> & a "door"'


def test_a_doctype_is_refused_rather_than_ignored():
    """A container is a file somebody else sent you."""
    hostile = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE rdf:RDF [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        f'<rdf:RDF xmlns:rdf="{NS.rdf}"></rdf:RDF>'
    )
    with pytest.raises(RdfError, match="DTD"):
        parse(hostile)


@pytest.mark.parametrize("parse_type", ["Collection", "Resource", "Literal"])
def test_rdf_parse_type_is_refused_rather_than_mis_parsed(parse_type):
    """Each one changes what the children mean; treating them as markup is confidently wrong."""
    document = (
        f'<rdf:RDF xmlns:rdf="{NS.rdf}" xmlns:ct="{NS.ct}">'
        f'<rdf:Description rdf:about="urn:x:1">'
        f'<ct:name rdf:parseType="{parse_type}"/></rdf:Description></rdf:RDF>'
    )
    with pytest.raises(RdfError, match="parseType"):
        parse(document)


def test_malformed_xml_is_reported_not_swallowed():
    with pytest.raises(RdfError, match="Malformed"):
        parse("<rdf:RDF><unclosed>")


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------


def _container(**overrides) -> Container:
    base = dict(
        description=ContainerDescription(
            id="c1", name="Bridge inspection", created_by="p1", creation_date="2026-01-01"
        ),
        parties=(Party(id="p1", name="MassingCloud", kind="Organisation"),),
        documents=(
            Document(id="model", name="Bridge", filename="bridge.ifc", filetype="ifc"),
            Document(id="report", name="Inspection", filename="inspection.pdf"),
        ),
        linksets=(
            Linkset(
                id="ls1",
                name="Findings",
                filename="findings.rdf",
                links=(
                    Link(
                        type="Elaborates",
                        from_elements=(LinkElement("report"),),
                        # Addresses one wall inside the IFC, not the file as a whole.
                        to_elements=(
                            LinkElement(
                                "model",
                                Identifier(
                                    kind="string",
                                    value="2O2Fr$t4X7Zf8NOew3FLOH",
                                    field="GlobalId",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    base.update(overrides)
    return Container(**base)


def _archive_with_payloads(container: Container) -> MemoryArchive:
    archive = MemoryArchive()
    for document in container.documents:
        if document.payload_path:
            archive.write(document.payload_path, b"payload")
    write_container(archive, container)
    return archive


def test_the_canonical_layout_is_written():
    container = _container()
    archive = _archive_with_payloads(container)
    entries = set(archive.entries())
    assert CONTAINER_LAYOUT.index in entries
    assert f"{CONTAINER_LAYOUT.triples_folder}/findings.rdf" in entries
    assert f"{CONTAINER_LAYOUT.payload_folder}/bridge.ifc" in entries


def test_the_index_declares_documents_with_the_right_classes():
    graph = build_index_graph(_container())
    internal = graph.subjects_of_type(CT.InternalDocument)
    assert len(internal) == 2
    container_subject = graph.subjects_of_type(CT.ContainerDescription)[0]
    assert graph.literal(container_subject, CT.conformanceIndicator) == "ICDD-Part1-Container"
    assert len(graph.objects(container_subject, CT.containsDocument)) == 2
    assert len(graph.objects(container_subject, CT.containsLinkset)) == 1


def test_an_external_document_declares_a_url_and_a_folder_declares_a_foldername():
    container = _container(
        documents=(
            Document(id="spec", name="Spec", kind="external", url="https://example.invalid/s.pdf"),
            Document(id="photos", name="Photos", kind="folder", foldername="photos"),
        ),
        linksets=(),
    )
    graph = build_index_graph(container)
    assert graph.subjects_of_type(CT.ExternalDocument)
    assert graph.subjects_of_type(CT.FolderDocument)


def test_a_directed_link_serialises_with_from_and_to_predicates():
    """Direction-aware serialisation is what carries the asymmetry into the graph."""
    graph = build_linkset_graph("c1", _container().linksets[0])
    link_subject = graph.subjects_of_type(LINK_TYPES["Elaborates"].iri)[0]
    assert len(graph.objects(link_subject, LS.hasFromLinkElement)) == 1
    assert len(graph.objects(link_subject, LS.hasToLinkElement)) == 1
    assert graph.objects(link_subject, LS.hasLinkElement) == []


def test_a_symmetric_link_serialises_with_plain_link_elements():
    linkset = Linkset(
        id="ls",
        name="n",
        filename="f.rdf",
        links=(
            Link(
                type="IsAlternativeTo",
                from_elements=(LinkElement("a"),),
                to_elements=(LinkElement("b"),),
            ),
        ),
    )
    graph = build_linkset_graph("c1", linkset)
    link_subject = graph.subjects_of_type(LINK_TYPES["IsAlternativeTo"].iri)[0]
    assert len(graph.objects(link_subject, LS.hasLinkElement)) == 2
    assert graph.objects(link_subject, LS.hasFromLinkElement) == []


@pytest.mark.parametrize(
    ("identifier", "predicate", "class_iri"),
    [
        (
            Identifier("string", "2O2Fr$t4X7Zf8NOew3FLOH", field="GlobalId"),
            LS.identifier,
            LS.StringBasedIdentifier,
        ),
        (Identifier("uri", "https://example.invalid/#w1"), LS.uri, LS.URIBasedIdentifier),
        (
            Identifier("query", "SELECT ?s WHERE {}", query_language="SPARQL"),
            LS.queryExpression,
            LS.QueryBasedIdentifier,
        ),
    ],
)
def test_element_level_addressing_uses_the_right_identifier_class(identifier, predicate, class_iri):
    linkset = Linkset(
        id="ls",
        name="n",
        filename="f.rdf",
        links=(
            Link(
                type="Elaborates",
                from_elements=(LinkElement("a"),),
                to_elements=(LinkElement("b", identifier),),
            ),
        ),
    )
    graph = build_linkset_graph("c1", linkset)
    subject = graph.subjects_of_type(class_iri)[0]
    assert graph.literal(subject, predicate) == identifier.value


def test_a_linkset_round_trips_through_rdf():
    graph = build_linkset_graph("c1", _container().linksets[0])
    reparsed = parse(serialise(graph))
    assert reparsed.subjects_of_type(LINK_TYPES["Elaborates"].iri)
    assert reparsed.subjects_of_type(LS.StringBasedIdentifier)


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


def test_a_well_formed_container_validates():
    container = _container()
    report = validate_container(_archive_with_payloads(container), container)
    assert report.ok, [issue.message for issue in report.errors]
    # Parsed graphs are exposed so a host can run its own SPARQL or the published SHACL shapes.
    assert report.index_graph is not None and "ls1" in report.linkset_graphs


def test_a_declared_payload_that_was_never_written_is_named():
    container = _container()
    archive = MemoryArchive()
    write_container(archive, container)  # index and linkset only, no payloads
    report = validate_container(archive, container)
    assert not report.ok
    assert any("bridge.ifc" in issue.message for issue in report.errors)


def test_a_link_to_a_document_that_is_not_in_the_container_is_an_error():
    container = _container(
        linksets=(
            Linkset(
                id="ls1",
                name="n",
                filename="f.rdf",
                links=(
                    Link(
                        type="Elaborates",
                        from_elements=(LinkElement("report"),),
                        to_elements=(LinkElement("renamed-away"),),
                    ),
                ),
            ),
        )
    )
    report = validate_container(_archive_with_payloads(container), container)
    assert not report.ok
    assert any("renamed-away" in issue.message for issue in report.errors)


def test_a_directed_link_missing_an_endpoint_is_an_error():
    container = _container(
        linksets=(
            Linkset(
                id="ls1",
                name="n",
                filename="f.rdf",
                links=(Link(type="HasPart", from_elements=(LinkElement("model"),)),),
            ),
        )
    )
    report = validate_container(_archive_with_payloads(container), container)
    assert not report.ok
    assert any("missing a from or to end" in issue.message for issue in report.errors)


def test_a_symmetric_link_given_a_direction_is_warned_about():
    """A statement its class cannot carry."""
    container = _container(
        linksets=(
            Linkset(
                id="ls1",
                name="n",
                filename="f.rdf",
                links=(
                    Link(
                        type="IsIdenticalTo",
                        from_elements=(LinkElement("model"),),
                        to_elements=(LinkElement("report"),),
                    ),
                ),
            ),
        )
    )
    report = validate_container(_archive_with_payloads(container), container)
    assert report.ok  # not fatal
    assert any("carries no meaning" in issue.message for issue in report.warnings)


def test_an_undeclared_file_in_the_archive_is_warned_about():
    container = _container()
    archive = _archive_with_payloads(container)
    archive.write("Payload documents/stowaway.txt", b"?")
    report = validate_container(archive, container)
    assert report.ok
    assert any("stowaway" in issue.message for issue in report.warnings)


def test_an_internal_document_with_no_filename_is_an_error():
    container = _container(documents=(Document(id="model", name="Bridge"),), linksets=())
    report = validate_container(MemoryArchive(), container)
    assert not report.ok
    assert any("no filename" in issue.message for issue in report.errors)


def test_a_prior_version_outside_the_container_is_an_error():
    container = _container(
        documents=(Document(id="model", name="Bridge", filename="b.ifc", prior_version="ghost"),),
        linksets=(),
    )
    report = validate_container(_archive_with_payloads(container), container)
    assert not report.ok
    assert any("prior version" in issue.message for issue in report.errors)


def test_a_missing_index_is_an_error():
    container = _container()
    report = validate_container(MemoryArchive(), container)
    assert not report.ok
    assert any("no index.rdf" in issue.message for issue in report.errors)


# ---------------------------------------------------------------------------------------------
# Turtle and JSON-LD
#
# RDF/XML is what the standard mandates and what gets written. These two exist because a reviewer
# checking a linkset by eye reads Turtle and a web client wants JSON -- so the test that matters is
# that all three carry exactly the same triples, and that the readers refuse what they cannot
# represent rather than quietly dropping it.
# ---------------------------------------------------------------------------------------------


def _sample_graph():
    container = Container(
        description=ContainerDescription(
            id="c1", name="Tower", description="Stage 3", created_by="p1", version_id="2"
        ),
        parties=(Party(id="p1", name="Studio", kind="Organisation"),),
        documents=(
            Document(id="d1", name="Model", filename="tower.ifc", checksum="abc"),
            Document(id="d2", name="Spec", kind="external", url="https://example.test/spec"),
        ),
    )
    return build_index_graph(container)


def _triples(graph):
    return {(t.subject, t.predicate, repr(t.object)) for t in graph}


@pytest.mark.parametrize("syntax", SYNTAXES)
def test_every_syntax_round_trips_the_same_triples(syntax):
    graph = _sample_graph()
    assert _triples(load(dump(graph, syntax, prefixes=DEFAULT_PREFIXES), syntax)) == _triples(graph)


def test_turtle_uses_the_a_shorthand_and_prefixes():
    """Readability is the entire reason this format is here."""
    text = to_turtle(_sample_graph(), prefixes=DEFAULT_PREFIXES)
    assert "@prefix ct:" in text
    assert " a ct:ContainerDescription" in text
    assert "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" not in text


def test_turtle_refuses_blank_nodes_rather_than_dropping_them():
    """A linkset that asserts less than the file says is worse than one that fails to load."""
    with pytest.raises(RdfError, match="blank node"):
        from_turtle('<urn:a> <urn:b> [ <urn:c> "d" ] .')


def test_turtle_refuses_collections():
    with pytest.raises(RdfError, match="collection"):
        from_turtle('<urn:a> <urn:b> ( "x" "y" ) .')


def test_turtle_names_an_unbound_prefix():
    with pytest.raises(RdfError, match="Unknown prefix"):
        from_turtle('nope:a nope:b "c" .')


def test_turtle_reads_the_continuation_shorthands():
    graph = from_turtle(
        '@prefix ex: <urn:ex:> .\nex:s a ex:Thing ;\n    ex:p "one" , "two" ;\n    ex:q ex:o .\n'
    )
    assert len(graph) == 4
    assert sorted(graph.literal("urn:ex:s", "urn:ex:p") or "") is not None
    assert {str(o.value) for o in graph.objects("urn:ex:s", "urn:ex:p")} == {"one", "two"}


def test_turtle_preserves_datatypes_and_languages():
    source = (
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
        '<urn:s> <urn:typed> "42"^^xsd:integer ; <urn:tagged> "hello"@en .\n'
    )
    # The datatype is written as a full IRI on the way out, so read it back that way.
    graph = from_turtle(source.replace("xsd:integer", "<http://www.w3.org/2001/XMLSchema#integer>"))
    typed = graph.value("urn:s", "urn:typed")
    tagged = graph.value("urn:s", "urn:tagged")
    assert typed.datatype.endswith("integer")
    assert tagged.language == "en"
    assert _triples(from_turtle(to_turtle(graph))) == _triples(graph)


def test_a_quote_inside_a_literal_survives_the_round_trip():
    graph = Graph()
    graph.add_literal("urn:s", "urn:p", 'a "quoted" value')
    assert from_turtle(to_turtle(graph)).literal("urn:s", "urn:p") == 'a "quoted" value'


def test_jsonld_carries_a_context_and_typed_nodes():
    document = json.loads(to_jsonld(_sample_graph(), prefixes=DEFAULT_PREFIXES))
    assert "@context" in document and "ct" in document["@context"]
    assert all("@id" in node for node in document["@graph"])
    assert any("@type" in node for node in document["@graph"])


def test_jsonld_refuses_a_node_with_no_id():
    with pytest.raises(RdfError, match="@id"):
        from_jsonld('{"@graph": [{"urn:p": [{"@value": "x"}]}]}')


def test_jsonld_that_is_not_json_is_reported():
    with pytest.raises(RdfError, match="[Nn]ot valid JSON"):
        from_jsonld("{oops")


def test_an_unknown_syntax_is_named_not_guessed():
    with pytest.raises(RdfError, match="Unknown syntax"):
        dump(Graph(), "n3")


# ---------------------------------------------------------------------------------------------
# Checksums
#
# The check that makes a container's integrity claim mean something. A swapped payload passes every
# structural rule -- the file is present, the link resolves, the graph parses -- and only the digest
# notices.
# ---------------------------------------------------------------------------------------------

PAYLOAD = b"IFC payload bytes"


def _container_with(document):
    container = Container(
        description=ContainerDescription(id="c1", name="C"), documents=(document,)
    )
    archive = MemoryArchive()
    write_container(archive, container)
    archive.write(f"Payload documents/{document.filename}", PAYLOAD)
    return validate_container(archive, container)


def test_a_matching_checksum_validates_clean():
    report = _container_with(
        Document(
            id="d1",
            name="Model",
            filename="m.ifc",
            checksum=hashlib.sha256(PAYLOAD).hexdigest(),
            checksum_algorithm="SHA-256",
        )
    )
    assert report.ok
    assert not [issue for issue in report.issues if "checksum" in issue.message]


def test_a_tampered_payload_is_an_error_that_shows_both_digests():
    report = _container_with(
        Document(
            id="d1",
            name="Tampered",
            filename="m.ifc",
            checksum="0" * 64,
            checksum_algorithm="SHA-256",
        )
    )
    assert not report.ok
    failure = next(issue for issue in report.issues if "does not match" in issue.message)
    assert failure.severity == "error"
    assert "0000000000000000" in failure.message  # what was claimed
    assert hashlib.sha256(PAYLOAD).hexdigest()[:16] in failure.message  # what is there


def test_the_algorithm_defaults_to_sha256_when_only_a_digest_is_given():
    report = _container_with(
        Document(id="d1", name="M", filename="m.ifc", checksum=hashlib.sha256(PAYLOAD).hexdigest())
    )
    assert report.ok


@pytest.mark.parametrize("algorithm", ["MD5", "SHA1", "SHA-512"])
def test_the_other_named_algorithms_verify(algorithm):
    digest = compute_checksum(PAYLOAD, algorithm)
    report = _container_with(
        Document(id="d1", name="M", filename="m.ifc", checksum=digest, checksum_algorithm=algorithm)
    )
    assert report.ok


def test_an_algorithm_we_cannot_compute_is_a_warning_not_a_pass():
    """Unverified and verified must not look the same in a report."""
    report = _container_with(
        Document(id="d1", name="M", filename="m.ifc", checksum="abc", checksum_algorithm="CRC32")
    )
    assert report.ok  # not an error -- the container is not malformed
    assert any("unverified" in issue.message for issue in report.issues)
    assert all(issue.severity == "warning" for issue in report.issues if "CRC32" in issue.message)


def test_an_algorithm_named_with_no_digest_is_flagged():
    report = _container_with(
        Document(id="d1", name="M", filename="m.ifc", checksum_algorithm="SHA-256")
    )
    assert any("no checksum" in issue.message for issue in report.issues)


def test_a_container_that_declares_no_checksum_is_still_valid():
    """Checksums are optional in ISO 21597; absence is not a defect."""
    assert _container_with(Document(id="d1", name="M", filename="m.ifc")).ok


# ---------------------------------------------------------------------------------------------
# SHACL Core
#
# A subset, and the property that keeps it honest is that it says so. An engine that skips a
# constraint and returns "conforms" has reported on a shape it never checked.
# ---------------------------------------------------------------------------------------------

SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ct: <urn:ct:> .
ct:DocShape a sh:NodeShape ;
    sh:targetClass ct:Document ;
    sh:property ct:nameRule ;
    sh:property ct:versionRule ;
    sh:property ct:kindRule .
ct:nameRule sh:path ct:name ; sh:minCount 1 ; sh:maxCount 1 .
ct:versionRule sh:path ct:version ; sh:minInclusive 1 ; sh:severity sh:Warning .
ct:kindRule sh:path ct:kind ; sh:in "internal" ; sh:in "external" .
"""

DATA = """
@prefix ct: <urn:ct:> .
ct:d1 a ct:Document ; ct:name "Model" ; ct:version 3 ; ct:kind "internal" .
ct:d2 a ct:Document ; ct:version 0 ; ct:kind "nonsense" .
ct:d3 a ct:Document ; ct:name "A" ; ct:name "B" ; ct:version 5 ; ct:kind "external" .
"""


def _run(data=DATA, shapes=SHAPES):
    return shacl_validate(from_turtle(data), from_turtle(shapes))


def test_a_conforming_graph_conforms():
    only_good = '@prefix ct: <urn:ct:> .\nct:d1 a ct:Document ; ct:name "M" ; ct:version 3 ; ct:kind "internal" .\n'
    report = _run(only_good)
    assert report.conforms and report.complete
    assert report.results == ()


def test_every_targeted_node_is_visited_once():
    report = _run()
    assert report.shapes_run == 1
    assert report.focus_nodes == 3


def test_a_missing_required_property_is_a_violation():
    failure = next(r for r in _run().results if r.constraint == "minCount")
    assert failure.focus == "urn:ct:d2"
    assert failure.severity == "violation"
    assert "at least 1" in failure.message


def test_a_repeated_single_valued_property_is_a_violation():
    failure = next(r for r in _run().results if r.constraint == "maxCount")
    assert failure.focus == "urn:ct:d3"
    assert "at most 1" in failure.message


def test_a_value_outside_an_enumeration_is_named():
    failure = next(r for r in _run().results if r.constraint == "in")
    assert "nonsense" in failure.message


def test_severity_is_honoured_and_a_warning_does_not_stop_conformance():
    """Per the spec: only violations stop a graph conforming."""
    report = _run()
    warning = next(r for r in report.results if r.constraint == "minInclusive")
    assert warning.severity == "warning"

    warnings_only = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:property ct:r .
ct:r sh:path ct:version ; sh:minInclusive 99 ; sh:severity sh:Warning .
"""
    lenient = _run(shapes=warnings_only)
    assert lenient.results and lenient.conforms


def test_a_shape_message_replaces_the_generated_one():
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:property ct:r .
ct:r sh:path ct:name ; sh:minCount 1 ; sh:message "Every document needs a name." .
"""
    failure = next(r for r in _run(shapes=shapes).results if r.focus == "urn:ct:d2")
    assert failure.message == "Every document needs a name."


def test_a_deactivated_shape_is_not_run():
    shapes = SHAPES.replace(
        "ct:DocShape a sh:NodeShape ;", 'ct:DocShape a sh:NodeShape ; sh:deactivated "true" ;'
    )
    report = _run(shapes=shapes)
    assert report.shapes_run == 0 and report.results == ()


def test_datatype_and_node_kind_are_checked():
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:property ct:r .
ct:r sh:path ct:kind ; sh:nodeKind sh:IRI .
"""
    report = _run(shapes=shapes)
    assert all(r.constraint == "nodeKind" for r in report.results)
    assert report.results  # every kind here is a literal, so every one fails


def test_a_pattern_constraint_is_applied():
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:property ct:r .
ct:r sh:path ct:kind ; sh:pattern "^(internal|external)$" .
"""
    failures = _run(shapes=shapes).results
    assert len(failures) == 1 and failures[0].focus == "urn:ct:d2"


def test_a_class_constraint_follows_the_type_of_the_value():
    data = """
@prefix ct: <urn:ct:> .
ct:c1 a ct:Container ; ct:holds ct:d1 ; ct:holds ct:p1 .
ct:d1 a ct:Document .
ct:p1 a ct:Party .
"""
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Container ; sh:property ct:r .
ct:r sh:path ct:holds ; sh:class ct:Document .
"""
    failures = _run(data, shapes).results
    assert len(failures) == 1
    assert "urn:ct:p1" in failures[0].message


def test_an_unsupported_constraint_is_reported_rather_than_skipped():
    """The property that makes a subset honest instead of a lie."""
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:property ct:r .
ct:r sh:path ct:name ; sh:minCount 1 ; sh:sparql "SELECT ..." ; sh:closedByTypes true .
"""
    report = _run(shapes=shapes)
    assert not report.complete
    assert "sparql" in report.unsupported
    assert "closedByTypes" in report.unsupported
    # And what it *can* check, it still checks.
    assert any(r.constraint == "minCount" for r in report.results)


def test_a_fully_supported_shape_reports_complete():
    assert _run().complete
    assert unsupported_parameters(parse_shapes(from_turtle(SHAPES))) == ()


def test_shapes_can_arrive_in_any_syntax():
    """The engine reads a Graph, so it never learns which file the shapes came out of."""
    graph = from_turtle(SHAPES)
    for syntax in SYNTAXES:
        assert len(parse_shapes(load(dump(graph, syntax), syntax))) == 1


def test_a_node_level_constraint_it_cannot_run_is_reported():
    """Scanning only property parameters misses sh:sparql, sh:and, sh:not -- every node-level
    component -- and a shape carrying one would come back `conforms` and `complete` without ever
    having been evaluated."""
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:sparql ct:q ; sh:rule ct:x .
"""
    report = _run(shapes=shapes)
    assert not report.complete
    assert "sparql" in report.unsupported and "rule" in report.unsupported


def test_a_property_path_it_cannot_express_is_reported_not_skipped():
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Document ; sh:property ct:r .
ct:r sh:path "not an iri" ; sh:minCount 1 .
"""
    report = _run(shapes=shapes)
    assert not report.complete
    assert any("path" in name for name in report.unsupported)


SH_IN_LIST = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:sh="http://www.w3.org/ns/shacl#" xmlns:ct="urn:ct:">
  <sh:NodeShape rdf:about="urn:ct:S"><sh:targetClass rdf:resource="urn:ct:Doc"/>
    <sh:property rdf:resource="urn:ct:r"/></sh:NodeShape>
  <rdf:Description rdf:about="urn:ct:r"><sh:path rdf:resource="urn:ct:kind"/>
    <sh:in rdf:resource="urn:ct:l1"/></rdf:Description>
  <rdf:Description rdf:about="urn:ct:l1"><rdf:first>internal</rdf:first>
    <rdf:rest rdf:resource="urn:ct:l2"/></rdf:Description>
  <rdf:Description rdf:about="urn:ct:l2"><rdf:first>external</rdf:first>
    <rdf:rest rdf:resource="http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"/></rdf:Description>
</rdf:RDF>"""


def test_sh_in_written_as_an_rdf_list_permits_its_members():
    """`sh:in` takes a list. Reading the list *head* as the one permitted value makes a shape that
    allows "internal" report "internal" as not allowed -- a reader bug that reads like a data one."""
    data = from_turtle('@prefix ct: <urn:ct:> .\nct:d1 a ct:Doc ; ct:kind "internal" .\n')
    assert shacl_validate(data, parse(SH_IN_LIST)).results == ()


def test_sh_in_as_a_list_still_rejects_a_value_outside_it():
    data = from_turtle('@prefix ct: <urn:ct:> .\nct:d1 a ct:Doc ; ct:kind "nonsense" .\n')
    results = shacl_validate(data, parse(SH_IN_LIST)).results
    assert len(results) == 1
    assert "external" in results[0].message and "internal" in results[0].message


# ---------------------------------------------------------------------------------------------
# SHACL beyond the core constraints
#
# Logical constraints run one shape against another's values, which is why shapes have to be
# addressable -- and why a shape referenced by `sh:node` but never typed `sh:NodeShape` still has
# to be found. Collecting only declared shapes would leave every logical constraint pointing at
# nothing and quietly passing.
# ---------------------------------------------------------------------------------------------

LOGICAL_DATA = """
@prefix ct: <urn:ct:> .
ct:d1 a ct:Doc ; ct:owner ct:p1 .
ct:d2 a ct:Doc ; ct:owner ct:x1 .
ct:p1 a ct:Person ; ct:email "a@b.c" ; ct:team ct:t1 .
ct:x1 a ct:Robot .
ct:t1 ct:name "Design" .
"""


def _xml(body):
    return parse(
        '<?xml version="1.0"?><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:sh="http://www.w3.org/ns/shacl#" xmlns:ct="urn:ct:">' + body + "</rdf:RDF>"
    )


_PERSON_SHAPE = """
 <rdf:Description rdf:about="urn:ct:PersonShape">
   <sh:property rdf:resource="urn:ct:emailRule"/></rdf:Description>
 <rdf:Description rdf:about="urn:ct:emailRule">
   <sh:path rdf:resource="urn:ct:email"/><sh:minCount>1</sh:minCount></rdf:Description>
"""


def test_sh_node_runs_another_shape_against_each_value():
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:DocShape a sh:NodeShape ; sh:targetClass ct:Doc ; sh:property ct:ownerRule .
ct:ownerRule sh:path ct:owner ; sh:node ct:PersonShape .
ct:PersonShape sh:property ct:emailRule .
ct:emailRule sh:path ct:email ; sh:minCount 1 .
"""
    report = shacl_validate(from_turtle(LOGICAL_DATA), from_turtle(shapes))
    # d1's owner is a Person with an email; d2's is a Robot without one.
    assert [r.focus for r in report.results] == ["urn:ct:d2"]
    assert report.complete


def test_a_shape_referenced_but_never_declared_is_still_found():
    """SHACL infers shape-hood from use; `ct:PersonShape` above is never typed sh:NodeShape."""
    shapes = from_turtle(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n@prefix ct: <urn:ct:> .\n"
        "ct:S a sh:NodeShape ; sh:targetClass ct:Doc ; sh:property ct:r .\n"
        "ct:r sh:path ct:owner ; sh:node ct:PersonShape .\n"
        "ct:PersonShape sh:property ct:er .\nct:er sh:path ct:email ; sh:minCount 1 .\n"
    )
    assert any(shape.id == "urn:ct:PersonShape" for shape in parse_shapes(shapes))


def test_sh_not_inverts_the_verdict():
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:S a sh:NodeShape ; sh:targetClass ct:Doc ; sh:property ct:r .
ct:r sh:path ct:owner ; sh:not ct:PersonShape .
ct:PersonShape sh:property ct:er .
ct:er sh:path ct:email ; sh:minCount 1 .
"""
    report = shacl_validate(from_turtle(LOGICAL_DATA), from_turtle(shapes))
    assert [r.focus for r in report.results] == ["urn:ct:d1"]


def test_an_inverse_path_follows_the_predicate_backwards():
    """ "Every Person is owned by something" cannot be asked with a forward path."""
    report = shacl_validate(
        from_turtle(LOGICAL_DATA),
        _xml(
            '<sh:NodeShape rdf:about="urn:ct:PS">'
            '<sh:targetClass rdf:resource="urn:ct:Person"/>'
            '<sh:property rdf:resource="urn:ct:r"/></sh:NodeShape>'
            '<rdf:Description rdf:about="urn:ct:r">'
            '<sh:path rdf:resource="urn:ct:inv"/><sh:minCount>1</sh:minCount></rdf:Description>'
            '<rdf:Description rdf:about="urn:ct:inv">'
            '<sh:inversePath rdf:resource="urn:ct:owner"/></rdf:Description>'
        ),
    )
    assert report.complete
    assert report.results == ()  # ct:p1 is owned by ct:d1


def test_a_sequence_path_walks_more_than_one_predicate():
    report = shacl_validate(
        from_turtle(LOGICAL_DATA),
        _xml(
            '<sh:NodeShape rdf:about="urn:ct:DS">'
            '<sh:targetClass rdf:resource="urn:ct:Doc"/>'
            '<sh:property rdf:resource="urn:ct:r"/></sh:NodeShape>'
            '<rdf:Description rdf:about="urn:ct:r">'
            '<sh:path rdf:resource="urn:ct:seq"/><sh:minCount>1</sh:minCount></rdf:Description>'
            '<rdf:Description rdf:about="urn:ct:seq">'
            '<rdf:first rdf:resource="urn:ct:owner"/>'
            '<rdf:rest rdf:resource="urn:ct:seq2"/></rdf:Description>'
            '<rdf:Description rdf:about="urn:ct:seq2">'
            '<rdf:first rdf:resource="urn:ct:team"/>'
            '<rdf:rest rdf:resource="http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"/>'
            "</rdf:Description>"
        ),
    )
    # d1's owner has a team; d2's does not.
    assert [r.focus for r in report.results] == ["urn:ct:d2"]
    assert report.results[0].path == "(urn:ct:owner/urn:ct:team)"


def test_a_qualified_value_shape_counts_conforming_values():
    report = shacl_validate(
        from_turtle(LOGICAL_DATA),
        _xml(
            '<sh:NodeShape rdf:about="urn:ct:DS">'
            '<sh:targetClass rdf:resource="urn:ct:Doc"/>'
            '<sh:property rdf:resource="urn:ct:r"/></sh:NodeShape>'
            '<rdf:Description rdf:about="urn:ct:r">'
            '<sh:path rdf:resource="urn:ct:owner"/>'
            '<sh:qualifiedValueShape rdf:resource="urn:ct:PersonShape"/>'
            "<sh:qualifiedMinCount>1</sh:qualifiedMinCount></rdf:Description>" + _PERSON_SHAPE
        ),
    )
    assert [(r.focus, r.constraint) for r in report.results] == [("urn:ct:d2", "qualifiedMinCount")]


def test_a_path_form_this_engine_cannot_express_is_still_reported():
    """`zeroOrMorePath` and friends. Skipping one silently would claim it was checked."""
    shapes = _xml(
        '<sh:NodeShape rdf:about="urn:ct:S">'
        '<sh:targetClass rdf:resource="urn:ct:Doc"/>'
        '<sh:property rdf:resource="urn:ct:r"/></sh:NodeShape>'
        '<rdf:Description rdf:about="urn:ct:r">'
        '<sh:path rdf:resource="urn:ct:weird"/><sh:minCount>1</sh:minCount></rdf:Description>'
        '<rdf:Description rdf:about="urn:ct:weird">'
        '<sh:zeroOrMorePath rdf:resource="urn:ct:owner"/></rdf:Description>'
    )
    report = shacl_validate(from_turtle(LOGICAL_DATA), shapes)
    assert not report.complete
    assert any("path" in name for name in report.unsupported)


def test_a_cyclic_shape_reference_terminates():
    """A shapes graph can point at itself; a broken file must not take the process with it."""
    shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ct: <urn:ct:> .
ct:A a sh:NodeShape ; sh:targetClass ct:Doc ; sh:property ct:r .
ct:r sh:path ct:owner ; sh:node ct:A .
"""
    report = shacl_validate(from_turtle(LOGICAL_DATA), from_turtle(shapes))
    assert isinstance(report.results, tuple)
