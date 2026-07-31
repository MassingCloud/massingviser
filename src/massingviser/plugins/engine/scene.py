"""Engine-neutral scene packages.

The platform will need a real-time engine eventually. The mistake to avoid is writing an Unreal
layer or a Unity layer -- both would bake one vendor's object model into the conversion path, and
the conversion path is the part that has to survive. This defines a neutral package instead: a JSON
manifest plus opaque binary payloads that an Unreal plugin, a Unity importer, a Bevy crate or a
native viewer can each read with a JSON parser and a file handle. **No Python runtime is required
on the far side.**

What makes it a BIM contract rather than a mesh dump:

- **GlobalId is the identity, everywhere.** An element costed in estimating, clashed in
  coordination and selected in the engine are the same element because all three key on the same
  string. A viewer's runtime id may travel as ``transient_local_id``, labelled transient, and is
  never allowed into an index.
- **Semantics travel.** Property sets stay unflattened, relationships stay typed edges, and the
  spatial parent and level are on every node. An importer that drops these has built another mesh
  importer.
- **Indexes are precomputed** by the exporter, which already holds the model, rather than on every
  load in every engine.
- **Metres, always.** ``source_units`` records what the model was authored in -- that is
  provenance -- but every coordinate that leaves is metres, so no consumer guesses. Transforms are
  column-major with translation at indices 12-14, stated explicitly because a transposed matrix
  loads without error and puts the model somewhere else.
- **Payloads stay separate.** An engine wants to parse a small manifest and then stream geometry it
  may never fully load. Base64 inside JSON forces the whole model through a parser and inflates it
  by a third.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ...kernel import KernelError, Result, err, ok

#: Column-major, translation at indices 12, 13, 14.
IDENTITY_TRANSFORM: tuple[float, ...] = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

PayloadRole = Literal["geometry", "texture", "metadata", "reality"]


@dataclass(frozen=True)
class SceneRelationship:
    """A typed edge, kept as an edge rather than flattened into a property."""

    type: str
    to_global_id: str


@dataclass(frozen=True)
class SceneNode:
    #: The identity. Never a viewer handle.
    global_id: str
    ifc_class: str
    #: Spatial parent and containing storey, on every node, so an importer never has to infer them.
    parent_global_id: str | None = None
    level_global_id: str | None = None
    transform: tuple[float, ...] = IDENTITY_TRANSFORM
    #: Unflattened property sets: ``{"Pset_WallCommon": {"FireRating": "60"}}``.
    property_sets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    relationships: tuple[SceneRelationship, ...] = ()
    payload_id: str | None = None
    geometry_index: int | None = None
    #: A viewer's runtime id. Labelled transient, used by nothing, never indexed.
    transient_local_id: int | str | None = None


@dataclass(frozen=True)
class PayloadRef:
    id: str
    role: PayloadRole
    path: str
    encoding: str
    byte_length: int


@dataclass(frozen=True)
class RealityLayer:
    id: str
    name: str
    payload_id: str | None = None
    #: So a splat arrives in the engine marked as something to render, not something to collide
    #: with or dimension.
    measurable: bool = False


@dataclass(frozen=True)
class SceneIndexes:
    by_global_id: Mapping[str, int]
    by_class: Mapping[str, tuple[int, ...]]
    by_level: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class ScenePackage:
    generator: str
    generated_at: str
    #: What the model was authored in. Provenance only -- every coordinate below is metres.
    source_units: str
    nodes: tuple[SceneNode, ...]
    indexes: SceneIndexes
    payloads: tuple[PayloadRef, ...] = ()
    reality_layers: tuple[RealityLayer, ...] = ()
    crs: str | None = None
    format_version: int = 1


def payload_path(payload_id: str, extension: str) -> str:
    return f"payloads/{payload_id}.{extension}"


def build_scene_package(
    *,
    generator: str,
    generated_at: str,
    source_units: str,
    nodes: Sequence[SceneNode],
    payloads: Sequence[PayloadRef] = (),
    reality_layers: Sequence[RealityLayer] = (),
    crs: str | None = None,
) -> Result[ScenePackage, KernelError]:
    """Assemble a package, refusing anything an engine would silently mis-load.

    Duplicate GlobalIds are rejected rather than allowed to displace each other in the index: the
    second would win, the first would vanish, and the failure would surface inside a C++ importer
    that finds a null.
    """
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for index, node in enumerate(nodes):
        if node.global_id in seen:
            duplicates.append(node.global_id)
            continue
        seen[node.global_id] = index

    if duplicates:
        return err(
            KernelError(
                "COMMAND_FAILED",
                f"Duplicate GlobalId(s) in the scene: {', '.join(sorted(set(duplicates))[:5])}. "
                "The second would displace the first in the index.",
                {"duplicates": sorted(set(duplicates))},
            )
        )

    known_payloads = {payload.id for payload in payloads}
    missing = sorted(
        {
            node.payload_id
            for node in nodes
            if node.payload_id is not None and node.payload_id not in known_payloads
        }
    )
    if missing:
        return err(
            KernelError(
                "COMMAND_FAILED",
                f"Node(s) reference payload(s) that are not declared: {', '.join(missing)}.",
                {"missing": missing},
            )
        )

    by_class: dict[str, list[int]] = {}
    by_level: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        by_class.setdefault(node.ifc_class, []).append(index)
        if node.level_global_id is not None:
            by_level.setdefault(node.level_global_id, []).append(index)

    return ok(
        ScenePackage(
            generator=generator,
            generated_at=generated_at,
            source_units=source_units,
            nodes=tuple(nodes),
            indexes=SceneIndexes(
                by_global_id=MappingProxyType(dict(seen)),
                by_class=MappingProxyType({k: tuple(v) for k, v in by_class.items()}),
                by_level=MappingProxyType({k: tuple(v) for k, v in by_level.items()}),
            ),
            payloads=tuple(payloads),
            reality_layers=tuple(reality_layers),
            crs=crs,
        )
    )


@dataclass(frozen=True)
class SceneValidation:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_scene_package(
    package: ScenePackage, *, available_payloads: Iterable[str] | None = None
) -> SceneValidation:
    """Catch staleness here, where the message is useful, not inside an importer that finds a null."""
    errors: list[str] = []
    warnings: list[str] = []

    if len(package.indexes.by_global_id) != len(package.nodes):
        errors.append(
            f"index covers {len(package.indexes.by_global_id)} of {len(package.nodes)} nodes"
        )
    for global_id, index in package.indexes.by_global_id.items():
        if index >= len(package.nodes) or package.nodes[index].global_id != global_id:
            errors.append(f'index entry "{global_id}" does not point at that node')
            break

    if available_payloads is not None:
        present = set(available_payloads)
        for payload in package.payloads:
            if payload.id not in present:
                errors.append(f'declared payload "{payload.id}" is not in the archive')

    # A viewer's contracts hand out no mesh buffers, so a semantic-only package is legitimate --
    # but the absence is reported rather than left for a consumer to discover.
    if not any(payload.role == "geometry" for payload in package.payloads):
        warnings.append(
            "no geometry payloads: this is the semantic half only, enough for selection, "
            "filtering and property inspection but not for rendering"
        )
    if package.source_units != "m":
        warnings.append(
            f'source units were "{package.source_units}"; coordinates must already be metres'
        )
    for node in package.nodes:
        if node.transient_local_id is not None and node.global_id in package.indexes.by_global_id:
            if not isinstance(node.global_id, str) or not node.global_id:
                errors.append("a node is indexed by something other than a GlobalId")
                break
    for layer in package.reality_layers:
        if layer.measurable and layer.payload_id is None:
            warnings.append(f'reality layer "{layer.id}" claims measurable but carries no payload')

    return SceneValidation(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


class SceneQuery:
    """Reference semantics for the runtime queries.

    An engine will reimplement these natively; this is what those implementations agree with.
    """

    __slots__ = ("_package",)

    def __init__(self, package: ScenePackage) -> None:
        self._package = package

    def by_global_id(self, global_id: str) -> SceneNode | None:
        index = self._package.indexes.by_global_id.get(global_id)
        return self._package.nodes[index] if index is not None else None

    def by_class(self, ifc_class: str) -> tuple[SceneNode, ...]:
        return tuple(
            self._package.nodes[index]
            for index in self._package.indexes.by_class.get(ifc_class, ())
        )

    def by_level(self, level_global_id: str) -> tuple[SceneNode, ...]:
        return tuple(
            self._package.nodes[index]
            for index in self._package.indexes.by_level.get(level_global_id, ())
        )

    def children_of(self, global_id: str) -> tuple[SceneNode, ...]:
        return tuple(node for node in self._package.nodes if node.parent_global_id == global_id)

    def related(self, global_id: str, relationship: str | None = None) -> tuple[SceneNode, ...]:
        node = self.by_global_id(global_id)
        if node is None:
            return ()
        targets = [
            edge.to_global_id
            for edge in node.relationships
            if relationship is None or edge.type == relationship
        ]
        found = [self.by_global_id(target) for target in targets]
        return tuple(item for item in found if item is not None)


def create_scene_query(package: ScenePackage) -> SceneQuery:
    return SceneQuery(package)


def to_manifest(package: ScenePackage) -> dict[str, Any]:
    """The JSON an engine actually parses.

    Written in camelCase because the consumers are C++, C# and Rust importers, and matching the
    TypeScript implementation's wire shape means one importer reads both.
    """
    return {
        "formatVersion": package.format_version,
        "generator": package.generator,
        "generatedAt": package.generated_at,
        "sourceUnits": package.source_units,
        "crs": package.crs,
        "nodes": [
            {
                "globalId": node.global_id,
                "ifcClass": node.ifc_class,
                "parentGlobalId": node.parent_global_id,
                "levelGlobalId": node.level_global_id,
                "transform": list(node.transform),
                "propertySets": {name: dict(values) for name, values in node.property_sets.items()},
                "relationships": [
                    {"type": edge.type, "toGlobalId": edge.to_global_id}
                    for edge in node.relationships
                ],
                "payloadId": node.payload_id,
                "geometryIndex": node.geometry_index,
            }
            for node in package.nodes
        ],
        "payloads": [
            {
                "id": payload.id,
                "role": payload.role,
                "path": payload.path,
                "encoding": payload.encoding,
                "byteLength": payload.byte_length,
            }
            for payload in package.payloads
        ],
        "realityLayers": [
            {
                "id": layer.id,
                "name": layer.name,
                "payloadId": layer.payload_id,
                "measurable": layer.measurable,
            }
            for layer in package.reality_layers
        ],
        "indexes": {
            "byGlobalId": dict(package.indexes.by_global_id),
            "byClass": {k: list(v) for k, v in package.indexes.by_class.items()},
            "byLevel": {k: list(v) for k, v in package.indexes.by_level.items()},
        },
    }
