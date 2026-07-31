"""IFC, via IfcOpenShell.

The single biggest thing that can move server-side. ThatOpen parses IFC *in the browser* with a
WebAssembly build of the same C++ engine; doing it here instead means the client never sees an IFC
file, never runs a parser, and never holds the memory that parsing needs.

Everything this module produces is keyed on **IfcGlobalId**, which is the identity the rest of the
platform already uses. An element measured by estimating, clashed by coordination, pinned by markup
and picked in the viewer is the same element in all four because all four key on the same string --
and here that string comes from the file rather than from anything this platform invented.

This is an optional extra. ``pip install massingviser[ifc]``; without it the platform runs exactly
as before, minus the ability to open an IFC file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.unit
import numpy as np

from ..geometry import Aabb
from ..kernel import KernelError, Result, err, ok
from ..plugins.coordination import SnapshotElement
from ..plugins.engine import PayloadRef, SceneNode, SceneRelationship
from ..plugins.estimating import TakeoffElement
from ..plugins.interop import ImportSummary

#: Classes that are building elements rather than spatial containers or annotation. Tessellating an
#: ``IfcSpace`` produces a room-shaped solid that clashes with everything inside it, which is why
#: spatial structure is read for hierarchy and excluded from geometry.
SPATIAL_CLASSES = frozenset(
    {"IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace"}
)


@dataclass
class IfcElement:
    global_id: str
    ifc_class: str
    name: str | None
    storey_global_id: str | None
    properties: dict[str, Any] = field(default_factory=dict)
    quantities: dict[str, float] = field(default_factory=dict)
    #: Flat float triples and index triples, in metres. Empty when the element has no geometry.
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None

    @property
    def box(self) -> Aabb | None:
        if self.vertices is None or len(self.vertices) == 0:
            return None
        return Aabb(tuple(self.vertices.min(axis=0)), tuple(self.vertices.max(axis=0)))


class IfcModel:
    """A parsed, tessellated IFC file held server-side.

    Tessellation happens once, on load, rather than per query. It is the expensive step and its
    result is what every downstream capability wants -- so paying for it up front is the whole
    argument for doing this on the server at all.
    """

    __slots__ = ("_file", "_elements", "_by_id", "model_id", "version", "source_units")

    def __init__(
        self,
        file: Any,
        *,
        model_id: str,
        version: str = "1",
        tessellate: bool = True,
    ) -> None:
        self._file = file
        self.model_id = model_id
        self.version = version
        # Recorded as provenance. Every coordinate below is metres regardless, because a consumer
        # that has to ask what unit a number is in will eventually guess wrong.
        self.source_units = _length_unit(file)
        self._elements: list[IfcElement] = []
        self._by_id: dict[str, IfcElement] = {}
        self._load(tessellate=tessellate)

    def __len__(self) -> int:
        return len(self._elements)

    @property
    def elements(self) -> tuple[IfcElement, ...]:
        return tuple(self._elements)

    def get(self, global_id: str) -> IfcElement | None:
        return self._by_id.get(global_id)

    def _load(self, *, tessellate: bool) -> None:
        storeys = _storey_of_element(self._file)

        products = [
            product
            for product in self._file.by_type("IfcProduct")
            if product.is_a() not in SPATIAL_CLASSES and getattr(product, "GlobalId", None)
        ]

        shapes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if tessellate and products:
            shapes = _tessellate(self._file, products)

        for product in products:
            global_id = product.GlobalId
            vertices, faces = shapes.get(global_id, (None, None))
            element = IfcElement(
                global_id=global_id,
                ifc_class=product.is_a(),
                name=getattr(product, "Name", None),
                storey_global_id=storeys.get(global_id),
                properties=_scalar_properties(product),
                quantities=_quantities(product),
                vertices=vertices,
                faces=faces,
            )
            self._elements.append(element)
            self._by_id[global_id] = element


def _length_unit(file: Any) -> str:
    try:
        scale = ifcopenshell.util.unit.calculate_unit_scale(file)
    except Exception:  # noqa: BLE001 -- a file with no unit assignment is malformed, not fatal
        return "m"
    return {1.0: "m", 0.001: "mm", 0.01: "cm", 0.3048: "ft"}.get(round(scale, 6), f"x{scale}")


def _storey_of_element(file: Any) -> dict[str, str]:
    """Map every element to its containing storey.

    Read from ``IfcRelContainedInSpatialStructure`` rather than inferred from elevation. Two
    elements at the same height can belong to different storeys -- a mezzanine, a split level --
    and a guess based on Z would put them in the wrong one.
    """
    mapping: dict[str, str] = {}
    for relation in file.by_type("IfcRelContainedInSpatialStructure"):
        structure = relation.RelatingStructure
        if structure is None or not structure.is_a("IfcBuildingStorey"):
            continue
        for element in relation.RelatedElements or ():
            if getattr(element, "GlobalId", None):
                mapping[element.GlobalId] = structure.GlobalId
    return mapping


def _scalar_properties(product: Any) -> dict[str, Any]:
    """Property sets, flattened to ``Pset.Property`` keys, scalars only.

    Non-scalar values are dropped rather than stringified: ``"[object Object]"`` is
    indistinguishable from a real value, an absent key is not.
    """
    out: dict[str, Any] = {}
    try:
        psets = ifcopenshell.util.element.get_psets(product)
    except Exception:  # noqa: BLE001
        return out
    for pset_name, values in psets.items():
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if key == "id":
                continue
            if isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes):
                out[f"{pset_name}.{key}"] = value
    return out


def _quantities(product: Any) -> dict[str, float]:
    """Quantity sets, which is where an estimator's numbers come from when the file has them."""
    out: dict[str, float] = {}
    try:
        psets = ifcopenshell.util.element.get_psets(product, qtos_only=True)
    except Exception:  # noqa: BLE001
        return out
    for values in psets.values():
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if key != "id" and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
    return out


def _tessellate(file: Any, products: Sequence[Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Triangulate every product that has geometry, in metres, in world coordinates.

    Uses the multi-threaded iterator rather than ``create_shape`` per element: on a real model the
    difference is minutes.
    """
    settings = ifcopenshell.geom.settings()
    # World coordinates, so an element's vertices are directly comparable with any other's -- which
    # is what a shared spatial index needs.
    try:
        settings.set("use-world-coords", True)
    except Exception:  # noqa: BLE001 -- older API spelling
        try:
            settings.set(settings.USE_WORLD_COORDS, True)
        except Exception:  # noqa: BLE001
            pass

    wanted = {product.GlobalId for product in products}
    shapes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    iterator = ifcopenshell.geom.iterator(settings, file, include=list(products))
    if not iterator.initialize():
        return shapes
    while True:
        shape = iterator.get()
        global_id = getattr(shape, "guid", None)
        if global_id in wanted:
            geometry = shape.geometry
            vertices = np.asarray(geometry.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(geometry.faces, dtype=np.int64).reshape(-1, 3)
            if len(vertices) and len(faces):
                shapes[global_id] = (vertices, faces)
        if not iterator.next():
            break
    return shapes


def open_ifc(source: str | bytes, *, model_id: str, version: str = "1") -> IfcModel:
    """Open an IFC file from a path or from its bytes."""
    if isinstance(source, (bytes, bytearray)):
        file = ifcopenshell.file.from_string(bytes(source).decode("utf-8", "replace"))
    else:
        file = ifcopenshell.open(source)
    return IfcModel(file, model_id=model_id, version=version)


# ---------------------------------------------------------------------------------------------
# Capability adapters
# ---------------------------------------------------------------------------------------------


class IfcImportAdapter:
    """Satisfies interop's ``ImportAdapterToken``."""

    format = "ifc"
    #: The STEP physical file header. IFC is ISO 10303-21, and every conformant file starts here.
    signatures = (b"ISO-10303-21;",)
    extensions = ("ifc", "ifczip")

    __slots__ = ("_models", "_counter")

    def __init__(self) -> None:
        self._models: dict[str, IfcModel] = {}
        self._counter = 0

    @property
    def models(self) -> Mapping[str, IfcModel]:
        return dict(self._models)

    async def read(self, payload: bytes) -> Result[ImportSummary, KernelError]:
        self._counter += 1
        model_id = f"ifc-{self._counter}"
        try:
            model = open_ifc(payload, model_id=model_id)
        except Exception as thrown:  # noqa: BLE001 -- third-party parser on untrusted input
            return err(KernelError("COMMAND_FAILED", f"IFC could not be read: {thrown}", {}))

        self._models[model_id] = model
        without_geometry = tuple(
            (element.global_id, "no geometry could be tessellated")
            for element in model.elements
            if element.vertices is None
        )
        warnings = []
        if model.source_units != "m":
            warnings.append(
                f"authored in {model.source_units}; every coordinate has been converted to metres"
            )
        return ok(
            ImportSummary(
                format="ifc",
                records=len(model),
                # Named, never silently dropped: an element with no geometry is invisible in a
                # viewer and unmeasurable in a takeoff, and both are worth knowing about.
                rejected=without_geometry,
                warnings=tuple(warnings),
            )
        )


class IfcModelSource:
    """One parsed model, published to every capability that asks for elements.

    The same object satisfies estimating's ``ModelElementSource``, coordination's
    ``ModelSnapshotSource``, markup's ``ElementResolver`` and the engine bridge's
    ``SceneNodeSource`` -- because all four want the same thing keyed the same way, which is the
    point of GlobalId being the identity.
    """

    __slots__ = ("_model",)

    def __init__(self, model: IfcModel) -> None:
        self._model = model

    # -- estimating.ModelElementSource --------------------------------------------------------

    def elements(self, model_id: str) -> Sequence[TakeoffElement]:
        if model_id != self._model.model_id:
            return ()
        return tuple(
            TakeoffElement(
                global_id=element.global_id,
                ifc_class=element.ifc_class,
                # Quantities first, then scalar properties -- a takeoff expression naming
                # `NetVolume` should get the quantity set's value, not a property that shadows it.
                properties={
                    **{k: v for k, v in element.properties.items() if isinstance(v, (int, float))},
                    **element.quantities,
                },
                level_global_id=element.storey_global_id,
            )
            for element in self._model.elements
        )

    def model_ids(self) -> Sequence[str]:
        return (self._model.model_id,)

    def model_version(self, model_id: str) -> str | None:
        return self._model.version if model_id == self._model.model_id else None

    # -- coordination.ModelSnapshotSource -----------------------------------------------------

    def snapshot(self, model_id: str, version: str) -> Sequence[SnapshotElement] | None:
        if model_id != self._model.model_id or version != self._model.version:
            return None
        out = []
        for element in self._model.elements:
            box = element.box
            out.append(
                SnapshotElement(
                    global_id=element.global_id,
                    ifc_class=element.ifc_class,
                    properties=dict(element.properties),
                    position=box.centre if box else None,
                    quantities=dict(element.quantities),
                )
            )
        return tuple(out)

    def versions(self, model_id: str) -> Sequence[str]:
        return (self._model.version,) if model_id == self._model.model_id else ()

    # -- markup.ElementResolver ---------------------------------------------------------------

    def global_ids(self, model_id: str) -> Sequence[str]:
        if model_id != self._model.model_id:
            return ()
        return tuple(element.global_id for element in self._model.elements)

    def exists(self, model_id: str, global_id: str) -> bool:
        return model_id == self._model.model_id and self._model.get(global_id) is not None

    # -- engine.SceneNodeSource ---------------------------------------------------------------

    def nodes(self) -> Sequence[SceneNode]:
        return tuple(
            SceneNode(
                global_id=element.global_id,
                ifc_class=element.ifc_class,
                level_global_id=element.storey_global_id,
                parent_global_id=element.storey_global_id,
                property_sets=_regroup(element.properties),
                relationships=(
                    (SceneRelationship("ContainedIn", element.storey_global_id),)
                    if element.storey_global_id
                    else ()
                ),
            )
            for element in self._model.elements
        )

    def payloads(self) -> Sequence[PayloadRef]:
        return ()

    def reality_layers(self) -> Sequence[Any]:
        return ()

    def source_units(self) -> str:
        return "m"

    def crs(self) -> str | None:
        return None

    # -- geometry -----------------------------------------------------------------------------

    def boxes(self) -> dict[str, Aabb]:
        """Bounds per element, for the spatial index."""
        return {
            element.global_id: box
            for element in self._model.elements
            if (box := element.box) is not None
        }

    def meshes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            element.global_id: (element.vertices, element.faces)
            for element in self._model.elements
            if element.vertices is not None and element.faces is not None
        }


def _regroup(flat: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Turn ``Pset.Property`` keys back into nested property sets.

    Unflattened is how they leave: an importer that receives a flat map has lost which set each
    property came from, and property sets are how IFC says what a value *means*.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, value in flat.items():
        pset, _, name = key.partition(".")
        out.setdefault(pset or "Properties", {})[name or key] = value
    return out
