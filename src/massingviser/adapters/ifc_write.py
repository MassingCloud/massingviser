"""Writing IFC.

The other direction. Reading is the harder half and was built first, but a platform that can only
read is a dead end: a concept model that cannot leave as IFC is a concept model nobody downstream
can use.

**Geometry is written as ``IfcPolygonalFaceSet``**, IFC4's tessellated form. The platform's
geometry is already triangles -- massing tessellates its own extrusions, and an imported model was
tessellated on the way in -- so writing a face set is lossless with respect to what is actually
held. Reconstructing swept solids and B-reps from triangles would be inventing parametric intent
the platform never had, and any consumer would be entitled to believe it.

**Everything is written in metres, in world coordinates**, with an identity placement per product.
That matches how geometry travels everywhere else here, and it means a coordinate written out is
the same number a coordinate read in would be.

What this does not write: no relationships beyond spatial containment, no materials, no types, no
classification. Each is a real IFC feature and none of them is invented here from data the platform
does not hold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import numpy as np

from ..kernel import KernelError, Result, err, ok

#: What an element becomes when the platform has no better idea. A proxy is the honest choice for
#: massing: it is a solid with a place in the spatial tree and no claim to be a wall or a slab.
DEFAULT_CLASS = "IfcBuildingElementProxy"

#: Classes IFC will not let us hang geometry off, or that belong to the spatial tree rather than
#: the element tree. An element arriving with one of these is written as a proxy instead.
_SPATIAL = frozenset({"IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace"})

#: IFC's own base64 alphabet for GlobalId. 22 characters, and not the standard one.
_GUID_ALPHABET = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$")

#: Where a non-IFC identity is recorded when the element did not arrive with a GlobalId.
SOURCE_ID_PROPERTY = "MassingViserId"


def is_ifc_guid(value: str) -> bool:
    """Whether a string is already a compliant ``IfcGloballyUniqueId``."""
    return len(value) == 22 and set(value) <= _GUID_ALPHABET


@dataclass(frozen=True)
class ExportElement:
    """One element on its way out."""

    global_id: str
    name: str
    ifc_class: str = DEFAULT_CLASS
    #: Storey key. Elements sharing one land in the same ``IfcBuildingStorey``.
    level: str | None = None
    #: Building key. Elements sharing one land in the same ``IfcBuilding``. A massing scheme has
    #: one per mass, so the spatial tree that comes out matches the one a modeller would draw.
    building: str | None = None
    vertices: Any = None
    faces: Any = None
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportSummary:
    elements: int
    storeys: int
    buildings: int = 1
    #: Elements written with no geometry, and why. Reported rather than dropped silently: an
    #: element that arrives in the recipient's viewer as nothing is worse than one that is absent.
    without_geometry: tuple[tuple[str, str], ...] = ()


def _face_set(file: Any, vertices: np.ndarray, faces: np.ndarray) -> Any:
    """A tessellated solid. IFC indices are **1-based**, which is the classic way to get this wrong.

    An off-by-one here does not fail: it produces a shape that loads, renders, and is subtly the
    wrong solid, with one vertex of every triangle pulled to the previous one.
    """
    coordinates = file.create_entity(
        "IfcCartesianPointList3D",
        CoordList=[[float(x), float(y), float(z)] for x, y, z in vertices],
    )
    polygons = [
        file.create_entity(
            "IfcIndexedPolygonalFace", CoordIndex=[int(a) + 1, int(b) + 1, int(c) + 1]
        )
        for a, b, c in faces
    ]
    return file.create_entity(
        "IfcPolygonalFaceSet", Coordinates=coordinates, Closed=True, Faces=polygons
    )


def write_ifc(
    elements: Sequence[ExportElement],
    *,
    project_name: str = "MassingViser export",
    site_name: str = "Site",
    building_name: str = "Building",
    schema: str = "IFC4",
) -> tuple[bytes, ExportSummary]:
    """Build an IFC file in memory and return it with a report of what went into it."""
    file = ifcopenshell.file(schema=schema)
    project = ifcopenshell.api.root.create_entity(file, ifc_class="IfcProject", name=project_name)
    # Metres, **explicitly**. `assign_unit()` with no arguments does not mean "SI defaults" -- it
    # writes millimetres, and a reader that honours the file then scales a 50 m block to 0.05. The
    # unit is stated here so a coordinate written out is the same number that was read in.
    ifcopenshell.api.unit.assign_unit(
        file,
        units=[
            ifcopenshell.api.unit.add_si_unit(file, unit_type="LENGTHUNIT"),
            ifcopenshell.api.unit.add_si_unit(file, unit_type="AREAUNIT"),
            ifcopenshell.api.unit.add_si_unit(file, unit_type="VOLUMEUNIT"),
        ],
    )

    model_context = ifcopenshell.api.context.add_context(file, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        file,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_context,
    )

    site = ifcopenshell.api.root.create_entity(file, ifc_class="IfcSite", name=site_name)
    ifcopenshell.api.aggregate.assign_object(file, products=[site], relating_object=project)

    # Buildings and storeys in the order the elements name them, so a re-export of an unchanged
    # model produces the same file rather than one that differs by dictionary iteration order.
    tree: dict[str, list[str]] = {}
    for element in elements:
        levels = tree.setdefault(element.building or building_name, [])
        key = element.level or "Level"
        if key not in levels:
            levels.append(key)

    buildings: dict[str, Any] = {}
    storeys: dict[tuple[str, str], Any] = {}
    for building_key, levels in tree.items():
        building = ifcopenshell.api.root.create_entity(
            file, ifc_class="IfcBuilding", name=building_key
        )
        ifcopenshell.api.aggregate.assign_object(file, products=[building], relating_object=site)
        buildings[building_key] = building
        for name in levels:
            storey = ifcopenshell.api.root.create_entity(
                file, ifc_class="IfcBuildingStorey", name=name
            )
            ifcopenshell.api.aggregate.assign_object(
                file, products=[storey], relating_object=building
            )
            storeys[(building_key, name)] = storey

    without_geometry: list[tuple[str, str]] = []

    for element in elements:
        ifc_class = element.ifc_class
        if ifc_class in _SPATIAL or not ifc_class.startswith("Ifc"):
            ifc_class = DEFAULT_CLASS
        try:
            product = ifcopenshell.api.root.create_entity(
                file, ifc_class=ifc_class, name=element.name
            )
        except Exception:  # noqa: BLE001 -- an unknown class in this schema, not a fatal error
            product = ifcopenshell.api.root.create_entity(
                file, ifc_class=DEFAULT_CLASS, name=element.name
            )

        # Identity, handled honestly. An element that came from an IFC file already has a valid
        # GlobalId and keeps it, so a read-write round trip preserves every quantity, pin and clash
        # decision keyed on it. An element from massing has an id like "mass-1:003", which is *not*
        # a GlobalId -- IFC gets a fresh compliant one and the original is written to a property, so
        # the correspondence survives without either format being lied to.
        if is_ifc_guid(element.global_id):
            product.GlobalId = element.global_id

        vertices = (
            np.asarray(element.vertices, dtype=np.float64).reshape(-1, 3)
            if (element.vertices is not None)
            else np.zeros((0, 3))
        )
        faces = (
            np.asarray(element.faces, dtype=np.int64).reshape(-1, 3)
            if (element.faces is not None)
            else np.zeros((0, 3), dtype=np.int64)
        )

        if len(vertices) and len(faces):
            shape = file.create_entity(
                "IfcShapeRepresentation",
                ContextOfItems=body,
                RepresentationIdentifier="Body",
                RepresentationType="Tessellation",
                Items=[_face_set(file, vertices, faces)],
            )
            product.Representation = file.create_entity(
                "IfcProductDefinitionShape", Representations=[shape]
            )
            ifcopenshell.api.geometry.edit_object_placement(file, product=product, matrix=np.eye(4))
        else:
            without_geometry.append((element.global_id, "no triangles to write"))

        ifcopenshell.api.spatial.assign_container(
            file,
            products=[product],
            relating_structure=storeys[
                (element.building or building_name, element.level or "Level")
            ],
        )

        scalars = {
            key: value
            for key, value in element.properties.items()
            if isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes)
        }
        if not is_ifc_guid(element.global_id):
            scalars[SOURCE_ID_PROPERTY] = element.global_id
        if scalars:
            pset = ifcopenshell.api.pset.add_pset(file, product=product, name="Pset_MassingViser")
            ifcopenshell.api.pset.edit_pset(file, pset=pset, properties=scalars)

    return file.to_string().encode("utf-8"), ExportSummary(
        elements=len(elements),
        storeys=len(storeys),
        buildings=len(buildings),
        without_geometry=tuple(without_geometry),
    )


class IfcExportAdapter:
    """Satisfies interop's ``ExportAdapterToken``.

    The elements come from a callable rather than from a capability lookup here, so this object
    stays a pure writer and the composition root decides what a "model" is.
    """

    format = "ifc"

    __slots__ = ("_source", "_summary")

    def __init__(self, source: Any) -> None:
        self._source = source
        self._summary: ExportSummary | None = None

    @property
    def last_summary(self) -> ExportSummary | None:
        return self._summary

    async def write(self, **options: Any) -> Result[bytes, KernelError]:
        try:
            elements = list(self._source())
        except Exception as thrown:  # noqa: BLE001
            return err(KernelError("COMMAND_FAILED", f"Could not collect elements: {thrown}", {}))
        if not elements:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "There is nothing to export: no element source produced any geometry.",
                    {},
                )
            )
        try:
            payload, summary = write_ifc(elements, **options)
        except Exception as thrown:  # noqa: BLE001 -- third-party writer
            return err(KernelError("COMMAND_FAILED", f"IFC could not be written: {thrown}", {}))
        self._summary = summary
        return ok(payload)
