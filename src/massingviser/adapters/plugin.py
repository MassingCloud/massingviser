"""The optional-adapters plugin.

Registers whatever extras are installed, and says plainly what is not. Activating it on a machine
with no extras is a no-op that logs why -- not a failure, because the platform's fifteen capability
families do not need any of this to run.

When an IFC file is imported, this plugin publishes the parsed model to **every** capability that
asked for elements: estimating gets a takeoff source, coordination gets snapshots and a
solid-accurate clash engine, markup gets an element resolver, the engine bridge gets scene nodes,
and the geometry layer gets a spatial index. One import, six capabilities, no plugin changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..kernel import CommandDefinition, PluginContext
from ..sdk import define_plugin
from . import REQUIREMENTS, available, load, missing

PLUGIN_ID = "massingviser.adapters"
PLUGIN_VERSION = "0.1.0"


class ADAPTER_COMMANDS:
    status = "adapters.status"
    open_ifc = "adapters.ifc.open"


def create_adapters_plugin() -> Any:
    """Install every optional adapter whose dependencies are present."""

    def activate(context: PluginContext) -> None:
        installed = available()
        absent = missing()

        ifc_adapter: Any = None

        if "ifc" in installed:
            from ..plugins.interop import ExportAdapterToken, ImportAdapterToken

            ifc_module = load("ifc")
            ifc_adapter = ifc_module.IfcImportAdapter()
            context.capabilities.provide(ImportAdapterToken, ifc_adapter, version=PLUGIN_VERSION)

            # Writing, from whatever currently answers "what elements are there, and what shape
            # are they?" -- the massing bridge today, an imported model once one is loaded.
            writer = load("ifc_write")
            context.capabilities.provide(
                ExportAdapterToken,
                writer.IfcExportAdapter(lambda: _export_elements(context)),
                version=PLUGIN_VERSION,
            )

            def publish(payload: Any) -> None:
                """Publish every parsed model to everything that consumes elements."""
                summary = payload.get("summary") if isinstance(payload, Mapping) else None
                if summary is None or getattr(summary, "format", None) != "ifc":
                    return
                for model_id, model in ifc_adapter.models.items():
                    _publish_model(context, ifc_module, model, model_id)

            from ..plugins.interop import INTEROP_EVENTS

            context.events.on(INTEROP_EVENTS.imported, publish)

            def open_ifc(params: Mapping[str, Any], _ctx: Any) -> Any:
                model = ifc_module.open_ifc(
                    params["path"],
                    model_id=params.get("model_id", "ifc"),
                    version=params.get("version", "1"),
                )
                ifc_adapter.models  # noqa: B018 -- keep the adapter's registry authoritative
                _publish_model(context, ifc_module, model, model.model_id)
                return {
                    "modelId": model.model_id,
                    "elements": len(model),
                    "units": model.source_units,
                }

            context.commands.register(
                CommandDefinition(
                    id=ADAPTER_COMMANDS.open_ifc, title="Open IFC file", handler=open_ifc
                )
            )

        def status(_params: Mapping[str, Any], _ctx: Any) -> Any:
            return {"available": available(), "missing": missing(), "requires": dict(REQUIREMENTS)}

        context.commands.register(
            CommandDefinition(id=ADAPTER_COMMANDS.status, title="Adapter status", handler=status)
        )

        context.logger.info(
            "Optional adapters installed",
            {"available": list(installed), "missing": sorted(absent)},
        )

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Optional adapters",
        description="IfcOpenShell, trimesh/manifold3d and pyproj, each behind a capability token.",
        activate=activate,
    )


class _IfcGeometrySource:
    """An IFC model's triangles, encoded once and served by content id.

    Built eagerly on import rather than on first request: the cost belongs to the import that the
    user already knows is slow, not to the first camera move, which they expect to be instant.
    """

    __slots__ = ("_set", "_refs", "_belongs")

    def __init__(self, shapes: Mapping[str, Any], belongs: Mapping[str, str]) -> None:
        from ..geometry import MESH_ENCODING, build_geometry_payloads
        from ..plugins.engine import PayloadRef, payload_path

        #: GlobalId -> the shape key it is a placement of. Several elements share one.
        self._belongs = dict(belongs)
        self._set = build_geometry_payloads(shapes)
        self._refs = tuple(
            PayloadRef(
                id=payload.id,
                role="geometry",
                path=payload_path(payload.id, "bin"),
                encoding=MESH_ENCODING,
                byte_length=payload.byte_length,
                lod=payload.lod,
                mesh_count=payload.mesh_count,
            )
            for payload in self._set.payloads
        )

    def payload_refs(self) -> Any:
        return self._refs

    def geometry(self) -> Any:
        """Every element's ladder, resolved through the shape it is a placement of.

        A thousand windows from one family all point at the same payload and the same index here;
        what makes them a thousand windows is the transform on each node.
        """
        from ..plugins.engine import GeometryRef

        ladders = {
            key: tuple(
                GeometryRef(
                    payload_id=placement.payload_id,
                    geometry_index=placement.geometry_index,
                    lod=placement.lod,
                    face_count=placement.face_count,
                )
                for placement in ladder
            )
            for key, ladder in self._set.placements.items()
        }
        return {
            global_id: ladders[key] for global_id, key in self._belongs.items() if key in ladders
        }

    def read(self, payload_id: str) -> bytes | None:
        found = self._set.by_id(payload_id)
        return found.data if found is not None else None


def _export_elements(context: PluginContext) -> list[Any]:
    """Collect what to write, from the same tokens everything else reads.

    Geometry comes from the geometry payload source and semantics from the scene node source, which
    is the same join the engine bridge makes -- so an IFC export and an engine package describe the
    same building rather than two independently-assembled ones.
    """
    from ..plugins.engine import SceneNodeSourceToken
    from . import load as _load

    writer = _load("ifc_write")
    meshes = _mesh_lookup(context)

    elements: list[Any] = []
    for provided in context.capabilities.get_all(SceneNodeSourceToken):
        for node in provided.value.nodes():
            vertices, faces = meshes.get(node.global_id, (None, None))
            # The payload holds the *shared* shape, so the node's placement has to be applied
            # before writing. Skipping this would stack every instance of a family at the origin.
            if vertices is not None and node.transform:
                from .ifc import place

                vertices = place(vertices, node.transform)
            flat = {
                f"{pset}.{key}": value
                for pset, values in (node.property_sets or {}).items()
                for key, value in (values or {}).items()
                if value is not None
            }
            elements.append(
                writer.ExportElement(
                    global_id=node.global_id,
                    name=node.global_id,
                    ifc_class=node.ifc_class,
                    level=node.level_global_id,
                    building=node.parent_global_id,
                    vertices=vertices,
                    faces=faces,
                    properties=flat,
                )
            )
    return elements


def _mesh_lookup(context: PluginContext) -> dict[str, Any]:
    """Decode the geometry payloads back to triangles.

    Round-tripping through the payload rather than keeping a second copy of every mesh: the payload
    is already the canonical geometry, and reading it back is how we know the exported solid is the
    one a client would have drawn.
    """
    from ..geometry import decode_mesh_batch
    from ..plugins.engine import GeometryPayloadSourceToken

    lookup: dict[str, Any] = {}
    for provided in context.capabilities.get_all(GeometryPayloadSourceToken):
        source = provided.value
        decoded: dict[str, Any] = {}
        for global_id, ladder in source.geometry().items():
            if global_id in lookup or not ladder:
                continue
            finest = ladder[0]
            if finest.payload_id not in decoded:
                data = source.read(finest.payload_id)
                if data is None:
                    continue
                decoded[finest.payload_id] = decode_mesh_batch(data)
            batch = decoded[finest.payload_id]
            if finest.geometry_index < len(batch):
                mesh = batch[finest.geometry_index]
                lookup[global_id] = (mesh.vertices, mesh.faces)
    return lookup


def _publish_model(context: PluginContext, ifc_module: Any, model: Any, model_id: str) -> None:
    """Wire one parsed IFC model into every capability that consumes elements."""
    from ..geometry import SceneIndex, SpatialIndexToken
    from ..plugins.coordination import ClashEngineToken, ModelSnapshotToken
    from ..plugins.engine import GeometryPayloadSourceToken, SceneNodeSourceToken
    from ..plugins.estimating import ModelElementSourceToken
    from ..plugins.markup import ElementResolverToken

    source = ifc_module.IfcModelSource(model)

    # Registered at a high priority so a real IFC model wins over the massing bridge's nominal
    # providers the moment one is loaded.
    for token in (
        ModelElementSourceToken,
        ModelSnapshotToken,
        ElementResolverToken,
        SceneNodeSourceToken,
    ):
        context.capabilities.provide(token, source, version=PLUGIN_VERSION, priority=10)

    shapes, belongs = source.instances()
    if shapes:
        # The expensive half, done once at import. Tessellation already happened when the file was
        # parsed; this decimates, chunks and content-addresses it, so what reaches a client is a
        # budget rather than a building.
        context.capabilities.provide(
            GeometryPayloadSourceToken,
            _IfcGeometrySource(shapes, belongs),
            version=PLUGIN_VERSION,
            priority=10,
        )

    boxes = source.boxes()
    if boxes:
        index = SceneIndex(boxes)

        class _Index:
            def build(self) -> SceneIndex:
                return index

            def pick(self, origin: Any, direction: Any, **options: Any) -> Any:
                return index.pick(origin, direction, **options)

            def cull(self, view_projection: Any) -> Any:
                return index.cull(view_projection)

            def groups(self) -> tuple[str, ...]:
                return (model_id,)

        context.capabilities.provide(
            SpatialIndexToken, _Index(), version=PLUGIN_VERSION, priority=10
        )

    if "solids" in available():
        solids = load("solids")
        # Grouped by storey, so a clash run compares floors rather than every element against
        # every other -- which for one model is mostly its own structure meeting itself.
        groups = {
            element.global_id: element.storey_global_id or model_id for element in model.elements
        }
        context.capabilities.provide(
            ClashEngineToken,
            solids.SolidClashEngine(source, model_id=model_id, groups=groups),
            version=PLUGIN_VERSION,
            priority=10,
        )
