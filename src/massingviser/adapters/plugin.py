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
            from ..plugins.interop import ImportAdapterToken

            ifc_module = load("ifc")
            ifc_adapter = ifc_module.IfcImportAdapter()
            context.capabilities.provide(ImportAdapterToken, ifc_adapter, version=PLUGIN_VERSION)

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

    __slots__ = ("_set", "_refs")

    def __init__(self, meshes: Mapping[str, Any]) -> None:
        from ..geometry import MESH_ENCODING, build_geometry_payloads
        from ..plugins.engine import PayloadRef, payload_path

        self._set = build_geometry_payloads(meshes)
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
        from ..plugins.engine import GeometryRef

        return {
            global_id: tuple(
                GeometryRef(
                    payload_id=placement.payload_id,
                    geometry_index=placement.geometry_index,
                    lod=placement.lod,
                    face_count=placement.face_count,
                )
                for placement in ladder
            )
            for global_id, ladder in self._set.placements.items()
        }

    def read(self, payload_id: str) -> bytes | None:
        found = self._set.by_id(payload_id)
        return found.data if found is not None else None


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

    meshes = source.meshes()
    if meshes:
        # The expensive half, done once at import. Tessellation already happened when the file was
        # parsed; this decimates, chunks and content-addresses it, so what reaches a client is a
        # budget rather than a building.
        context.capabilities.provide(
            GeometryPayloadSourceToken,
            _IfcGeometrySource(meshes),
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
