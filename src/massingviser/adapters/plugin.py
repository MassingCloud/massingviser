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


def _publish_model(context: PluginContext, ifc_module: Any, model: Any, model_id: str) -> None:
    """Wire one parsed IFC model into every capability that consumes elements."""
    from ..geometry import SceneIndex, SpatialIndexToken
    from ..plugins.coordination import ClashEngineToken, ModelSnapshotToken
    from ..plugins.engine import SceneNodeSourceToken
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
