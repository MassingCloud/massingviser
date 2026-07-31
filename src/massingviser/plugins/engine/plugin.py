from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ...kernel import (
    CapabilityToken,
    CommandDefinition,
    KernelError,
    PluginContext,
    Result,
    UIContribution,
    create_capability_token,
    err,
    ok,
)
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .scene import (
    PayloadRef,
    RealityLayer,
    SceneNode,
    ScenePackage,
    SceneValidation,
    build_scene_package,
    create_scene_query,
    to_manifest,
    validate_scene_package,
)

PLUGIN_ID = "massingviser.engine-bridge"
PLUGIN_VERSION = "0.1.0"


@runtime_checkable
class SceneNodeSource(Protocol):
    """Supplies the semantic half of a scene.

    Whatever holds the model provides this -- a viewer, an IFC parser, or the massing bridge. The
    engine package never learns which, which is what keeps this family free of ``three`` and of
    viser alike.
    """

    def nodes(self) -> Sequence[SceneNode]: ...
    def payloads(self) -> Sequence[PayloadRef]: ...
    def reality_layers(self) -> Sequence[RealityLayer]: ...
    def source_units(self) -> str: ...
    def crs(self) -> str | None: ...


SceneNodeSourceToken: CapabilityToken[SceneNodeSource] = create_capability_token(
    "engine.scene-source"
)


@runtime_checkable
class SceneExportService(Protocol):
    async def build(self) -> Result[ScenePackage, KernelError]: ...
    def validate(self, package: ScenePackage, **options: Any) -> SceneValidation: ...
    async def manifest(self) -> Result[str, KernelError]: ...


SceneExportToken: CapabilityToken[SceneExportService] = create_capability_token("engine.export")


class ENGINE_COMMANDS:
    build = "engine.scene.build"
    export_manifest = "engine.scene.manifest"


class ENGINE_EVENTS:
    scene_built = "engine.scene.built"


class SceneExportServiceImpl:
    __slots__ = ("_context", "_clock")

    def __init__(self, context: PluginContext, clock: Clock) -> None:
        self._context = context
        self._clock = clock

    def _sources(self) -> list[Any]:
        return [p.value for p in self._context.capabilities.get_all(SceneNodeSourceToken)]

    async def build(self) -> Result[ScenePackage, KernelError]:
        sources = self._sources()
        if not sources:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    "No scene source is installed, so there is nothing to export.",
                    {},
                )
            )

        units = {source.source_units() for source in sources}
        if len(units) > 1:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Sources disagree about units ({', '.join(sorted(units))}). Every coordinate "
                    "that leaves must be metres, and guessing which source is right hides a "
                    "problem that has to be fixed upstream.",
                    {"units": sorted(units)},
                )
            )
        declared = {source.crs() for source in sources if source.crs() is not None}
        if len(declared) > 1:
            # Refused rather than picking one, for the same reason.
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Sources declare different CRSs ({', '.join(sorted(declared))}).",
                    {"crs": sorted(declared)},
                )
            )

        return build_scene_package(
            generator=f"massingviser/{PLUGIN_VERSION}",
            generated_at=self._clock.iso(),
            source_units=next(iter(units)) if units else "m",
            nodes=[node for source in sources for node in source.nodes()],
            payloads=[payload for source in sources for payload in source.payloads()],
            reality_layers=[
                layer for source in sources for layer in source.reality_layers()
            ],
            crs=next(iter(declared)) if declared else None,
        )

    def validate(self, package: ScenePackage, **options: Any) -> SceneValidation:
        return validate_scene_package(package, **options)

    async def manifest(self) -> Result[str, KernelError]:
        built = await self.build()
        if not built.ok:
            return err(built.error)
        self._context.events.emit(
            ENGINE_EVENTS.scene_built, {"nodes": len(built.value.nodes)}
        )
        return ok(json.dumps(to_manifest(built.value), indent=2))


def create_engine_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    resolved_clock = clock or SystemClock()

    def activate(context: PluginContext) -> None:
        service = SceneExportServiceImpl(context, resolved_clock)
        context.capabilities.provide(SceneExportToken, service, version=PLUGIN_VERSION)

        async def build(_params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.build()
            if not result.ok:
                raise result.error
            return result.value

        async def manifest(_params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.manifest()
            if not result.ok:
                raise result.error
            return result.value

        context.commands.register(
            CommandDefinition(id=ENGINE_COMMANDS.build, title="Build scene package", handler=build)
        )
        context.commands.register(
            CommandDefinition(
                id=ENGINE_COMMANDS.export_manifest,
                title="Export scene manifest",
                handler=manifest,
            )
        )
        context.logger.info("Engine bridge ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Engine bridge",
        description="Engine-neutral scene packages: GlobalId-keyed nodes, precomputed indexes, "
        "semantics and payloads by reference.",
        activate=activate,
    )


engine_plugin = create_engine_plugin()
