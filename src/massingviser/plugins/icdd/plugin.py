from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

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
from .container import (
    Container,
    ContainerArchive,
    MemoryArchive,
    ValidationReport,
    invert_link,
    validate_container,
    write_container,
)

PLUGIN_ID = "massingviser.icdd"
PLUGIN_VERSION = "0.1.0"


@runtime_checkable
class IcddService(Protocol):
    def write(self, archive: ContainerArchive, container: Container) -> Result[None, KernelError]: ...
    def validate(self, archive: ContainerArchive, container: Container) -> ValidationReport: ...


IcddToken: CapabilityToken[IcddService] = create_capability_token("icdd.service")


class ICDD_COMMANDS:
    write = "icdd.container.write"
    validate = "icdd.container.validate"


class IcddServiceImpl:
    __slots__ = ("_context",)

    def __init__(self, context: PluginContext) -> None:
        self._context = context

    def write(self, archive: ContainerArchive, container: Container) -> Result[None, KernelError]:
        try:
            write_container(archive, container)
        except ValueError as thrown:
            return err(KernelError("COMMAND_FAILED", str(thrown), {}))
        return ok(None)

    def validate(self, archive: ContainerArchive, container: Container) -> ValidationReport:
        return validate_container(archive, container)


def create_icdd_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    def activate(context: PluginContext) -> None:
        service = IcddServiceImpl(context)
        context.capabilities.provide(IcddToken, service, version=PLUGIN_VERSION)

        def write(params: Mapping[str, Any], _ctx: Any) -> Any:
            archive = params.get("archive") or MemoryArchive()
            result = service.write(archive, params["container"])
            if not result.ok:
                raise result.error
            return archive

        def validate(params: Mapping[str, Any], _ctx: Any) -> Any:
            return service.validate(params["archive"], params["container"])

        context.commands.register(
            CommandDefinition(id=ICDD_COMMANDS.write, title="Write ICDD container", handler=write)
        )
        context.commands.register(
            CommandDefinition(
                id=ICDD_COMMANDS.validate, title="Validate ICDD container", handler=validate
            )
        )
        context.logger.info("ICDD capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="ICDD",
        description="ISO 21597 containers: ontologies, RDF/XML codec, assembly, linking, "
        "validation.",
        activate=activate,
    )


icdd_plugin = create_icdd_plugin()
