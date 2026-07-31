from __future__ import annotations

from typing import Any, Mapping

from ...kernel import CommandDefinition, CommandInvocation, PluginContext, UIContribution
from ...schema import FamilyRepositoryRecord
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    FAMILY_COMMANDS,
    FAMILY_PERMISSIONS,
    FamilyLibraryRegistryToken,
    FamilyParameterToken,
    FamilyPlacementToken,
    FamilyResolverToken,
    FamilyVersionToken,
    PackageQuery,
    PlacementOptions,
)
from .services import (
    FamilyLibraryRegistryServiceImpl,
    FamilyParameterServiceImpl,
    FamilyPlacementServiceImpl,
    FamilyResolverServiceImpl,
    FamilyRuntime,
    FamilyVersionServiceImpl,
    create_family_stores,
)

PLUGIN_ID = "massingviser.family-libraries"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_families_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Reusable content libraries, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = FamilyRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_family_stores(context)

        registry = FamilyLibraryRegistryServiceImpl(runtime, stores)
        resolver = FamilyResolverServiceImpl(runtime, stores)
        parameters = FamilyParameterServiceImpl(runtime, stores)
        placement = FamilyPlacementServiceImpl(runtime, stores, parameters)
        versions = FamilyVersionServiceImpl(runtime, stores, parameters)

        context.capabilities.provide(FamilyLibraryRegistryToken, registry, version=PLUGIN_VERSION)
        context.capabilities.provide(FamilyResolverToken, resolver, version=PLUGIN_VERSION)
        context.capabilities.provide(FamilyParameterToken, parameters, version=PLUGIN_VERSION)
        context.capabilities.provide(FamilyPlacementToken, placement, version=PLUGIN_VERSION)
        context.capabilities.provide(FamilyVersionToken, versions, version=PLUGIN_VERSION)

        async def add_repository(params: Mapping[str, Any], _ctx: Any) -> Any:
            record = params["repository"]
            if not isinstance(record, FamilyRepositoryRecord):
                record = FamilyRepositoryRecord(**dict(record))
            return _unwrap(await registry.add_repository(record))

        async def sync(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await registry.sync(params.get("repository_id")))

        async def search(params: Mapping[str, Any], _ctx: Any) -> Any:
            return registry.search(
                PackageQuery(
                    text=params.get("text"),
                    category=params.get("category"),
                    tags=tuple(params.get("tags", ())),
                    repository_id=params.get("repository_id"),
                )
            )

        async def place(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await placement.place(
                    params["package_id"],
                    params["version"],
                    PlacementOptions(
                        transform=tuple(params.get("transform", ())),
                        parameters=params.get("parameters", {}),
                        name=params.get("name"),
                        model_id=params.get("model_id"),
                        level_id=params.get("level_id"),
                    ),
                )
            )

        async def set_parameters(params: Mapping[str, Any], _ctx: Any) -> dict[str, Any]:
            instance_id = params["instance_id"]
            previous = dict(parameters.get(instance_id))
            _unwrap(await parameters.set(instance_id, params["parameters"]))
            return {"instance_id": instance_id, "previous": previous}

        async def upgrade(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await versions.upgrade(params["instance_ids"], params["to_version"])
            )

        for command in (
            CommandDefinition(
                id=FAMILY_COMMANDS.add_repository,
                title="Add family repository",
                permission=FAMILY_PERMISSIONS.manage_repositories,
                handler=add_repository,
            ),
            CommandDefinition(
                id=FAMILY_COMMANDS.sync_repositories,
                title="Sync libraries",
                permission=FAMILY_PERMISSIONS.manage_repositories,
                handler=sync,
            ),
            CommandDefinition(
                id=FAMILY_COMMANDS.search_packages, title="Search families", handler=search
            ),
            CommandDefinition(
                id=FAMILY_COMMANDS.place_instance,
                title="Place family",
                permission=FAMILY_PERMISSIONS.place,
                handler=place,
                create_inverse=lambda _params, record: CommandInvocation(
                    "family.instance.remove", {"instance_id": record.id}
                ),
            ),
            CommandDefinition(
                id="family.instance.remove",
                title="Remove family instance",
                permission=FAMILY_PERMISSIONS.place,
                handler=lambda params, _ctx: placement.remove(params["instance_id"]),
            ),
            CommandDefinition(
                id=FAMILY_COMMANDS.set_parameters,
                title="Set family parameters",
                permission=FAMILY_PERMISSIONS.place,
                handler=set_parameters,
                create_inverse=lambda _params, result: CommandInvocation(
                    FAMILY_COMMANDS.set_parameters,
                    {
                        "instance_id": result["instance_id"],
                        "parameters": result["previous"],
                    },
                ),
            ),
            CommandDefinition(
                id=FAMILY_COMMANDS.upgrade_instances,
                title="Upgrade family instances",
                permission=FAMILY_PERMISSIONS.place,
                handler=upgrade,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="families.panel", point="panel", title="Libraries", placement="left", order=25
            )
        )
        context.logger.info("Family libraries capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Family libraries",
        description="Repository adapters, semver resolution, package validation, placement, "
        "parameter checking and version upgrade.",
        permissions=[
            FAMILY_PERMISSIONS.place,
            FAMILY_PERMISSIONS.manage_repositories,
            FAMILY_PERMISSIONS.publish,
        ],
        activate=activate,
    )


families_plugin = create_families_plugin()
