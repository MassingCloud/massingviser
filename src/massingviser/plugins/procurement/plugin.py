from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...kernel import CommandDefinition, PluginContext, UIContribution
from ...sdk import Clock, IdFactory, SequentialIdFactory, SystemClock, define_plugin
from .contracts import (
    PROCUREMENT_COMMANDS,
    PROCUREMENT_PERMISSIONS,
    FieldStatusToken,
    InspectionToken,
    InstallProgressToken,
    PackageToken,
    VendorScopeToken,
)
from .services import (
    FieldStatusServiceImpl,
    InspectionServiceImpl,
    InstallProgressServiceImpl,
    PackageServiceImpl,
    ProcurementRuntime,
    VendorScopeServiceImpl,
    create_procurement_stores,
)

PLUGIN_ID = "massingviser.procurement-field"
PLUGIN_VERSION = "0.1.0"


def _unwrap(result: Any) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def create_procurement_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Procurement and field, packaged as a plugin."""
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        runtime = ProcurementRuntime(context=context, clock=resolved_clock, ids=resolved_ids)
        stores = create_procurement_stores(context)

        packages = PackageServiceImpl(runtime, stores)
        vendors = VendorScopeServiceImpl(runtime, stores, packages)
        field_status = FieldStatusServiceImpl(runtime, stores)
        inspections = InspectionServiceImpl(runtime, stores)
        progress = InstallProgressServiceImpl(runtime, stores, field_status, packages)

        context.capabilities.provide(PackageToken, packages, version=PLUGIN_VERSION)
        context.capabilities.provide(VendorScopeToken, vendors, version=PLUGIN_VERSION)
        context.capabilities.provide(FieldStatusToken, field_status, version=PLUGIN_VERSION)
        context.capabilities.provide(InspectionToken, inspections, version=PLUGIN_VERSION)
        context.capabilities.provide(InstallProgressToken, progress, version=PLUGIN_VERSION)

        async def create_package(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await packages.create(**dict(params)))

        async def from_boq(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await packages.from_boq_lines(
                    params["boq_line_ids"], params["name"], params["code"]
                )
            )

        async def award(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(
                await vendors.award(params["package_id"], params["vendor_id"], params["value"])
            )

        async def record_status(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await field_status.record(**dict(params)))

        async def create_inspection(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await inspections.create(**dict(params)))

        async def compute_progress(params: Mapping[str, Any], _ctx: Any) -> Any:
            return _unwrap(await progress.compute(params["package_id"], params["data_date"]))

        for command in (
            CommandDefinition(
                id=PROCUREMENT_COMMANDS.create_package,
                title="Create package",
                permission=PROCUREMENT_PERMISSIONS.manage_packages,
                handler=create_package,
            ),
            CommandDefinition(
                id=PROCUREMENT_COMMANDS.package_from_boq,
                title="Package from bill",
                permission=PROCUREMENT_PERMISSIONS.manage_packages,
                handler=from_boq,
            ),
            CommandDefinition(
                id=PROCUREMENT_COMMANDS.award_package,
                title="Award package",
                # Awarding commits money and is a one-way door, so it carries its own permission.
                permission=PROCUREMENT_PERMISSIONS.award,
                handler=award,
            ),
            CommandDefinition(
                id=PROCUREMENT_COMMANDS.record_field_status,
                title="Record field status",
                permission=PROCUREMENT_PERMISSIONS.record_field,
                handler=record_status,
            ),
            CommandDefinition(
                id=PROCUREMENT_COMMANDS.create_inspection,
                title="Record inspection",
                permission=PROCUREMENT_PERMISSIONS.inspect,
                handler=create_inspection,
            ),
            CommandDefinition(
                id=PROCUREMENT_COMMANDS.compute_progress,
                title="Compute install progress",
                handler=compute_progress,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="procurement.panel",
                point="panel",
                title="Procurement",
                placement="right",
                order=50,
            )
        )
        context.logger.info("Procurement and field capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Procurement and field",
        description="Packages from the bill, vendor comparison and award, element-level field "
        "status, inspection and earned value.",
        permissions=[
            PROCUREMENT_PERMISSIONS.manage_packages,
            PROCUREMENT_PERMISSIONS.award,
            PROCUREMENT_PERMISSIONS.record_field,
            PROCUREMENT_PERMISSIONS.inspect,
        ],
        activate=activate,
    )


procurement_plugin = create_procurement_plugin()
