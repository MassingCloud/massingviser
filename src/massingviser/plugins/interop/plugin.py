from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...kernel import CommandDefinition, KernelError, PluginContext, Result, UIContribution, err, ok
from ...sdk import (
    Clock,
    IdFactory,
    RecordStore,
    SystemClock,
    create_record_store,
    define_plugin,
)
from .contracts import (
    INTEROP_COMMANDS,
    INTEROP_EVENTS,
    INTEROP_PERMISSIONS,
    ConnectorPolicy,
    ExportAdapterToken,
    FormatDetection,
    ImportAdapterToken,
    ImportSummary,
    InteropToken,
)

PLUGIN_ID = "massingviser.interop"
PLUGIN_VERSION = "0.1.0"


class InteropServiceImpl:
    __slots__ = ("_context", "_clock", "_policies")

    def __init__(
        self, context: PluginContext, clock: Clock, policies: RecordStore[ConnectorPolicy]
    ) -> None:
        self._context = context
        self._clock = clock
        self._policies = policies

    def _importers(self) -> list[Any]:
        return [p.value for p in self._context.capabilities.get_all(ImportAdapterToken)]

    def _exporters(self) -> list[Any]:
        return [p.value for p in self._context.capabilities.get_all(ExportAdapterToken)]

    def formats(self) -> tuple[str, ...]:
        return tuple(sorted({adapter.format for adapter in self._importers()}))

    def detect(self, payload: bytes, filename: str | None = None) -> FormatDetection:
        """Sniff the bytes, then compare with what the name claimed."""
        claimed: str | None = None
        if filename and "." in filename:
            suffix = filename.rsplit(".", 1)[-1].lower()
            claimed = next(
                (
                    adapter.format
                    for adapter in self._importers()
                    if suffix in {e.lower().lstrip(".") for e in adapter.extensions}
                ),
                None,
            )

        for adapter in self._importers():
            for signature in adapter.signatures:
                if signature and payload.startswith(signature):
                    return FormatDetection(
                        format=adapter.format,
                        confidence="certain",
                        claimed_format=claimed,
                        detail=f"matched signature {signature!r}",
                    )

        if claimed is not None:
            # Nothing in the bytes confirmed it. Reported as *claimed*, so a caller can decide
            # whether to trust a filename -- rather than being told "detected" and believing it.
            return FormatDetection(
                format=claimed,
                confidence="claimed",
                claimed_format=claimed,
                detail="no signature matched; the extension is the only evidence",
            )
        return FormatDetection(format=None, confidence="unknown", detail="unrecognised content")

    def set_policy(self, policy: ConnectorPolicy) -> None:
        self._policies.remove_where(lambda p: p.format == policy.format)
        self._policies.add(policy)

    def policy(self, format: str) -> ConnectorPolicy:
        found = self._policies.find(lambda p: p.format == format)
        # Unknown formats default to review rather than to trusted. A governance default that
        # permits is not governance.
        return found or ConnectorPolicy(format=format, trust="review")

    def policies(self) -> tuple[ConnectorPolicy, ...]:
        return self._policies.all()

    async def import_payload(
        self, payload: bytes, *, filename: str | None = None, format: str | None = None
    ) -> Result[ImportSummary, KernelError]:
        detection = self.detect(payload, filename)
        chosen = format or detection.format
        if chosen is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    "No installed adapter recognises this content, and the filename gave no "
                    "usable hint.",
                    {"filename": filename},
                )
            )

        policy = self.policy(chosen)
        if policy.trust == "blocked":
            self._context.events.emit(
                INTEROP_EVENTS.blocked, {"format": chosen, "reason": policy.reason}
            )
            return err(
                KernelError(
                    "PERMISSION_DENIED",
                    f'Importing "{chosen}" is blocked by policy'
                    + (f": {policy.reason}" if policy.reason else "."),
                    {"format": chosen},
                )
            )

        adapter = next((a for a in self._importers() if a.format == chosen), None)
        if adapter is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    f'No import adapter for "{chosen}".',
                    {"format": chosen, "available": list(self.formats())},
                )
            )

        result = await adapter.read(payload)
        if not result.ok:
            return err(result.error)

        summary = result.value
        if detection.disputed:
            # The disagreement travels with the import rather than being resolved silently.
            summary = ImportSummary(
                format=summary.format,
                records=summary.records,
                rejected=summary.rejected,
                warnings=(
                    *summary.warnings,
                    f'the filename claimed "{detection.claimed_format}" but the content is '
                    f'"{detection.format}"',
                ),
            )
        self._context.events.emit(INTEROP_EVENTS.imported, {"summary": summary})
        return ok(summary)

    async def export(self, format: str, **options: Any) -> Result[bytes, KernelError]:
        adapter = next((a for a in self._exporters() if a.format == format), None)
        if adapter is None:
            return err(
                KernelError(
                    "CAPABILITY_NOT_FOUND",
                    f'No export adapter for "{format}".',
                    {"format": format},
                )
            )
        written = await adapter.write(**options)
        if written.ok:
            self._context.events.emit(
                INTEROP_EVENTS.exported, {"format": format, "bytes": len(written.value)}
            )
        return written


def create_interop_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    """Format detection, import/export dispatch and connector governance."""
    resolved_clock = clock or SystemClock()

    def activate(context: PluginContext) -> None:
        policies: RecordStore[ConnectorPolicy] = create_record_store(context.state, "policies")
        service = InteropServiceImpl(context, resolved_clock, policies)
        context.capabilities.provide(InteropToken, service, version=PLUGIN_VERSION)

        def detect(params: Mapping[str, Any], _ctx: Any) -> Any:
            return service.detect(params["payload"], params.get("filename"))

        async def import_payload(params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.import_payload(
                params["payload"],
                filename=params.get("filename"),
                format=params.get("format"),
            )
            if not result.ok:
                raise result.error
            return result.value

        async def export(params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.export(params["format"], **params.get("options", {}))
            if not result.ok:
                raise result.error
            return result.value

        def set_policy(params: Mapping[str, Any], _ctx: Any) -> Any:
            service.set_policy(
                ConnectorPolicy(
                    format=params["format"],
                    trust=params.get("trust", "review"),
                    reason=params.get("reason"),
                    reviewed_at=resolved_clock.iso(),
                )
            )
            return service.policy(params["format"])

        for command in (
            CommandDefinition(id=INTEROP_COMMANDS.detect, title="Detect format", handler=detect),
            CommandDefinition(
                id=INTEROP_COMMANDS.import_payload,
                title="Import",
                permission=INTEROP_PERMISSIONS.import_data,
                handler=import_payload,
            ),
            CommandDefinition(
                id=INTEROP_COMMANDS.export,
                title="Export",
                permission=INTEROP_PERMISSIONS.export_data,
                handler=export,
            ),
            CommandDefinition(
                id=INTEROP_COMMANDS.set_policy,
                title="Set connector policy",
                permission=INTEROP_PERMISSIONS.govern,
                handler=set_policy,
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="interop.panel", point="panel", title="Import/Export", placement="left", order=60
            )
        )
        context.logger.info("Interop capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Interop",
        description="Content-first format detection, import/export dispatch, connector governance.",
        permissions=[
            INTEROP_PERMISSIONS.import_data,
            INTEROP_PERMISSIONS.export_data,
            INTEROP_PERMISSIONS.govern,
        ],
        activate=activate,
    )


interop_plugin = create_interop_plugin()
