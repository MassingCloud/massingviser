"""``massingviser.plugins.interop`` -- getting data in and out.

Format detection is **content-first**. An extension is a claim made by whoever last renamed the
file, and acting on it is how a mislabelled ``.ifc`` that is really a zip gets handed to an IFC
parser and produces a confusing error three layers down. Sniffing the bytes and reporting a
disagreement is cheaper than debugging that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from ...kernel import CapabilityToken, KernelError, Result, create_capability_token
from ...schema import Id, IsoTimestamp

#: How much confidence a detection carries.
#:
#: ``certain`` means a magic number matched. ``likely`` means structure matched but not a
#: signature. ``claimed`` means nothing matched and only the extension says so -- which is exactly
#: the case a caller should be told about rather than shielded from.
Confidence = Literal["certain", "likely", "claimed", "unknown"]


@dataclass(frozen=True)
class FormatDetection:
    format: str | None
    confidence: Confidence
    #: What the filename claimed, when that disagrees with the bytes.
    claimed_format: str | None = None
    detail: str | None = None

    @property
    def disputed(self) -> bool:
        """The bytes and the name disagree. Almost always worth surfacing."""
        return (
            self.claimed_format is not None
            and self.format is not None
            and self.claimed_format != self.format
        )


@dataclass(frozen=True)
class ImportSummary:
    format: str
    records: int
    #: Rows, entities or nodes the adapter could not read. Named, never silently dropped.
    rejected: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class ImportAdapter(Protocol):
    """Reads one format. Registered as a capability, so a new format is a new plugin."""

    @property
    def format(self) -> str: ...
    #: Magic-number prefixes that identify this format with certainty.
    @property
    def signatures(self) -> Sequence[bytes]: ...
    @property
    def extensions(self) -> Sequence[str]: ...
    async def read(self, payload: bytes) -> Result[ImportSummary, KernelError]: ...


ImportAdapterToken: CapabilityToken[ImportAdapter] = create_capability_token("interop.import")


@runtime_checkable
class ExportAdapter(Protocol):
    @property
    def format(self) -> str: ...
    async def write(self, **options: Any) -> Result[bytes, KernelError]: ...


ExportAdapterToken: CapabilityToken[ExportAdapter] = create_capability_token("interop.export")


TrustLevel = Literal["trusted", "review", "blocked"]


@dataclass(frozen=True)
class ConnectorPolicy:
    """Governance over what may be imported from where.

    A federated platform accepts files from outside the organisation. Deciding per-format whether
    that is allowed -- and recording the decision -- is cheaper than discovering afterwards that
    somebody imported a supplier's model into a live project.
    """

    format: str
    trust: TrustLevel = "review"
    reason: str | None = None
    reviewed_at: IsoTimestamp | None = None


@runtime_checkable
class InteropService(Protocol):
    #: Sniffs bytes first, filename second, and says which one it believed.
    def detect(self, payload: bytes, filename: str | None = None) -> FormatDetection: ...
    def formats(self) -> tuple[str, ...]: ...
    async def import_payload(
        self, payload: bytes, *, filename: str | None = None, format: str | None = None
    ) -> Result[ImportSummary, KernelError]: ...
    async def export(self, format: str, **options: Any) -> Result[bytes, KernelError]: ...

    def set_policy(self, policy: ConnectorPolicy) -> None: ...
    def policy(self, format: str) -> ConnectorPolicy: ...
    def policies(self) -> tuple[ConnectorPolicy, ...]: ...


InteropToken: CapabilityToken[InteropService] = create_capability_token("interop.service")


class INTEROP_COMMANDS:
    detect = "interop.detect"
    import_payload = "interop.import"
    export = "interop.export"
    set_policy = "interop.policy.set"


class INTEROP_PERMISSIONS:
    import_data = "interop.import"
    export_data = "interop.export"
    govern = "interop.govern"


class INTEROP_EVENTS:
    imported = "interop.imported"
    exported = "interop.exported"
    blocked = "interop.blocked"
