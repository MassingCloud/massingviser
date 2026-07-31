"""Coordinate reference systems, via pyproj.

``GeoReference`` has recorded source CRS, vertical datum, origin offset and *how the georeference
was established* since the schema was written. Nothing transformed between them until now, which
meant the platform could say where a model claimed to be and could not check.

Two things this makes possible that a transform matrix alone cannot:

- **Reconcile two datasets on different datums.** A survey in EPSG:27700 and a scan in EPSG:4326
  agree about the building or they do not, and finding out is a projection, not an argument.
- **Render near the origin without lying about it.** A British National Grid easting is around
  530000; a 32-bit float carries about seven significant digits, so geometry at true coordinates
  jitters and z-fights. The origin offset makes the local frame *reversible* rather than a fudge.

An optional extra. Without pyproj the platform still records georeferences; it just cannot project
between them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from ..kernel import KernelError, Result, err, ok
from ..schema import GeoReference


@dataclass(frozen=True)
class CrsInfo:
    code: str
    name: str
    is_projected: bool
    is_geographic: bool
    #: Axis unit of the horizontal axes, e.g. ``metre`` or ``degree``.
    unit: str
    #: True when the CRS carries a vertical component of its own.
    has_vertical: bool


def describe(code: str) -> Result[CrsInfo, KernelError]:
    """Resolve an authority-qualified code to something the platform can reason about."""
    try:
        crs = CRS.from_user_input(code)
    except CRSError as thrown:
        return err(
            KernelError("COMMAND_FAILED", f'"{code}" is not a CRS pyproj recognises: {thrown}', {})
        )
    axis = crs.axis_info[0] if crs.axis_info else None
    return ok(
        CrsInfo(
            code=code,
            name=crs.name,
            is_projected=bool(crs.is_projected),
            is_geographic=bool(crs.is_geographic),
            unit=getattr(axis, "unit_name", "unknown") if axis else "unknown",
            has_vertical=bool(crs.is_vertical) or len(crs.axis_info) > 2,
        )
    )


class CoordinateTransformer:
    """Projects points between two CRSs, honouring the origin offset in both directions.

    ``always_xy`` is set deliberately. Half of EPSG's geographic systems declare latitude first, and
    a library that honours that faithfully will silently swap a project's X and Y the moment
    somebody switches from a projected CRS to a geographic one. Forcing easting-northing ordering
    everywhere makes the platform's own convention the only one in play.
    """

    __slots__ = ("_forward", "_inverse", "source", "target", "_offset")

    def __init__(
        self,
        source: str,
        target: str,
        *,
        origin_offset: Sequence[float] | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self._offset = np.asarray(origin_offset or (0.0, 0.0, 0.0), dtype=np.float64)
        self._forward = Transformer.from_crs(source, target, always_xy=True)
        self._inverse = Transformer.from_crs(target, source, always_xy=True)

    def to_world(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Local scene coordinates to true coordinates in the target CRS."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3) + self._offset
        x, y, z = self._forward.transform(array[:, 0], array[:, 1], array[:, 2])
        return np.column_stack([x, y, z])

    def to_local(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """True coordinates back to the local frame the renderer works in."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        x, y, z = self._inverse.transform(array[:, 0], array[:, 1], array[:, 2])
        return np.column_stack([x, y, z]) - self._offset


def transformer_for(
    reference: GeoReference, target: str = "EPSG:4326"
) -> Result[CoordinateTransformer, KernelError]:
    """Build a transformer from a ``GeoReference`` record."""
    for code in (reference.source_crs, target):
        described = describe(code)
        if not described.ok:
            return err(described.error)
    try:
        return ok(
            CoordinateTransformer(
                reference.source_crs, target, origin_offset=reference.origin_offset
            )
        )
    except Exception as thrown:  # noqa: BLE001
        return err(
            KernelError(
                "COMMAND_FAILED",
                f"No transformation from {reference.source_crs} to {target}: {thrown}",
                {},
            )
        )


@dataclass(frozen=True)
class GeoValidation:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_georeference(reference: GeoReference) -> GeoValidation:
    """Check a georeference against what pyproj actually knows.

    The distinction the schema draws between ``survey`` and ``assumed`` is preserved here rather
    than collapsed: an unverified georeference is a *warning*, not an error, because plenty of early
    work is legitimately assumed -- but it is said out loud, every time, so it cannot quietly become
    treated as measured.
    """
    errors: list[str] = []
    warnings: list[str] = []

    described = describe(reference.source_crs)
    if not described.ok:
        errors.append(described.error.message)
    else:
        info = described.value
        if info.is_geographic and reference.units == "m":
            # Degrees are not metres, and a model that says otherwise will be 111 km per unit out.
            errors.append(
                f'"{reference.source_crs}" is geographic (axes in {info.unit}) but the reference '
                'declares units of "m"'
            )
        if info.is_projected and reference.origin_offset is None:
            magnitude = 1e5
            warnings.append(
                f'"{reference.source_crs}" is projected, so coordinates run to around {magnitude:g}; '
                "with no origin offset a 32-bit renderer will jitter"
            )

    if reference.target_crs:
        described_target = describe(reference.target_crs)
        if not described_target.ok:
            errors.append(described_target.error.message)

    if reference.vertical_datum is None:
        # Two datasets can agree exactly in plan and sit a metre apart in height.
        warnings.append("no vertical datum: heights cannot be reconciled with another dataset")
    if not reference.verified:
        warnings.append(
            f'georeference method is "{reference.method or "unstated"}", not survey or control '
            "points -- treat positions as provisional"
        )
    return GeoValidation(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
