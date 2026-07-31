"""Georeferencing.

Shared rather than owned by any one family: a reference model, a laser scan, a Gaussian splat and a
site boundary all need to say where on Earth they are, and they need to say it the same way. A
transform alone is not georeferencing -- it places geometry relative to a project origin nobody else
knows, which is why a scan aligned only by matrix cannot be checked against a survey or dropped into
a GIS scene.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

from .common import IsoTimestamp

#: An authority-qualified CRS code, e.g. ``"EPSG:27700"``.
#:
#: Kept as a string rather than an enum because the useful set is the EPSG registry, which is tens
#: of thousands of entries and changes without this codebase being rebuilt.
CrsCode = str

LinearUnit = Literal["m", "mm", "cm", "ft", "us-ft"]

#: Metres per unit.
#:
#: ``ft`` and ``us-ft`` differ by about 2 ppm -- which is metres across a large site, and the
#: reason the two are not collapsed into one entry.
METRES_PER_UNIT: Mapping[str, float] = MappingProxyType(
    {"m": 1.0, "mm": 0.001, "cm": 0.01, "ft": 0.3048, "us-ft": 1200 / 3937}
)

_CRS = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):([A-Za-z0-9._-]+)$")


@dataclass(frozen=True)
class ParsedCrs:
    authority: str
    code: str


def parse_crs_code(code: CrsCode) -> ParsedCrs | None:
    """Split ``"EPSG:27700"`` into its parts. ``None`` for anything not authority-qualified."""
    match = _CRS.match(code.strip())
    if not match:
        return None
    return ParsedCrs(authority=match.group(1).upper(), code=match.group(2))


def convert_length(value: float, from_unit: LinearUnit, to_unit: LinearUnit) -> float:
    """Convert a length between linear units, via metres."""
    if from_unit == to_unit:
        return value
    return value * METRES_PER_UNIT[from_unit] / METRES_PER_UNIT[to_unit]


GeoReferenceMethod = Literal[
    #: Established by a surveyor against known control.
    "survey",
    #: From GNSS on the capture device.
    "gps",
    #: Solved from matched control points.
    "control-points",
    #: Stated by a human without independent verification.
    "declared",
    #: Nothing verified it. Present so the absence of evidence is recorded rather than implied.
    "assumed",
]


@dataclass(frozen=True)
class GeoAccuracy:
    #: Metres, 1-sigma.
    horizontal: float | None = None
    vertical: float | None = None


@dataclass(frozen=True)
class GeoReference:
    #: CRS the source data is expressed in.
    source_crs: CrsCode
    units: LinearUnit = "m"
    #: CRS the project works in. Absent means the source CRS is also the working one.
    target_crs: CrsCode | None = None
    #: Vertical datum, e.g. ``"EGM2008"``, ``"ODN"``, ``"NAVD88"``.
    #:
    #: Separate from the horizontal CRS because they are genuinely independent: two datasets can
    #: agree exactly in plan and be a metre apart in height, which is the difference between a slab
    #: that clashes and one that does not.
    vertical_datum: str | None = None
    #: Offset subtracted from world coordinates to keep scene values near the origin.
    #:
    #: Projected coordinates run to six or seven digits (a British National Grid easting is around
    #: 530000). A 32-bit float holds about seven significant digits, so geometry rendered at true
    #: coordinates jitters visibly and z-fights. Every renderer works in a local frame; recording
    #: the offset is what makes that frame reversible instead of a lossy fudge.
    origin_offset: tuple[float, float, float] | None = None
    #: Local frame to georeferenced frame, when the data was captured in an engineering grid.
    local_to_global: tuple[float, ...] | None = None
    #: Rotation from project north to true north, degrees clockwise.
    true_north_angle: float | None = None
    method: GeoReferenceMethod | None = None
    accuracy: GeoAccuracy | None = None
    established_at: IsoTimestamp | None = None

    @property
    def verified(self) -> bool:
        """Whether the georeference rests on evidence rather than assertion."""
        return self.method in ("survey", "control-points")


@dataclass(frozen=True)
class Extent:
    """Axis-aligned bounds.

    ``z`` is optional because plan-only extents are common in GIS sources.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    zmin: float | None = None
    zmax: float | None = None


def extent_is_valid(extent: Extent) -> bool:
    if extent.xmax < extent.xmin or extent.ymax < extent.ymin:
        return False
    if extent.zmin is not None and extent.zmax is not None and extent.zmax < extent.zmin:
        return False
    return True


def extent_span(extent: Extent) -> float:
    """Longest horizontal span, in the extent's own units. Used for plausibility checks."""
    return max(extent.xmax - extent.xmin, extent.ymax - extent.ymin)


@dataclass(frozen=True)
class RealityDerivatives:
    """Products derived from the same capture as a reality dataset.

    Held as links rather than embedded because they arrive at different times -- an orthomosaic can
    take hours of external processing after the splat is already viewable -- and because they are
    usually large files a project references rather than owns.
    """

    source_imagery_uri: str | None = None
    orthomosaic_uri: str | None = None
    point_cloud_uri: str | None = None
    mesh_uri: str | None = None
    #: Digital surface model.
    dsm_uri: str | None = None
    #: Digital terrain model.
    dtm_uri: str | None = None


#: What a dataset may legitimately be used for.
#:
#: A Gaussian splat is superb for looking at and unsuitable for measuring: it is a view-dependent
#: radiance representation, not a surface. Recording intent lets the platform refuse to measure
#: against one instead of silently returning a number that looks like a dimension.
DatasetPurpose = Literal["visualization", "inspection", "context", "analysis"]

#: Purposes for which a dataset may back a measurement or a derived authored object.
ANALYSIS_PURPOSES: tuple[DatasetPurpose, ...] = ("analysis",)
