"""Planar Procrustes: fitting captured reality onto project coordinates.

Planar rather than full 3D because that is what site registration actually is. A scan and a model
agree about which way is up -- both are gravity-referenced -- so the free parameters are a rotation
about Z, a translation, and sometimes a scale. Solving for a general 3D rotation instead lets a
noisy control point tip the whole dataset, which looks like a successful fit and puts the building
on a slope.

Pure maths over plain tuples: no numpy, so it stays testable and dependency-free like everything
else below the viewer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: Below this, the control points are effectively coincident and no rotation is recoverable.
EPSILON = 1e-9


@dataclass(frozen=True)
class PlanarFit:
    #: Rotation about Z, radians, counter-clockwise.
    rotation: float
    scale: float
    translation: tuple[float, float, float]
    #: Root-mean-square residual, in project units. The number that says whether to trust the fit.
    rms_error: float
    point_count: int

    def as_matrix(self) -> tuple[float, ...]:
        """Column-major 4x4, translation at indices 12-14.

        Stated because the two conventions are indistinguishable at the type level and a transposed
        matrix does not fail -- it just puts everything in the wrong place.
        """
        cos = math.cos(self.rotation) * self.scale
        sin = math.sin(self.rotation) * self.scale
        tx, ty, tz = self.translation
        return (
            cos, sin, 0.0, 0.0,
            -sin, cos, 0.0, 0.0,
            0.0, 0.0, self.scale, 0.0,
            tx, ty, tz, 1.0,
        )

    def apply(self, point: Sequence[float]) -> tuple[float, float, float]:
        cos = math.cos(self.rotation) * self.scale
        sin = math.sin(self.rotation) * self.scale
        x, y = point[0], point[1]
        z = point[2] if len(point) > 2 else 0.0
        return (
            cos * x - sin * y + self.translation[0],
            sin * x + cos * y + self.translation[1],
            self.scale * z + self.translation[2],
        )


def fit_planar(
    sources: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    allow_scale: bool = False,
) -> PlanarFit | None:
    """Least-squares similarity transform in the XY plane, with Z handled as a pure shift.

    Returns ``None`` when the inputs cannot determine one: fewer than two pairs, mismatched
    lengths, or control points that coincide. Returning a plausible identity instead would be
    worse -- an unregistered scan that reports itself as aligned is exactly the failure the
    ``method`` field on ``GeoReference`` exists to prevent.
    """
    if len(sources) != len(targets) or len(sources) < 2:
        return None

    count = len(sources)
    sx = sum(p[0] for p in sources) / count
    sy = sum(p[1] for p in sources) / count
    sz = sum((p[2] if len(p) > 2 else 0.0) for p in sources) / count
    tx = sum(p[0] for p in targets) / count
    ty = sum(p[1] for p in targets) / count
    tz = sum((p[2] if len(p) > 2 else 0.0) for p in targets) / count

    # Cross-covariance of the centred point sets, reduced to the two terms a planar rotation needs.
    numerator = 0.0
    denominator = 0.0
    source_energy = 0.0
    for source, target in zip(sources, targets):
        ax, ay = source[0] - sx, source[1] - sy
        bx, by = target[0] - tx, target[1] - ty
        denominator += ax * bx + ay * by
        numerator += ax * by - ay * bx
        source_energy += ax * ax + ay * ay

    if source_energy < EPSILON:
        return None  # every control point is in the same place

    rotation = math.atan2(numerator, denominator)
    scale = 1.0
    if allow_scale:
        magnitude = math.hypot(numerator, denominator)
        if magnitude < EPSILON:
            return None
        scale = magnitude / source_energy

    cos = math.cos(rotation) * scale
    sin = math.sin(rotation) * scale
    translation = (
        tx - (cos * sx - sin * sy),
        ty - (sin * sx + cos * sy),
        tz - scale * sz,
    )

    fit = PlanarFit(
        rotation=rotation,
        scale=scale,
        translation=translation,
        rms_error=0.0,
        point_count=count,
    )
    squared = 0.0
    for source, target in zip(sources, targets):
        px, py, pz = fit.apply(source)
        qz = target[2] if len(target) > 2 else 0.0
        squared += (px - target[0]) ** 2 + (py - target[1]) ** 2 + (pz - qz) ** 2
    return PlanarFit(
        rotation=rotation,
        scale=scale,
        translation=translation,
        rms_error=math.sqrt(squared / count),
        point_count=count,
    )
