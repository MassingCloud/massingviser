"""Narrow-phase clash, via trimesh and manifold3d.

The BVH in ``geometry`` finds *candidates* -- pairs whose bounding boxes meet. That is the pass that
makes this one tractable, and on its own it over-reports badly: two diagonal braces crossing in plan
have overlapping boxes and may miss by a metre.

This adapter takes each candidate and intersects the actual solids. What comes back is a volume,
which is a far better number than box penetration for triage: an estimator or an engineer looking at
a clash list wants to know whether two things overlap by a litre or a cubic metre, and a bounding
box cannot tell them apart.

Registered against the same ``ClashEngineToken`` the BVH engine uses, at a higher priority. Nothing
in coordination changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import trimesh

from ..geometry import Aabb, SceneIndex
from ..plugins.coordination import RawClash
from ..schema import ElementRef

#: Below this, an intersection is numerical noise from two coplanar faces rather than a clash.
#: A litre is small enough to catch anything real and large enough to reject touching surfaces,
#: which in a real model are everywhere -- every slab meets every wall.
MINIMUM_VOLUME = 1e-3


@runtime_checkable
class MeshSource(Protocol):
    """Where solids come from. The IFC adapter is one; a fragments reader would be another."""

    def meshes(self) -> Mapping[str, tuple[Any, Any]]: ...
    def boxes(self) -> Mapping[str, Aabb]: ...


@dataclass(frozen=True)
class SolidClash:
    a: str
    b: str
    #: Cubic metres of overlap. The number a triage decision is actually made on.
    volume: float
    centre: tuple[float, float, float]


class SolidClashEngine:
    """Broad-phase then narrow-phase, in that order and for that reason."""

    __slots__ = ("_source", "_model_id", "_groups", "_cache")

    def __init__(
        self,
        source: MeshSource,
        *,
        model_id: str,
        groups: Mapping[str, str] | None = None,
    ) -> None:
        self._source = source
        self._model_id = model_id
        #: GlobalId -> which side of a clash test it is on. Without groups every element is tested
        #: against every other, including its own discipline, which is noise.
        self._groups = dict(groups or {})
        self._cache: dict[str, trimesh.Trimesh] = {}

    def _mesh(self, global_id: str) -> trimesh.Trimesh | None:
        if global_id in self._cache:
            return self._cache[global_id]
        raw = self._source.meshes().get(global_id)
        if raw is None:
            return None
        vertices, faces = raw
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
            faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
            process=False,
        )
        self._cache[global_id] = mesh
        return mesh

    def pairs(self, tolerance: float = 0.0) -> tuple[SolidClash, ...]:
        boxes = dict(self._source.boxes())
        if not boxes:
            return ()
        groups = self._groups or {global_id: "all" for global_id in boxes}
        index = SceneIndex(boxes, groups=groups)

        names = sorted(set(groups.values()))
        candidates: list[tuple[str, str]] = []
        if len(names) == 1:
            # One group: everything against everything, which the BVH still makes cheap.
            for candidate in index.clash(names[0], names[0], tolerance=tolerance):
                candidates.append((candidate.a, candidate.b))
        else:
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    for candidate in index.clash(left, right, tolerance=tolerance):
                        candidates.append((candidate.a, candidate.b))

        found: list[SolidClash] = []
        for a, b in candidates:
            if a == b:
                continue
            volume, centre = self._intersect(a, b)
            if volume is None or volume < MINIMUM_VOLUME:
                continue
            found.append(SolidClash(a=a, b=b, volume=volume, centre=centre))
        return tuple(sorted(found, key=lambda clash: -clash.volume))

    def _intersect(self, a: str, b: str) -> tuple[float | None, tuple[float, float, float]]:
        left = self._mesh(a)
        right = self._mesh(b)
        if left is None or right is None:
            return None, (0.0, 0.0, 0.0)
        try:
            overlap = left.intersection(right)
        except Exception:  # noqa: BLE001
            # A boolean on degenerate or non-manifold input fails, and a failed narrow-phase must
            # not lose the candidate: fall back to reporting it on box evidence alone.
            return self._fallback(a, b)
        if overlap is None or overlap.is_empty or len(overlap.faces) == 0:
            return None, (0.0, 0.0, 0.0)
        try:
            centre = tuple(float(value) for value in overlap.centroid)
        except Exception:  # noqa: BLE001
            centre = (0.0, 0.0, 0.0)
        return abs(float(overlap.volume)), centre

    def _fallback(self, a: str, b: str) -> tuple[float | None, tuple[float, float, float]]:
        boxes = self._source.boxes()
        left, right = boxes.get(a), boxes.get(b)
        if left is None or right is None:
            return None, (0.0, 0.0, 0.0)
        spans = [
            max(0.0, min(left.max[axis], right.max[axis]) - max(left.min[axis], right.min[axis]))
            for axis in range(3)
        ]
        centre = tuple(
            (max(left.min[axis], right.min[axis]) + min(left.max[axis], right.max[axis])) / 2
            for axis in range(3)
        )
        return spans[0] * spans[1] * spans[2], centre

    # -- coordination.ClashEngine -------------------------------------------------------------

    def intersect(
        self, a: Sequence[Any], b: Sequence[Any], kind: str, tolerance: float
    ) -> Sequence[RawClash]:
        return tuple(
            RawClash(
                a=ElementRef(self._model_id, clash.a),
                b=ElementRef(self._model_id, clash.b),
                point=clash.centre,
                # Reported as volume, not depth. Coordination stores it as `distance`, and the
                # unit travels in the clash test's `kind` rather than being guessed at.
                distance=round(clash.volume, 9),
            )
            for clash in self.pairs(tolerance)
        )
