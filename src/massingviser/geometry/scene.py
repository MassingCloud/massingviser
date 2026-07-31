"""A spatial index over a whole scene, and the questions a client would otherwise ask itself.

Three operations, all answered server-side:

- **pick** -- what is under this ray, nearest first;
- **cull** -- what is inside this camera frustum;
- **clash** -- which elements of one set meet another.

All three are the same tree descent, and all three return **GlobalIds**, so an answer from the
index is already in the identity that markup, cost, coordination and the engine bridge all key on.
A client that asks "what did I click" gets back something it can immediately look up.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..kernel import CapabilityToken, create_capability_token
from .bvh import Aabb, Bvh


@dataclass(frozen=True)
class Pick:
    global_id: str
    #: Distance along the ray, in metres.
    distance: float


@dataclass(frozen=True)
class ClashCandidate:
    a: str
    b: str
    #: Overlap depth along the shallowest axis, in metres. Negative for a clearance shortfall.
    penetration: float


def frustum_from_matrix(
    view_projection: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    """Extract six inward-facing planes from a column-major view-projection matrix.

    Column-major with translation at 12-14, matching every transform in this platform. The
    Gribb-Hartmann extraction: each plane is a sum or difference of two matrix rows, which is why
    it works for orthographic and perspective alike without special-casing either.
    """
    m = list(view_projection)
    if len(m) != 16:
        raise ValueError("A view-projection matrix has sixteen elements.")

    def row(index: int) -> tuple[float, float, float, float]:
        return (m[index], m[index + 4], m[index + 8], m[index + 12])

    x, y, z, w = row(0), row(1), row(2), row(3)
    planes = [
        tuple(w[i] + x[i] for i in range(4)),  # left
        tuple(w[i] - x[i] for i in range(4)),  # right
        tuple(w[i] + y[i] for i in range(4)),  # bottom
        tuple(w[i] - y[i] for i in range(4)),  # top
        tuple(w[i] + z[i] for i in range(4)),  # near
        tuple(w[i] - z[i] for i in range(4)),  # far
    ]
    normalised = []
    for plane in planes:
        length = math.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        # Normalising matters: an unnormalised plane still classifies points correctly but its
        # distances are meaningless, and the box test uses them.
        normalised.append(tuple(value / length for value in plane) if length else plane)
    return tuple(normalised)


class SceneIndex:
    """A rebuilt-on-change spatial index over labelled boxes.

    Rebuilt wholesale rather than updated incrementally. A refit BVH degrades as elements move, and
    a building's index is tens of thousands of boxes -- rebuilding is milliseconds, and always
    correct beats usually correct.
    """

    __slots__ = ("_bvh", "_boxes", "_groups")

    def __init__(
        self,
        boxes: Mapping[str, Aabb],
        *,
        groups: Mapping[str, str] | None = None,
    ) -> None:
        self._boxes = dict(boxes)
        #: GlobalId -> the set it belongs to (a model, a discipline). Clash needs to know which
        #: side an element is on, and an element belongs to exactly one.
        self._groups = dict(groups or {})
        labels = list(self._boxes)
        self._bvh = Bvh(labels, [self._boxes[label] for label in labels])

    def __len__(self) -> int:
        return len(self._boxes)

    @property
    def bounds(self) -> Aabb | None:
        return self._bvh.bounds()

    def box_of(self, global_id: str) -> Aabb | None:
        return self._boxes.get(global_id)

    def labels(self) -> tuple[str, ...]:
        """Everything the index holds, in a stable order.

        Exposed because callers legitimately need to ask *what is in here* -- an id keyed at a
        different granularity than the caller has, for instance -- and the alternative is each of
        them keeping a second copy of the same list and letting it drift.
        """
        return tuple(sorted(self._boxes))

    def pick(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        max_distance: float = math.inf,
        limit: int = 16,
    ) -> tuple[Pick, ...]:
        hits = self._bvh.raycast(origin, direction, max_distance=max_distance)
        return tuple(Pick(global_id, distance) for global_id, distance in hits[:limit])

    def cull(self, view_projection: Sequence[float]) -> tuple[str, ...]:
        """What a camera can see. The client sends its matrix and gets back ids."""
        return self._bvh.query_frustum(frustum_from_matrix(view_projection))

    def within(self, box: Aabb, *, tolerance: float = 0.0) -> tuple[str, ...]:
        return self._bvh.query_aabb(box, tolerance=tolerance)

    def clash(
        self, left_group: str, right_group: str, *, tolerance: float = 0.0
    ) -> tuple[ClashCandidate, ...]:
        """Broad-phase clash between two groups.

        Broad-phase only, and it says so. Bounding boxes overlapping is a *candidate*, not a
        collision -- two diagonal braces crossing in plan have overlapping boxes and may miss
        entirely. Narrow-phase needs the actual solids, which is a geometry-kernel job behind its
        own capability; this is the pass that makes the narrow one tractable.
        """
        left = self._subset(left_group)
        right = self._subset(right_group)
        if left is None or right is None:
            return ()
        seen: set[tuple[str, str]] = set()
        found: list[ClashCandidate] = []
        for a, b, penetration in left.overlapping_pairs(right, tolerance=tolerance):
            if a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            found.append(ClashCandidate(a=key[0], b=key[1], penetration=penetration))
        return tuple(found)

    def _subset(self, group: str) -> Bvh | None:
        labels = [label for label, name in self._groups.items() if name == group]
        if not labels:
            return None
        return Bvh(labels, [self._boxes[label] for label in labels])

    @staticmethod
    def from_extrusions(
        elements: Iterable[tuple[str, str, Sequence[Sequence[float]], float, float]],
    ) -> SceneIndex:
        """Build from ``(global_id, group, footprint, base, height)`` tuples.

        The shape massing produces: a plan outline extruded between two elevations.
        """
        boxes: dict[str, Aabb] = {}
        groups: dict[str, str] = {}
        for global_id, group, footprint, base, height in elements:
            if not footprint:
                continue
            xs = [float(point[0]) for point in footprint]
            ys = [float(point[1]) for point in footprint]
            boxes[global_id] = Aabb(
                (min(xs), min(ys), float(base)), (max(xs), max(ys), float(base) + float(height))
            )
            groups[global_id] = group
        return SceneIndex(boxes, groups=groups)


#: What a client asks the server about space.
#:
#: Declared as a capability rather than a concrete class so the index can be backed by massing
#: today and by a fragments/IFC pipeline tomorrow without the caller changing. Everything it
#: returns is a GlobalId, so an answer is directly usable by markup, cost and coordination.
#:
#: Providers implement ``pick(origin, direction, **options)``, ``cull(view_projection)`` and
#: ``groups()``.
SpatialIndexToken: CapabilityToken[object] = create_capability_token("geometry.spatial-index")
