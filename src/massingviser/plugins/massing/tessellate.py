"""Turning a massing profile into triangles.

This has no counterpart in ``massingifc``, which ships no viewer and therefore never needs a mesh.
It is the half MassingViser adds -- but it stays here, in the capability plugin, rather than in the
viewer, and it holds to the same rule as everything else: **no rendering library is imported**.
Output is plain Python tuples, and the viser shell converts to arrays at its own boundary. That is
what lets the tessellator be unit-tested headlessly and reused by an exporter that has never heard
of viser.

Ear clipping is implemented here rather than taken from a dependency because the alternative --
``shapely``/``mapbox_earcut``/``triangle`` -- is an optional extra of an optional extra, and a
massing tool that cannot draw a floor plate unless a C extension happens to be installed is not a
massing tool. Footprints have tens of vertices; O(n^2) is free at that size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .geometry import EPSILON, is_clockwise, normalise_winding, polygon_area, signed_area

Point2 = "tuple[float, float]"
Point3 = "tuple[float, float, float]"


@dataclass(frozen=True)
class Mesh:
    """A triangle soup in metres, Z up.

    Kept as plain tuples so nothing here depends on numpy; the shell converts once, at the edge.
    """

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]

    @property
    def is_empty(self) -> bool:
        return not self.faces

    def merged_with(self, other: "Mesh") -> "Mesh":
        offset = len(self.vertices)
        return Mesh(
            vertices=self.vertices + other.vertices,
            faces=self.faces + tuple((a + offset, b + offset, c + offset) for a, b, c in other.faces),
        )


EMPTY_MESH = Mesh((), ())


def merge(meshes: Sequence[Mesh]) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        faces.extend((a + offset, b + offset, c + offset) for a, b, c in mesh.faces)
    return Mesh(tuple(vertices), tuple(faces))


# ---------------------------------------------------------------------------------------------
# Ear clipping
# ---------------------------------------------------------------------------------------------


def _cross(
    o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _coincident(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < EPSILON and abs(a[1] - b[1]) < EPSILON


def _point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_negative = d1 < -EPSILON or d2 < -EPSILON or d3 < -EPSILON
    has_positive = d1 > EPSILON or d2 > EPSILON or d3 > EPSILON
    # Strictly inside or on an edge, but not straddling -- a vertex sitting exactly on the ear's
    # edge still blocks the clip, because clipping it would produce a zero-area sliver.
    return not (has_negative and has_positive)


def ear_clip(ring: Sequence[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Triangulate a simple counter-clockwise ring, returning index triples into ``ring``."""
    count = len(ring)
    if count < 3:
        return []

    indices = list(range(count))
    if is_clockwise(ring):
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []
    # Each successful clip removes one vertex, so n-2 clips finish the job. The guard bounds the
    # pathological case where no ear is found (a ring that is not actually simple) instead of
    # spinning forever.
    guard = 0
    max_iterations = count * count + 16

    while len(indices) > 3 and guard < max_iterations:
        guard += 1
        clipped = False
        size = len(indices)
        for position in range(size):
            i_prev = indices[position - 1]
            i_curr = indices[position]
            i_next = indices[(position + 1) % size]
            a, b, c = ring[i_prev], ring[i_curr], ring[i_next]

            if _cross(a, b, c) <= EPSILON:
                continue  # reflex or collinear: not an ear

            blocked = False
            for offset, other in enumerate(indices):
                if other in (i_prev, i_curr, i_next):
                    continue
                point = ring[other]
                # Bridged holes leave two vertices at identical coordinates. One of them is a
                # corner of this very ear, so testing containment on the other would block every
                # candidate and the ring would never clip -- which is exactly what a naive
                # implementation does to any polygon with a hole.
                if _coincident(point, a) or _coincident(point, b) or _coincident(point, c):
                    continue
                # Only reflex vertices can intrude into an ear. Convex ones are already accounted
                # for by their own neighbours, and testing them costs correctness as well as time
                # once duplicate points are in play.
                prev_point = ring[indices[offset - 1]]
                next_point = ring[indices[(offset + 1) % size]]
                if _cross(prev_point, point, next_point) > EPSILON:
                    continue
                if _point_in_triangle(point, a, b, c):
                    blocked = True
                    break
            if blocked:
                continue

            triangles.append((i_prev, i_curr, i_next))
            indices.pop(position)
            clipped = True
            break

        if not clipped:
            # No ear anywhere: the ring is self-intersecting or fully degenerate. Emit what we have
            # rather than looping; `validate_profile` is the place that reports this as an error.
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _bridge_one_hole(
    ring: list[tuple[float, float]], hole: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Splice a hole into an outer ring with a zero-width bridge.

    The classic Eberly construction: take the hole's rightmost vertex, cast a ray to +x, and join
    it to the outer ring at the closest mutually-visible vertex.
    """
    if len(hole) < 3:
        return ring

    # The hole must wind opposite to the outer ring, or the bridge folds the ring onto itself.
    hole_ring = list(reversed(hole)) if not is_clockwise(hole) else list(hole)

    m_index = max(range(len(hole_ring)), key=lambda i: hole_ring[i][0])
    m = hole_ring[m_index]

    # Closest ring edge hit by the ray from M towards +x.
    best_x = float("inf")
    best_edge: int | None = None
    for i in range(len(ring)):
        a = ring[i]
        b = ring[(i + 1) % len(ring)]
        if (a[1] > m[1]) == (b[1] > m[1]):
            continue  # edge does not straddle the ray
        t = (m[1] - a[1]) / (b[1] - a[1])
        x = a[0] + t * (b[0] - a[0])
        if x >= m[0] - EPSILON and x < best_x:
            best_x = x
            best_edge = i

    if best_edge is None:
        # The hole is not inside the ring. `validate_profile` reports that; here we simply refuse
        # to produce a corrupt bridge.
        return ring

    a = ring[best_edge]
    b = ring[(best_edge + 1) % len(ring)]
    p_index = best_edge if a[0] > b[0] else (best_edge + 1) % len(ring)

    # A reflex vertex inside the triangle (M, intersection, P) would make the bridge cross the
    # ring; the standard fix is to re-aim at whichever such vertex sits at the smallest angle to
    # the ray.
    intersection = (best_x, m[1])
    p = ring[p_index]
    best_angle = float("inf")
    for i, candidate in enumerate(ring):
        if i == p_index:
            continue
        if not _point_in_triangle(candidate, m, intersection, p):
            continue
        prev_v = ring[i - 1]
        next_v = ring[(i + 1) % len(ring)]
        if _cross(prev_v, candidate, next_v) > EPSILON:
            continue  # convex vertices cannot block visibility
        dx = candidate[0] - m[0]
        dy = abs(candidate[1] - m[1])
        distance = (dx * dx + dy * dy) ** 0.5
        angle = dy / distance if distance > EPSILON else 0.0
        if angle < best_angle:
            best_angle = angle
            p_index = i

    rotated = hole_ring[m_index:] + hole_ring[:m_index]
    return (
        ring[: p_index + 1]
        + rotated
        + [rotated[0]]
        + [ring[p_index]]
        + ring[p_index + 1 :]
    )


def triangulate(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Triangulate a polygon with optional holes.

    Returns the working ring (holes bridged in, so it may contain duplicated points) and the
    triangles as indices into it.
    """
    ring = normalise_winding(outer)
    for hole in sorted(holes, key=lambda h: -max((p[0] for p in h), default=0.0)):
        if len(hole) >= 3 and polygon_area(hole) > EPSILON:
            ring = _bridge_one_hole(ring, hole)
    return ring, ear_clip(ring)


# ---------------------------------------------------------------------------------------------
# Extrusion
# ---------------------------------------------------------------------------------------------


def _wall_strip(
    ring: Sequence[tuple[float, float]], z0: float, z1: float, flip: bool
) -> Mesh:
    """A closed band of quads between two elevations.

    ``flip`` reverses the winding, which is how hole walls end up facing inwards -- a courtyard
    whose walls face out is a courtyard you can see through from the street.
    """
    count = len(ring)
    if count < 3:
        return EMPTY_MESH

    vertices: list[tuple[float, float, float]] = []
    for x, y in ring:
        vertices.append((x, y, z0))
        vertices.append((x, y, z1))

    faces: list[tuple[int, int, int]] = []
    for i in range(count):
        j = (i + 1) % count
        bottom_i, top_i = 2 * i, 2 * i + 1
        bottom_j, top_j = 2 * j, 2 * j + 1
        if flip:
            faces.append((bottom_i, top_i, bottom_j))
            faces.append((bottom_j, top_i, top_j))
        else:
            faces.append((bottom_i, bottom_j, top_i))
            faces.append((bottom_j, top_j, top_i))
    return Mesh(tuple(vertices), tuple(faces))


def _cap(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]],
    z: float,
    facing_up: bool,
) -> Mesh:
    ring, triangles = triangulate(outer, holes)
    if not triangles:
        return EMPTY_MESH
    vertices = tuple((x, y, z) for x, y in ring)
    if facing_up:
        faces = tuple(triangles)
    else:
        faces = tuple((c, b, a) for a, b, c in triangles)
    return Mesh(vertices, faces)


def extrude(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]],
    base_elevation: float,
    height: float,
) -> Mesh:
    """A closed solid: floor, ceiling, outer walls and one wall per hole."""
    if len(outer) < 3 or height <= 0:
        return EMPTY_MESH

    ring = normalise_winding(outer)
    top = base_elevation + height
    parts = [
        _cap(ring, holes, base_elevation, facing_up=False),
        _cap(ring, holes, top, facing_up=True),
        _wall_strip(ring, base_elevation, top, flip=False),
    ]
    for hole in holes:
        if len(hole) >= 3:
            parts.append(_wall_strip(normalise_winding(hole), base_elevation, top, flip=True))
    return merge(parts)


@dataclass(frozen=True)
class StoryMesh:
    index: int
    elevation: float
    height: float
    mesh: Mesh


def extrude_stories(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]],
    story_heights: Sequence[float],
    base_elevation: float = 0.0,
    #: Per-story outline override, keyed by story index -- setbacks and tapers.
    story_outlines: dict[int, Sequence[tuple[float, float]]] | None = None,
    #: Gap left between slabs so stories read as separate plates rather than one block.
    slab_gap: float = 0.0,
) -> list[StoryMesh]:
    """One solid per story.

    Massing is a story-aware tool, so the geometry is story-aware too: rendering a mass as a single
    extrusion makes it impossible to select, colour or hide a floor, which is most of what a user
    wants to do with one.
    """
    outlines = story_outlines or {}
    elevation = base_elevation
    stories: list[StoryMesh] = []
    for index, height in enumerate(story_heights):
        outline = outlines.get(index, outer)
        drawn = max(height - slab_gap, height * 0.5) if slab_gap else height
        stories.append(
            StoryMesh(
                index=index,
                elevation=elevation,
                height=height,
                mesh=extrude(outline, holes if outline is outer else (), elevation, drawn),
            )
        )
        elevation += height
    return stories
