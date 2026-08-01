"""Planar geometry for massing.

Pure functions over plain tuples, with no dependency on a 3D library. Massing metrics are the
numbers a scheme is judged on -- area, GFA, volume -- so they need to be computable and testable
without a renderer, a WebGL context, or a loaded model.

Masses are vertical extrusions of a horizontal profile, so everything here works in the XY plane
and treats Z as elevation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Point2 = "tuple[float, float]"

#: Tolerance for coordinate comparison, in project units (metres). ~0.01 mm.
EPSILON = 1e-5


def to_xy(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in points]


def signed_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace area, signed.

    The sign carries the winding direction, which callers need in order to normalise orientation
    before extruding -- a profile sketched clockwise would otherwise produce inward-facing
    surfaces.
    """
    if len(points) < 3:
        return 0.0
    total = 0.0
    count = len(points)
    for index in range(count):
        current = points[index]
        nxt = points[(index + 1) % count]
        total += current[0] * nxt[1] - nxt[0] * current[1]
    return total / 2.0


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return abs(signed_area(points))


def is_clockwise(points: Sequence[tuple[float, float]]) -> bool:
    return signed_area(points) < 0


def normalise_winding(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the ring wound counter-clockwise, reversing only if needed."""
    return list(reversed(points)) if is_clockwise(points) else list(points)


def perimeter(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    count = len(points)
    for index in range(count):
        current = points[index]
        nxt = points[(index + 1) % count]
        total += math.hypot(nxt[0] - current[0], nxt[1] - current[1])
    return total


def centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Area centroid -- not the average of the vertices.

    The vertex average is wrong for any ring with unevenly spaced points, which is most real
    footprints. It matters because the centroid is what a mass is rotated and scaled about.
    """
    if not points:
        return (0.0, 0.0)
    area = signed_area(points)
    if abs(area) < EPSILON:
        # Degenerate ring: fall back to the vertex average rather than dividing by ~zero.
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
    cx = 0.0
    cy = 0.0
    count = len(points)
    for index in range(count):
        current = points[index]
        nxt = points[(index + 1) % count]
        cross = current[0] * nxt[1] - nxt[0] * current[1]
        cx += (current[0] + nxt[0]) * cross
        cy += (current[1] + nxt[1]) * cross
    return (cx / (6 * area), cy / (6 * area))


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < EPSILON:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]) -> bool:
    return (
        point[0] <= max(a[0], b[0]) + EPSILON
        and point[0] >= min(a[0], b[0]) - EPSILON
        and point[1] <= max(a[1], b[1]) + EPSILON
        and point[1] >= min(a[1], b[1]) - EPSILON
    )


def segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    # Collinear overlap still counts as an intersection for validation purposes.
    if o1 == 0 and _on_segment(a1, a2, b1):
        return True
    if o2 == 0 and _on_segment(a1, a2, b2):
        return True
    if o3 == 0 and _on_segment(b1, b2, a1):
        return True
    if o4 == 0 and _on_segment(b1, b2, a2):
        return True
    return False


def is_simple_polygon(points: Sequence[tuple[float, float]]) -> bool:
    """Whether a ring is simple (non-self-intersecting).

    O(n^2), which is the right trade here: footprints have tens of vertices, not thousands, and a
    sweep-line implementation would be considerably more code to get right for no felt benefit.
    """
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        a1 = points[i]
        a2 = points[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent edges legitimately share a vertex; the closing edge is adjacent to the
            # first.
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1 = points[j]
            b2 = points[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        a = polygon[i]
        b = polygon[j]
        if (a[1] > point[1]) != (b[1] > point[1]) and point[0] < (
            (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
        ):
            inside = not inside
        j = i
    return inside


def net_area(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> float:
    """Footprint area with courtyards and light wells subtracted."""
    area = polygon_area(outer)
    for hole in holes:
        area -= polygon_area(hole)
    return area


ProfileIssueCode = Literal[
    "too-few-points",
    "zero-area",
    "self-intersecting",
    "hole-outside-outer",
    "hole-self-intersecting",
]


@dataclass(frozen=True)
class ProfileValidationIssue:
    code: ProfileIssueCode
    message: str
    hole_index: int | None = None


def validate_profile(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> list[ProfileValidationIssue]:
    """Check a sketched outline before it becomes geometry.

    Catching these at sketch time is the difference between a clear "this outline crosses itself"
    and a mass that silently computes a nonsensical area -- the shoelace formula happily returns a
    number for a bow-tie, and that number is meaningless.
    """
    issues: list[ProfileValidationIssue] = []

    if len(outer) < 3:
        issues.append(
            ProfileValidationIssue("too-few-points", "A profile needs at least three points.")
        )
        return issues  # everything below assumes a ring

    if polygon_area(outer) < EPSILON:
        issues.append(ProfileValidationIssue("zero-area", "The profile encloses no area."))
    if not is_simple_polygon(outer):
        issues.append(
            ProfileValidationIssue("self-intersecting", "The profile outline crosses itself.")
        )

    for hole_index, hole in enumerate(holes):
        if len(hole) < 3:
            continue
        if not is_simple_polygon(hole):
            issues.append(
                ProfileValidationIssue(
                    "hole-self-intersecting",
                    f"Opening {hole_index + 1} crosses itself.",
                    hole_index,
                )
            )
        if not all(point_in_polygon(point, outer) for point in hole):
            issues.append(
                ProfileValidationIssue(
                    "hole-outside-outer",
                    f"Opening {hole_index + 1} is not fully inside the profile.",
                    hole_index,
                )
            )

    return issues


@dataclass(frozen=True)
class StoryGeometry:
    index: int
    elevation: float
    height: float
    area: float
    perimeter: float
    excluded_from_gfa: bool = False


#: How far a matrix may stray from a rotation about z before it stops being one. Tight, because
#: the point of the check is to refuse rather than approximate: a matrix that is nearly a rotation
#: is not a rotation, and quietly treating it as one moves the building.
RIGID_TOLERANCE = 1e-9


def as_planar_rigid(
    matrix: Sequence[float],
) -> tuple[float, float, float, float, float] | None:
    """Read a column-major 4x4 as a rotation about z and a translation.

    Returns ``(cos, sin, dx, dy, dz)``, or ``None`` when the matrix is not that -- a rotation about
    x or y, a scale, a shear or a projection all return ``None``.

    A mass here is a vertical extrusion of a horizontal profile, so a rotation about z and a
    translation are exactly the transforms that survive as another mass of the same kind. Tilting
    one does not produce a mass with a different profile; it produces something massing cannot
    represent at all. Approximating that -- dropping the tilt, or baking it into the footprint --
    would answer with a building the caller did not ask for, so it is refused instead.
    """
    if len(matrix) != 16:
        return None
    values = [float(v) for v in matrix]
    # Column-major: column c, row r is at 4c + r.
    if any(abs(values[i]) > RIGID_TOLERANCE for i in (2, 6, 8, 9)):
        return None  # z mixes with x or y -- a tilt.
    if abs(values[10] - 1.0) > RIGID_TOLERANCE:
        return None  # z is scaled or mirrored.
    if any(abs(values[i]) > RIGID_TOLERANCE for i in (3, 7, 11)):
        return None  # a projective row.
    if abs(values[15] - 1.0) > RIGID_TOLERANCE:
        return None

    cos, sin = values[0], values[1]
    # The second column must be the first rotated a quarter turn, and the pair must be unit
    # length. Together these rule out scale, shear and mirroring, which the first column alone
    # cannot: (2, 0) and (0, 2) is a uniform scale and passes any per-column test you write.
    if abs(values[4] + sin) > RIGID_TOLERANCE or abs(values[5] - cos) > RIGID_TOLERANCE:
        return None
    if abs(cos * cos + sin * sin - 1.0) > RIGID_TOLERANCE:
        return None
    return (cos, sin, values[12], values[13], values[14])


def invert_planar_rigid(matrix: Sequence[float]) -> tuple[float, ...]:
    """The transform that undoes ``matrix``.

    Built from the decomposition rather than by inverting the 4x4 numerically. The matrix that
    comes back is the exact inverse -- a rotation's inverse is its transpose, with no solve and no
    conditioning -- so the only error left is in applying it, and that does not accumulate: a mass
    moved and unmoved fifty times comes back within about 1e-13 m, which is nine orders below the
    tolerance anything downstream measures at.
    """
    rigid = as_planar_rigid(matrix)
    if rigid is None:
        raise ValueError("Not a rotation about z with a translation; cannot be inverted as one.")
    cos, sin, dx, dy, dz = rigid
    # R is orthonormal, so its inverse is its transpose; the translation comes back through it.
    return (
        cos,
        -sin,
        0.0,
        0.0,
        sin,
        cos,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -(cos * dx + sin * dy),
        -(-sin * dx + cos * dy),
        -dz,
        1.0,
    )


def apply_planar_rigid(
    points: Sequence[Sequence[float]], cos: float, sin: float, dx: float, dy: float
) -> list[tuple[float, float, float]]:
    """Rotate about the world z axis, then translate. Z is carried through untouched."""
    moved: list[tuple[float, float, float]] = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        z = float(point[2]) if len(point) > 2 else 0.0
        moved.append((cos * x - sin * y + dx, sin * x + cos * y + dy, z))
    return moved


def story_elevations(heights: Sequence[float], base_elevation: float = 0.0) -> list[float]:
    """Cumulative base elevation of each story."""
    elevations: list[float] = []
    current = base_elevation
    for height in heights:
        elevations.append(current)
        current += height
    return elevations


def resolve_story_heights(
    story_count: int,
    heights: Sequence[float] | None,
    fallback_height: float = 3.5,
) -> list[float]:
    """Expand per-story heights into a uniform list.

    Tolerates a heights list that disagrees with the story count -- which happens constantly while
    a user is editing -- by padding with the last known height rather than failing. A massing tool
    that refuses to compute mid-edit is unusable.
    """
    resolved: list[float] = []
    last = fallback_height
    for index in range(story_count):
        height = heights[index] if heights is not None and index < len(heights) else None
        if height is not None and height > 0:
            last = height
        resolved.append(last)
    return resolved


@dataclass(frozen=True)
class MassMetricsResult:
    footprint_area: float
    gross_floor_area: float
    volume: float
    envelope_area: float
    story_count: int
    height: float
    stories: tuple[StoryGeometry, ...]


def compute_mass_metrics(
    *,
    outer: Sequence[tuple[float, float]],
    story_heights: Sequence[float],
    holes: Sequence[Sequence[tuple[float, float]]] = (),
    base_elevation: float = 0.0,
    #: Story indices excluded from gross floor area, e.g. plant levels.
    excluded_stories: Iterable[int] = (),
    #: Per-story outline override, keyed by story index -- for setbacks and tapers.
    story_outlines: Mapping[int, Sequence[tuple[float, float]]] | None = None,
) -> MassMetricsResult:
    """The numbers a massing scheme is judged on.

    Computed per story rather than as ``footprint x height`` so that setbacks, tapers and excluded
    plant levels are handled by the same code path as the simple case. Treating the simple case
    specially is how tools end up reporting a GFA that quietly ignores the setback the user just
    drew.
    """
    excluded = set(excluded_stories)
    elevations = story_elevations(story_heights, base_elevation)
    footprint_area = net_area(outer, holes)
    hole_perimeter = sum(perimeter(hole) for hole in holes)
    outlines = story_outlines or {}

    stories: list[StoryGeometry] = []
    for index, height in enumerate(story_heights):
        outline = outlines.get(index)
        area = polygon_area(outline) if outline else footprint_area
        ring = perimeter(outline) if outline else perimeter(outer) + hole_perimeter
        stories.append(
            StoryGeometry(
                index=index,
                elevation=elevations[index] if index < len(elevations) else 0.0,
                height=height,
                area=area,
                perimeter=ring,
                excluded_from_gfa=index in excluded,
            )
        )

    gross_floor_area = sum(story.area for story in stories if not story.excluded_from_gfa)
    volume = sum(story.area * story.height for story in stories)
    # Facade area plus the roof and the ground slab. The roof is the topmost story's area, which is
    # not the footprint once a setback exists.
    facade_area = sum(story.perimeter * story.height for story in stories)
    roof_area = stories[-1].area if stories else footprint_area
    height = sum(story_heights)

    return MassMetricsResult(
        footprint_area=footprint_area,
        gross_floor_area=gross_floor_area,
        volume=volume,
        envelope_area=facade_area + roof_area + footprint_area,
        story_count=len(stories),
        height=height,
        stories=tuple(stories),
    )


def floor_area_ratio(gross_floor_area: float, site_area: float | None) -> float | None:
    """Gross floor area divided by site area. ``None`` when there is no site to divide by."""
    if site_area is None or site_area < EPSILON:
        return None
    return gross_floor_area / site_area
