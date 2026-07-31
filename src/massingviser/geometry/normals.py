"""Vertex normals with a crease threshold.

The reason this is not a one-liner: **flat shading is right for a wall and smooth shading is right
for a scanned surface, and a building model contains both.** Averaging every incident face normal
rounds off the corners of a box; using face normals everywhere facets a curved roof. Either choice
is wrong for half of a real model.

The fix is a crease angle. A corner whose face disagrees with its vertex's average by more than the
threshold keeps its own normal and gets a vertex of its own; everything flatter than that shares.
That is what "smoothing groups" means in every DCC tool, and it produces the right answer for both
cases from the same setting:

- **A cube.** Three faces meet at 90 degrees, so the vertex average points down the body diagonal
  and sits 54.7 degrees off each face. Every corner splits, and the cube shades flat.
- **A sphere.** Adjacent faces differ by a couple of degrees, well inside the threshold, so every
  corner keeps the average and the sphere shades smooth.

Comparing each corner against the **vertex average** rather than against its neighbours pairwise is
deliberate: it is one vectorised pass instead of a walk over the one-ring, and it cannot produce the
pairwise method's failure mode, where a chain of individually-flat transitions smooths its way
around a sharp edge.

Face normals are **area-weighted**. A sliver triangle and a large one meeting at a vertex do not
contribute equally to what that surface looks like, and weighting by area is what stops a
tessellation artefact from tilting the shading of the face beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Degrees. 30 sits above the facet angle of any reasonably tessellated curve and below the
#: shallowest angle anyone builds a corner at, which is what makes one default serve both.
DEFAULT_CREASE_DEGREES = 30.0

#: Normals are rounded to this many decimals before being compared for sharing. Without it, two
#: corners that agree to fifteen decimal places become two vertices, and a cube ends up with
#: twenty-four of them for no visual difference.
_DEDUPE_DECIMALS = 5


@dataclass(frozen=True)
class ShadedMesh:
    """Positions, faces and per-vertex normals, with vertices split where the surface creases."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unnormalised face normals, whose length is twice the triangle area.

    Left unnormalised on purpose -- the length *is* the area weight, so accumulating these directly
    weights each face's contribution by how much of the surface it actually is.
    """
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    corners = vertices[faces]
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def _normalised(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    # A degenerate triangle has no direction to offer. Zero rather than NaN: a NaN normal poisons
    # every vertex it touches and the model renders black.
    safe = np.where(lengths > 0, lengths, 1.0)
    return vectors / safe[:, None]


def compute_shading(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    crease_degrees: float = DEFAULT_CREASE_DEGREES,
) -> ShadedMesh:
    """Split vertices along creases and return per-vertex normals.

    The returned mesh has at least as many vertices as it started with -- one per distinct
    (position, normal) pair actually used. Faces are re-indexed onto it.
    """
    vertex_array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(face_array) == 0 or len(vertex_array) == 0:
        return ShadedMesh(vertex_array, face_array, np.zeros_like(vertex_array))
    if not 0.0 <= crease_degrees <= 180.0:
        raise ValueError("A crease angle is between 0 and 180 degrees.")

    weighted = face_normals(vertex_array, face_array)
    unit_faces = _normalised(weighted)

    # Area-weighted average per vertex, in one pass.
    accumulated = np.zeros_like(vertex_array)
    np.add.at(accumulated, face_array.reshape(-1), np.repeat(weighted, 3, axis=0))
    smooth = _normalised(accumulated)

    corner_vertices = face_array.reshape(-1)
    corner_faces = np.repeat(np.arange(len(face_array)), 3)
    corner_smooth = smooth[corner_vertices]
    corner_flat = unit_faces[corner_faces]

    alignment = np.einsum("ij,ij->i", corner_smooth, corner_flat)
    threshold = np.cos(np.radians(crease_degrees))
    # Below the threshold the corner is on a crease and keeps its own face's normal.
    creased = alignment < threshold
    corner_normals = np.where(creased[:, None], corner_flat, corner_smooth)

    # One output vertex per distinct (source vertex, normal). Rounding first so corners that agree
    # to floating-point noise still share.
    keys = np.concatenate(
        [
            corner_vertices[:, None].astype(np.float64),
            np.round(corner_normals, _DEDUPE_DECIMALS),
        ],
        axis=1,
    )
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)

    return ShadedMesh(
        vertices=vertex_array[corner_vertices[first]],
        faces=inverse.reshape(-1, 3),
        normals=corner_normals[first],
    )
