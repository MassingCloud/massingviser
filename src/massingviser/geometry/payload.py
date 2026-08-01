"""Geometry payloads: the buffers an engine actually draws.

The scene package has always carried the semantic half -- nodes, property sets, typed edges,
precomputed indexes -- and declared payloads *by reference*. This is the other half: the bytes
those references point at.

**The format is a flat little-endian buffer, deliberately.** Not glTF, not FBX, not pickle. A
consumer needs a file handle and the ability to read a uint32; there is no schema to install, no
parser to vendor, and no vendor object model baked into the conversion path -- which is the whole
argument the scene package makes. A three.js loader for this is about forty lines, and the same
buffer is a `memcpy` into a vertex buffer in C++.

    offset  type        field
    0       char[4]     "MVMS"
    4       uint32      version (2; v1 is the same layout with no normals block)
    8       uint32      mesh_count
    12      uint32      flags            bit 0 = normals present
    16      uint32      vertex_count     total across every mesh in the chunk
    20      uint32      index_count      total
    24      uint32      lod              0 is the finest level
    28      uint32      reserved (0)
    32      directory   mesh_count x 4 x uint32:
                          vertex_offset, vertex_count, index_offset, index_count
            positions   float32[3] x vertex_count
            normals     float32[3] x vertex_count   (only when flags & 1)
            indices     uint32     x index_count

Everything after the header is tightly packed and 4-byte aligned by construction. **Indices are
local to their mesh**, so a consumer can slice one element out and upload it without rebasing.

Three decisions worth stating:

- **Normals are optional and crease-aware.** Flat shading is right for a wall and smooth shading is
  right for a scan, so ``geometry.normals`` splits vertices only where the surface actually creases
  and the buffer carries the result. A consumer that would rather derive its own asks for
  ``shade=False`` and gets positions and indices alone.
- **Chunked, not one buffer per model.** A chunk holds whole meshes up to a vertex budget, so
  editing one wall invalidates one chunk rather than the building. This mirrors what
  ``massingviser.vcs`` does with detached lists, and for the same reason.
- **Content-addressed.** A payload's id is ``sha256(buffer)[:32]``, the same convention the version
  control uses. Two models containing an identical chunk store and send it once, and a client that
  already holds an id never needs it again -- so "what changed" is a set difference over ids rather
  than a diff over geometry.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lod import decimate_to_budget
from .normals import DEFAULT_CREASE_DEGREES, compute_shading

MAGIC = b"MVMS"
FORMAT_VERSION = 2

#: Versions this build can read. v1 had no normals block; v2 adds one behind ``FLAG_NORMALS``.
#: Reading both costs one branch and means a payload written by an older build still loads.
READABLE_VERSIONS = frozenset({1, 2})

#: Header flag bit 0: a normals block sits between the positions and the indices.
FLAG_NORMALS = 1

#: What a ``PayloadRef`` declares its encoding to be. Versioned in the string so a consumer can
#: refuse a buffer it does not understand instead of reading a header it will misinterpret.
MESH_ENCODING = "massingviser-mesh/2"
HEADER_SIZE = 32
DIRECTORY_ENTRY_SIZE = 16

#: Matches ``vcs.objects.ID_LENGTH``. The two id spaces are separate but the convention is shared,
#: so anything that can hold one can hold the other.
ID_LENGTH = 32

#: Vertices per chunk. 65536 is the point below which a consumer may use 16-bit indices if it
#: wants to, and is a reasonable draw call. It is a *ceiling on whole meshes*, never a split point:
#: a mesh larger than the budget gets a chunk of its own rather than being cut in half.
DEFAULT_CHUNK_VERTICES = 65_536

#: Coarsest last. The finest level is the source mesh, so the first budget is a ceiling on it.
DEFAULT_LOD_BUDGETS: tuple[int, ...] = (20_000, 5_000, 1_000)

#: A level has to remove at least this fraction of the faces above it to be worth shipping.
#:
#: Without a floor, a budget that happens to sit just under the source produces a level that is a
#: few per cent smaller and a whole payload larger -- the client downloads a second copy of the
#: model to save nothing. 0.3 is the point where the transfer starts paying for itself.
MIN_LOD_REDUCTION = 0.3


@dataclass(frozen=True)
class MeshInput:
    """One element's triangles, in metres, in world coordinates."""

    global_id: str
    vertices: np.ndarray
    faces: np.ndarray
    #: Per-vertex, same length as ``vertices``. Supplied by ``compute_shading`` rather than built
    #: here, because getting them right needs the crease logic and a caller may already have them.
    normals: np.ndarray | None = None


@dataclass(frozen=True)
class MeshEntry:
    """Where one mesh sits inside a chunk."""

    global_id: str
    geometry_index: int
    vertex_count: int
    face_count: int


@dataclass(frozen=True)
class EncodedPayload:
    """A chunk: the bytes, their content id, and what is in them."""

    id: str
    data: bytes
    lod: int
    entries: tuple[MeshEntry, ...]

    @property
    def byte_length(self) -> int:
        return len(self.data)

    @property
    def mesh_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class GeometryPlacement:
    """Where to find one element's geometry at one level of detail."""

    payload_id: str
    geometry_index: int
    lod: int
    face_count: int


@dataclass(frozen=True)
class GeometryPayloadSet:
    payloads: tuple[EncodedPayload, ...] = ()
    #: GlobalId -> its placement at each level, finest first.
    placements: Mapping[str, tuple[GeometryPlacement, ...]] = field(default_factory=dict)

    def by_id(self, payload_id: str) -> EncodedPayload | None:
        for payload in self.payloads:
            if payload.id == payload_id:
                return payload
        return None

    @property
    def total_bytes(self) -> int:
        return sum(payload.byte_length for payload in self.payloads)


def _as_mesh(vertices: object, faces: object) -> tuple[np.ndarray, np.ndarray]:
    vertex_array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return vertex_array, face_array


def encode_mesh_batch(meshes: Sequence[MeshInput], *, lod: int = 0) -> EncodedPayload:
    """Pack meshes into one buffer and content-address it.

    Vertices are written as float32. The platform computes in float64 and stores in float64; the
    narrowing happens exactly here, at the boundary where the data stops being something to measure
    and becomes something to draw, because no GPU consumes float64 positions anyway.
    """
    if lod < 0:
        raise ValueError("LOD levels start at 0.")

    # All or nothing per chunk. The flag is in the header, not the directory, so a chunk where some
    # meshes carried normals and others did not could not describe itself.
    shaded = [mesh.normals is not None for mesh in meshes]
    if any(shaded) and not all(shaded):
        missing = [mesh.global_id for mesh in meshes if mesh.normals is None]
        raise ValueError(
            "A chunk carries normals for every mesh or for none. Missing: "
            f"{', '.join(missing[:5])}."
        )
    with_normals = bool(meshes) and all(shaded)

    entries: list[MeshEntry] = []
    position_blocks: list[np.ndarray] = []
    normal_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    directory = bytearray()
    vertex_cursor = 0
    index_cursor = 0

    for geometry_index, mesh in enumerate(meshes):
        vertex_array, face_array = _as_mesh(mesh.vertices, mesh.faces)
        vertex_count = len(vertex_array)
        index_count = face_array.size

        if with_normals:
            normal_array = np.asarray(mesh.normals, dtype=np.float64).reshape(-1, 3)
            if len(normal_array) != vertex_count:
                raise ValueError(
                    f'Mesh "{mesh.global_id}" has {len(normal_array)} normals for '
                    f"{vertex_count} vertices."
                )
            normal_blocks.append(normal_array.astype("<f4", copy=False).reshape(-1))

        if vertex_count and face_array.size:
            highest = int(face_array.max())
            if highest >= vertex_count:
                # A face indexing past the end of its own vertices reads whatever the next mesh in
                # the chunk put there, so it renders as garbage rather than failing.
                raise ValueError(
                    f'Mesh "{mesh.global_id}" has a face indexing vertex {highest} '
                    f"of {vertex_count}."
                )

        directory += struct.pack("<4I", vertex_cursor, vertex_count, index_cursor, index_count)
        position_blocks.append(vertex_array.astype("<f4", copy=False).reshape(-1))
        index_blocks.append(face_array.astype("<u4", copy=False).reshape(-1))
        entries.append(
            MeshEntry(
                global_id=mesh.global_id,
                geometry_index=geometry_index,
                vertex_count=vertex_count,
                face_count=len(face_array),
            )
        )
        vertex_cursor += vertex_count
        index_cursor += index_count

    header = struct.pack(
        "<4s7I",
        MAGIC,
        FORMAT_VERSION,
        len(meshes),
        FLAG_NORMALS if with_normals else 0,
        vertex_cursor,
        index_cursor,
        lod,
        0,
    )
    positions = np.concatenate(position_blocks) if position_blocks else np.empty(0, dtype="<f4")
    indices = np.concatenate(index_blocks) if index_blocks else np.empty(0, dtype="<u4")
    normals = np.concatenate(normal_blocks) if normal_blocks else np.empty(0, dtype="<f4")
    data = b"".join(
        (
            header,
            bytes(directory),
            positions.tobytes(),
            normals.tobytes(),
            indices.tobytes(),
        )
    )
    return EncodedPayload(
        id=hashlib.sha256(data).hexdigest()[:ID_LENGTH],
        data=data,
        lod=lod,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class DecodedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    #: ``None`` when the payload carried no normals block -- a v1 buffer, or one written with
    #: shading turned off.
    normals: np.ndarray | None = None


def decode_mesh_batch(data: bytes) -> tuple[DecodedMesh, ...]:
    """Read a payload back.

    Exists so the round trip is testable and so a Python consumer is not forced to reimplement the
    reader to check what a payload contains. An engine implements this natively in about as many
    lines as it takes here.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError("Payload is shorter than its header.")
    magic, version, mesh_count, flags, vertex_count, index_count, lod, _ = struct.unpack(
        "<4s7I", data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"Not a mesh payload (magic {magic!r}).")
    if version not in READABLE_VERSIONS:
        readable = ", ".join(f"v{v}" for v in sorted(READABLE_VERSIONS))
        raise ValueError(f"Payload format v{version}; this build reads {readable}.")
    del lod

    has_normals = bool(flags & FLAG_NORMALS)
    directory_end = HEADER_SIZE + mesh_count * DIRECTORY_ENTRY_SIZE
    positions_end = directory_end + vertex_count * 3 * 4
    normals_end = positions_end + (vertex_count * 3 * 4 if has_normals else 0)
    expected = normals_end + index_count * 4
    if len(data) != expected:
        raise ValueError(f"Payload declares {expected} bytes but carries {len(data)}.")

    positions = np.frombuffer(data, dtype="<f4", count=vertex_count * 3, offset=directory_end)
    positions = positions.reshape(-1, 3)
    normals = (
        np.frombuffer(data, dtype="<f4", count=vertex_count * 3, offset=positions_end).reshape(
            -1, 3
        )
        if has_normals
        else None
    )
    indices = np.frombuffer(data, dtype="<u4", count=index_count, offset=normals_end)

    meshes: list[DecodedMesh] = []
    for index in range(mesh_count):
        start = HEADER_SIZE + index * DIRECTORY_ENTRY_SIZE
        v_offset, v_count, i_offset, i_count = struct.unpack(
            "<4I", data[start : start + DIRECTORY_ENTRY_SIZE]
        )
        meshes.append(
            DecodedMesh(
                vertices=positions[v_offset : v_offset + v_count],
                faces=indices[i_offset : i_offset + i_count].reshape(-1, 3),
                normals=(normals[v_offset : v_offset + v_count] if normals is not None else None),
            )
        )
    return tuple(meshes)


def chunk_meshes(
    meshes: Sequence[MeshInput], *, chunk_vertices: int = DEFAULT_CHUNK_VERTICES
) -> tuple[tuple[MeshInput, ...], ...]:
    """Group whole meshes into vertex-budgeted chunks.

    Greedy over the given order, and the order is the caller's to fix -- ``build_geometry_payloads``
    sorts by GlobalId first, because a chunk boundary that moves with dictionary iteration order
    would change every content id for no change in geometry.
    """
    if chunk_vertices < 1:
        raise ValueError("A chunk has to hold at least one vertex.")
    chunks: list[tuple[MeshInput, ...]] = []
    current: list[MeshInput] = []
    used = 0
    for mesh in meshes:
        count = len(np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3))
        if current and used + count > chunk_vertices:
            chunks.append(tuple(current))
            current, used = [], 0
        current.append(mesh)
        used += count
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


#: Grid the vertices are snapped to before two shapes are compared, in metres. One micron is far
#: below any tolerance a building is modelled to and far above float32 noise, so it absorbs the
#: difference between two authorings of the same thing without being able to merge two things that
#: genuinely differ.
SHAPE_PRECISION = 1e-6


def _shape_fingerprint(vertices: np.ndarray, faces: np.ndarray) -> str:
    """A hash of a mesh's shape, independent of where it sits.

    Translated to its own minimum corner and snapped to a grid, so two identical windows authored
    separately -- each with its own local origin -- hash the same. Rotation is deliberately *not*
    normalised: a canonical orientation needs principal axes, and those are degenerate for exactly
    the shapes a building is full of. A box has no unique "first" axis, so any rule for picking one
    is arbitrary, and an arbitrary rule merges two boxes that differ.
    """
    if not len(vertices):
        return "empty"
    grid = np.round((vertices - vertices.min(axis=0)) / SHAPE_PRECISION).astype(np.int64)
    digest = hashlib.sha256()
    digest.update(grid.tobytes())
    digest.update(np.asarray(faces, dtype=np.int64).tobytes())
    return digest.hexdigest()[:ID_LENGTH]


def deduplicate_by_translation(
    meshes: Mapping[str, tuple[Any, Any]],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[str, tuple[float, float, float]]]
]:
    """Collapse meshes that are the same shape in different places.

    Returns ``({shape key: geometry at the origin}, {mesh key: (shape key, offset)})``.

    This catches what a representation id cannot: two elements authored separately that happen to
    be the same shape. It will *not* catch two that differ only by rotation, and does not pretend
    to -- see `_shape_fingerprint`.
    """
    shapes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    placements: dict[str, tuple[str, tuple[float, float, float]]] = {}

    for key, (raw_vertices, raw_faces) in sorted(meshes.items()):
        vertices, faces = _as_mesh(raw_vertices, raw_faces)
        if not len(vertices) or not len(faces):
            continue
        corner = vertices.min(axis=0)
        fingerprint = _shape_fingerprint(vertices, faces)
        if fingerprint not in shapes:
            shapes[fingerprint] = (vertices - corner, faces)
        placements[key] = (fingerprint, (float(corner[0]), float(corner[1]), float(corner[2])))
    return shapes, placements


def _decimated(mesh: MeshInput, budget: int) -> MeshInput:
    result = decimate_to_budget(mesh.vertices, mesh.faces, max_faces=budget)
    return MeshInput(mesh.global_id, result.vertices, result.faces)


def _shaded(mesh: MeshInput, crease_degrees: float) -> MeshInput:
    shaded = compute_shading(mesh.vertices, mesh.faces, crease_degrees=crease_degrees)
    return MeshInput(mesh.global_id, shaded.vertices, shaded.faces, shaded.normals)


def build_geometry_payloads(
    meshes: Mapping[str, tuple[object, object]],
    *,
    lod_budgets: Sequence[int] = DEFAULT_LOD_BUDGETS,
    chunk_vertices: int = DEFAULT_CHUNK_VERTICES,
    min_reduction: float = MIN_LOD_REDUCTION,
    shade: bool = True,
    crease_degrees: float = DEFAULT_CREASE_DEGREES,
) -> GeometryPayloadSet:
    """Turn ``{global_id: (vertices, faces)}`` into content-addressed, chunked, LOD'd payloads.

    Level 0 is the source geometry. Each further level is decimated from the **original** rather
    than from the level above, so error does not compound down the ladder.

    A level that does not cut at least ``min_reduction`` of the faces above it is dropped. Shipping
    a level that looks the same and costs a whole payload is a straight loss: the client pays the
    transfer and gets no fewer triangles for it.

    Normals are computed **per level**, after decimation. Reusing level 0's normals lower down
    would light a simplified surface as though it still had the detail it lost, which reads as
    shading that slides over the geometry as the camera pulls back. Pass ``shade=False`` for
    positions and indices only.
    """
    ordered = [
        MeshInput(global_id, *_as_mesh(vertices, faces))
        for global_id, (vertices, faces) in sorted(meshes.items())
    ]
    ordered = [mesh for mesh in ordered if len(mesh.faces)]
    if not ordered:
        return GeometryPayloadSet()

    payloads: list[EncodedPayload] = []
    placements: dict[str, list[GeometryPlacement]] = {}

    levels: list[list[MeshInput]] = [ordered]
    previous_total = sum(len(mesh.faces) for mesh in ordered)
    for budget in sorted(lod_budgets, reverse=True):
        decimated = [_decimated(mesh, budget) for mesh in ordered]
        decimated = [mesh for mesh in decimated if len(mesh.faces)]
        total = sum(len(mesh.faces) for mesh in decimated)
        if not decimated or total > previous_total * (1.0 - min_reduction):
            continue
        levels.append(decimated)
        previous_total = total

    if shade:
        levels = [[_shaded(mesh, crease_degrees) for mesh in level] for level in levels]

    for lod, level in enumerate(levels):
        for chunk in chunk_meshes(level, chunk_vertices=chunk_vertices):
            payload = encode_mesh_batch(chunk, lod=lod)
            payloads.append(payload)
            for entry in payload.entries:
                placements.setdefault(entry.global_id, []).append(
                    GeometryPlacement(
                        payload_id=payload.id,
                        geometry_index=entry.geometry_index,
                        lod=lod,
                        face_count=entry.face_count,
                    )
                )

    return GeometryPayloadSet(
        payloads=tuple(payloads),
        placements={global_id: tuple(found) for global_id, found in sorted(placements.items())},
    )
