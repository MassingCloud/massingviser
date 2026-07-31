"""``massingviser.geometry`` -- server-side spatial compute.

The layer that exists so the browser does not have to. Everything here answers a spatial question
about a model the server already holds: what is under this ray, what is inside this frustum, which
of these two disciplines collide. The eventual JavaScript layer sends a camera and receives ids.

This is the **only** package besides ``viewer`` permitted to import numpy, and the permission is
deliberate: this is compute, vectorising it is the point, and keeping it out of the kernel, schema,
SDK and plugins is what keeps those runnable anywhere.
"""

from .bvh import LEAF_SIZE, Aabb, Bvh
from .lod import DecimatedMesh, cluster_decimate, decimate_to_budget, lod_chain
from .payload import (
    DEFAULT_CHUNK_VERTICES,
    DEFAULT_LOD_BUDGETS,
    MESH_ENCODING,
    DecodedMesh,
    EncodedPayload,
    GeometryPayloadSet,
    GeometryPlacement,
    MeshEntry,
    MeshInput,
    build_geometry_payloads,
    chunk_meshes,
    decode_mesh_batch,
    encode_mesh_batch,
)
from .scene import (
    ClashCandidate,
    Pick,
    SceneIndex,
    SpatialIndexToken,
    frustum_from_matrix,
)

__all__ = [
    "LEAF_SIZE",
    "Aabb",
    "Bvh",
    "DEFAULT_CHUNK_VERTICES",
    "DEFAULT_LOD_BUDGETS",
    "MESH_ENCODING",
    "DecimatedMesh",
    "DecodedMesh",
    "EncodedPayload",
    "GeometryPayloadSet",
    "GeometryPlacement",
    "MeshEntry",
    "MeshInput",
    "build_geometry_payloads",
    "chunk_meshes",
    "decode_mesh_batch",
    "encode_mesh_batch",
    "cluster_decimate",
    "decimate_to_budget",
    "lod_chain",
    "ClashCandidate",
    "Pick",
    "SceneIndex",
    "SpatialIndexToken",
    "frustum_from_matrix",
]
