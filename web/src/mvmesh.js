/**
 * Reader for the `massingviser-mesh/2` payload format.
 *
 * Zero dependencies, and no three.js in this file on purpose -- decoding is arithmetic over a
 * DataView, and keeping it separate means the format can be tested in Node without a WebGL
 * context, and reused by a consumer that is not three.js at all.
 *
 * The format, mirrored from `massingviser/geometry/payload.py`:
 *
 *     0   char[4]  "MVMS"          16  uint32  vertexCount (chunk total)
 *     4   uint32   version (2)     20  uint32  indexCount
 *     8   uint32   meshCount       24  uint32  lod (0 = finest)
 *     12  uint32   flags           28  uint32  reserved
 *     32  directory  meshCount x { vertexOffset, vertexCount, indexOffset, indexCount }
 *         positions  float32[3] x vertexCount
 *         normals    float32[3] x vertexCount   (only when flags & 1)
 *         indices    uint32     x indexCount    (local to each mesh)
 *
 * Little-endian throughout, which is stated explicitly on every read below rather than trusted to
 * the host: a big-endian machine reading these as native would get plausible garbage.
 */

export const MAGIC = 0x534d564d; // "MVMS" read as a little-endian uint32
export const READABLE_VERSIONS = [1, 2];
export const FLAG_NORMALS = 1;
export const HEADER_SIZE = 32;
export const DIRECTORY_ENTRY_SIZE = 16;

export class MeshFormatError extends Error {}

/**
 * Decode a payload into one entry per mesh.
 *
 * Returns views onto the original buffer rather than copies. Nothing here reallocates the
 * geometry, so a 40 MB chunk costs 40 MB and not 80.
 *
 * @param {ArrayBuffer} buffer
 * @returns {{lod: number, meshes: Array<{positions: Float32Array, normals: Float32Array|null,
 *            indices: Uint32Array}>}}
 */
export function decodeMeshBatch(buffer) {
  if (buffer.byteLength < HEADER_SIZE) {
    throw new MeshFormatError('Payload is shorter than its header.');
  }
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) {
    throw new MeshFormatError('Not a mesh payload: wrong magic.');
  }
  const version = view.getUint32(4, true);
  if (!READABLE_VERSIONS.includes(version)) {
    throw new MeshFormatError(
      `Payload format v${version}; this build reads v${READABLE_VERSIONS.join(', v')}.`,
    );
  }

  const meshCount = view.getUint32(8, true);
  const flags = view.getUint32(12, true);
  const vertexCount = view.getUint32(16, true);
  const indexCount = view.getUint32(20, true);
  const lod = view.getUint32(24, true);
  const hasNormals = (flags & FLAG_NORMALS) !== 0;

  const directoryEnd = HEADER_SIZE + meshCount * DIRECTORY_ENTRY_SIZE;
  const positionsEnd = directoryEnd + vertexCount * 3 * 4;
  const normalsEnd = positionsEnd + (hasNormals ? vertexCount * 3 * 4 : 0);
  const expected = normalsEnd + indexCount * 4;
  if (buffer.byteLength !== expected) {
    throw new MeshFormatError(
      `Payload declares ${expected} bytes but carries ${buffer.byteLength}.`,
    );
  }

  const positions = new Float32Array(buffer, directoryEnd, vertexCount * 3);
  const normals = hasNormals ? new Float32Array(buffer, positionsEnd, vertexCount * 3) : null;
  const indices = new Uint32Array(buffer, normalsEnd, indexCount);

  const meshes = [];
  for (let index = 0; index < meshCount; index += 1) {
    const entry = HEADER_SIZE + index * DIRECTORY_ENTRY_SIZE;
    const vertexOffset = view.getUint32(entry, true);
    const meshVertices = view.getUint32(entry + 4, true);
    const indexOffset = view.getUint32(entry + 8, true);
    const meshIndices = view.getUint32(entry + 12, true);
    meshes.push({
      positions: positions.subarray(vertexOffset * 3, (vertexOffset + meshVertices) * 3),
      normals: normals ? normals.subarray(vertexOffset * 3, (vertexOffset + meshVertices) * 3) : null,
      indices: indices.subarray(indexOffset, indexOffset + meshIndices),
    });
  }
  return { lod, meshes };
}

/**
 * Choose a level from a node's ladder.
 *
 * The ladder is ordered finest-first, so this walks *down* to the coarsest level whose triangles
 * are still worth drawing at this distance. Screen-space error would be more principled, but it
 * needs the projection matrix and the element's radius; distance in metres is what a client always
 * has and it is monotonic in the same direction.
 *
 * @param {Array<{lod: number, faceCount: number}>} ladder finest first
 * @param {number} distance metres from the camera
 * @param {number} [metresPerLevel] distance at which each further level becomes acceptable
 */
export function selectLod(ladder, distance, metresPerLevel = 60) {
  if (!ladder || ladder.length === 0) return null;
  const step = Math.floor(distance / metresPerLevel);
  return ladder[Math.min(Math.max(step, 0), ladder.length - 1)];
}

/**
 * Which payloads a client still needs, given what it holds.
 *
 * The same set difference the server computes in `engine.scene.plan`, available locally so a client
 * can decide what to request without a round trip first. Both must agree, which is why the rule is
 * written once here and once there and tested on both sides.
 */
export function planTransfer(manifestPayloads, held) {
  const have = held instanceof Set ? held : new Set(held || []);
  const current = new Set(manifestPayloads.map((payload) => payload.id));
  return {
    fetch: manifestPayloads.filter((payload) => !have.has(payload.id)),
    reuse: [...have].filter((id) => current.has(id)).sort(),
    stale: [...have].filter((id) => !current.has(id)).sort(),
  };
}

/**
 * Build the lookup a viewer actually needs: GlobalId -> its ladder, and payload id -> what is in it.
 */
export function indexManifest(manifest) {
  const byGlobalId = new Map();
  for (const node of manifest.nodes || []) {
    if (node.geometry && node.geometry.length) {
      byGlobalId.set(node.globalId, node);
    }
  }
  const payloads = new Map((manifest.payloads || []).map((payload) => [payload.id, payload]));
  return { byGlobalId, payloads };
}
