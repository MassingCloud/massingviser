/**
 * The three.js layer.
 *
 * Everything it does is in this file, and the list is short on purpose: fetch the manifest, ask the
 * server which buffers it does not already hold, upload those to the GPU, run the camera, and hand
 * clicks back to the server as a ray. It never parses IFC, never builds a spatial index, never
 * decimates a mesh and never decides what is visible.
 *
 * A payload is fetched **once ever**. Its id is a hash of its contents, so a buffer under a given
 * id can never change -- which is what lets the cache be a plain Map with no invalidation logic and
 * no staleness window.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { decodeMeshBatch, indexManifest, selectLod } from './mvmesh.js';

const PALETTE = [0x4c78a8, 0x54a24b, 0xf58518, 0xe45756, 0x72b7b2, 0xb279a2];

export class SceneViewer {
  constructor(canvas, { endpoint = '' } = {}) {
    this.endpoint = endpoint;
    this.payloads = new Map(); // payload id -> decoded batch, fetched at most once
    this.meshes = new Map(); // GlobalId -> THREE.Mesh
    this.currentLod = new Map(); // GlobalId -> the level currently uploaded

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1d21);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
    this.camera.position.set(90, -90, 70);
    this.camera.up.set(0, 0, 1); // Z-up, because the model is a building and not a game level

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.0));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(60, -80, 120);
    this.scene.add(sun);
    this.scene.add(new THREE.GridHelper(400, 40, 0x2f343a, 0x24282d).rotateX(Math.PI / 2));

    this.raycaster = new THREE.Raycaster();
    this.selected = null;

    window.addEventListener('resize', () => this.resize());
    this.renderer.domElement.addEventListener('pointerdown', (event) => this.onPointerDown(event));
    this.resize();
  }

  resize() {
    const { clientWidth, clientHeight } = this.renderer.domElement.parentElement;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
  }

  async api(path, body) {
    const response = await fetch(`${this.endpoint}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.json();
  }

  /** Fetch and decode a payload, or return the copy already held. */
  async payload(id) {
    if (this.payloads.has(id)) return this.payloads.get(id);
    const response = await fetch(`${this.endpoint}/api/payload/${id}.bin`);
    if (!response.ok) throw new Error(`payload ${id}: ${response.status}`);
    const decoded = decodeMeshBatch(await response.arrayBuffer());
    this.payloads.set(id, decoded);
    return decoded;
  }

  async load() {
    this.manifest = await this.api('/api/manifest');
    const { byGlobalId } = indexManifest(this.manifest);
    this.nodes = byGlobalId;

    // Ask the server what we are missing rather than assuming. On a reload with a warm cache this
    // comes back empty and nothing is transferred.
    const plan = await this.api('/api/plan', { have: [...this.payloads.keys()] });
    this.status = {
      nodes: this.manifest.nodes.length,
      drawable: byGlobalId.size,
      fetched: plan.fetch.length,
      bytes: plan.fetchBytes,
    };

    for (const [globalId, node] of byGlobalId) {
      await this.upload(globalId, node.geometry[0]);
    }
    this.frameAll();
    return this.status;
  }

  /** Put one element's chosen level on the GPU, replacing whatever level it had. */
  async upload(globalId, level) {
    if (this.currentLod.get(globalId) === level.lod) return;
    const batch = await this.payload(level.payloadId);
    const source = batch.meshes[level.geometryIndex];
    if (!source) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(source.positions, 3));
    if (source.normals) {
      geometry.setAttribute('normal', new THREE.BufferAttribute(source.normals, 3));
    } else {
      // Only when the server chose not to send them. Doing it here costs a pass over the mesh and
      // gives smooth shading everywhere, which is wrong on a box -- hence the server default.
      geometry.computeVertexNormals();
    }
    geometry.setIndex(new THREE.BufferAttribute(source.indices, 1));

    const existing = this.meshes.get(globalId);
    if (existing) {
      existing.geometry.dispose();
      existing.geometry = geometry;
    } else {
      const hash = [...globalId].reduce((total, character) => total + character.charCodeAt(0), 0);
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshLambertMaterial({ color: PALETTE[hash % PALETTE.length] }),
      );
      mesh.userData.globalId = globalId;
      this.meshes.set(globalId, mesh);
      this.scene.add(mesh);
    }
    this.currentLod.set(globalId, level.lod);
  }

  /** Re-pick a level per element as the camera moves. */
  async refreshLod() {
    const camera = this.camera.position;
    const work = [];
    for (const [globalId, node] of this.nodes) {
      const mesh = this.meshes.get(globalId);
      if (!mesh) continue;
      if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
      const centre = mesh.geometry.boundingSphere.center;
      const level = selectLod(node.geometry, camera.distanceTo(centre));
      if (level && this.currentLod.get(globalId) !== level.lod) {
        work.push(this.upload(globalId, level));
      }
    }
    await Promise.all(work);
  }

  /**
   * Clicking asks the *server* what is under the ray.
   *
   * three.js could intersect the meshes it holds, but only the ones it holds -- anything culled or
   * not yet uploaded would be unclickable. The server's BVH covers the whole model whether or not
   * it has been drawn, and returns the GlobalId that cost, markup and clash already key on.
   */
  async onPointerDown(event) {
    const bounds = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const { origin, direction } = this.raycaster.ray;

    const { hits } = await this.api('/api/pick', {
      origin: [origin.x, origin.y, origin.z],
      direction: [direction.x, direction.y, direction.z],
    });
    this.select(hits.length ? hits[0].globalId : null);
    if (this.onSelect) this.onSelect(hits.length ? hits[0] : null);
  }

  select(globalId) {
    if (this.selected) {
      const previous = this.meshes.get(this.selected);
      if (previous) previous.material.emissive.setHex(0x000000);
    }
    this.selected = globalId;
    const mesh = globalId && this.meshes.get(globalId);
    if (mesh) mesh.material.emissive.setHex(0x3a5a8a);
  }

  frameAll() {
    const box = new THREE.Box3();
    for (const mesh of this.meshes.values()) box.expandByObject(mesh);
    if (box.isEmpty()) return;
    const centre = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    this.controls.target.copy(centre);
    this.camera.position.copy(centre).add(new THREE.Vector3(radius, -radius, radius * 0.8));
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  start() {
    let sinceLodCheck = 0;
    const tick = () => {
      requestAnimationFrame(tick);
      this.controls.update();
      // Re-levelling every frame would re-upload geometry during a drag; twice a second is well
      // inside what a moving camera can notice.
      if (performance.now() - sinceLodCheck > 500) {
        sinceLodCheck = performance.now();
        this.refreshLod();
      }
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }
}
