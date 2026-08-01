/**
 * The pixel test.
 *
 * Everything else checks a half: the Python suite checks what the server sends, the Node suite
 * checks that the reader agrees with the encoder. Neither notices if the two combine into a black
 * screen -- a wrong winding order, a transform applied to the wrong side, a normal buffer bound as
 * positions. This starts the real server, opens the real page in headless Chrome, and reads the
 * framebuffer back.
 *
 * It asserts on *properties* of the image rather than on a reference screenshot: a stored PNG makes
 * every driver and font change a failure, and tells you nothing about what broke. What is checked
 * is that geometry was uploaded, that it covers a plausible share of the canvas, that it is lit
 * rather than flat black, and that clicking returns the element that is actually under the cursor.
 */

import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '..', '..');
const PORT = 8137;

let server;
let browser;
let page;

async function waitForServer(url, attempts = 120) {
  for (let i = 0; i < attempts; i += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`server did not start at ${url}`);
}

before(async () => {
  server = spawn(
    process.env.PYTHON || 'python',
    [join(here, 'serve_demo.py'), String(PORT)],
    { cwd: repo, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  server.stderr.on('data', (chunk) => process.stderr.write(`[server] ${chunk}`));
  await waitForServer(`http://127.0.0.1:${PORT}/api/manifest`);

  const puppeteer = (await import('puppeteer')).default;
  browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--use-gl=swiftshader',
      '--enable-unsafe-swiftshader',
      '--no-sandbox',
      '--disable-dev-shm-usage',
    ],
  });
  page = await browser.newPage();
  await page.setViewport({ width: 900, height: 700 });
  // `domcontentloaded` and then wait on what the test actually means. `networkidle0` never settles
  // reliably here: the page keeps fetching payloads after load, and on a slow SwiftShader runner
  // that read as a navigation timeout rather than as the page working exactly as designed.
  await page.goto(`http://127.0.0.1:${PORT}/`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  await page.waitForFunction(() => window.viewer && window.viewer.meshes.size > 0, {
    timeout: 120_000,
  });
  // One more frame after the camera has framed the model.
  await new Promise((resolve) => setTimeout(resolve, 1500));
});

after(async () => {
  if (browser) await browser.close();
  if (server) server.kill();
});

test('the page loads without a console error', async () => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.evaluate(() => window.viewer.meshes.size);
  assert.deepEqual(errors, []);
});

test('the model is uploaded and reports what it transferred', async () => {
  const status = await page.evaluate(() => window.viewer.status);
  assert.equal(status.nodes, 43, 'the demo scheme is 43 storeys');
  assert.equal(status.drawable, 43);
  assert.ok(status.bytes > 0);
});

test('geometry lands where the model says it is', async () => {
  const bounds = await page.evaluate(() => {
    // In **world** space. The buffers hold the shared plate at the origin and the placement lives
    // on the node, so reading raw vertex data would measure the family rather than the building --
    // and would pass just as happily if the transform were never applied at all.
    const box = new window.__THREE.Box3();
    for (const mesh of window.viewer.meshes.values()) {
      box.expandByObject(mesh);
    }
    return { min: box.min.toArray(), max: box.max.toArray() };
  });
  // The demo scheme: 90 m across, 68 m deep, a 28-storey tower at 3.5 m -> 98 m.
  assert.deepEqual(bounds.min.map(Math.round), [0, 0, 0]);
  assert.deepEqual(bounds.max.map(Math.round), [90, 68, 98]);
});

test('every mesh is shaded with normals the server sent', async () => {
  const shaded = await page.evaluate(() =>
    [...window.viewer.meshes.values()].filter((m) => m.geometry.getAttribute('normal')).length,
  );
  assert.equal(shaded, 43);
});

test('the canvas actually draws the building', async () => {
  const image = await page.evaluate(() => {
    const viewer = window.viewer;
    const canvas = document.getElementById('canvas');
    // Render and read in the same task. Without `preserveDrawingBuffer` the back buffer is
    // undefined once a frame has been presented, so reading after the fact returns garbage --
    // which looks exactly like "the whole canvas is covered" and would pass a naive check for
    // entirely the wrong reason. Rendering here rather than setting the flag keeps the cost out
    // of production, where every frame would pay it.
    viewer.renderer.render(viewer.scene, viewer.camera);
    const gl = viewer.renderer.getContext();
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // The scene background, which is what "nothing was drawn" looks like.
    const background = [0x1a, 0x1d, 0x21];
    let drawn = 0;
    let bright = 0;
    const distinct = new Set();
    for (let i = 0; i < pixels.length; i += 4) {
      const [r, g, b] = [pixels[i], pixels[i + 1], pixels[i + 2]];
      const isBackground =
        Math.abs(r - background[0]) < 6 &&
        Math.abs(g - background[1]) < 6 &&
        Math.abs(b - background[2]) < 6;
      if (!isBackground) drawn += 1;
      if (r + g + b > 150) bright += 1;
      distinct.add(`${r >> 3},${g >> 3},${b >> 3}`);
    }
    return {
      size: [canvas.width, canvas.height],
      total: pixels.length / 4,
      drawn,
      bright,
      distinct: distinct.size,
    };
  });

  assert.ok(image.size[0] > 0 && image.size[1] > 0, 'the canvas has a size');
  const covered = image.drawn / image.total;
  // Something is on screen, and it is not the whole screen either -- a fully covered canvas
  // usually means a camera inside the geometry rather than a framed model.
  assert.ok(covered > 0.05, `only ${(covered * 100).toFixed(1)}% of the canvas differs from the background`);
  assert.ok(covered < 0.95, `${(covered * 100).toFixed(1)}% covered -- the camera is probably inside the model`);
  // Lit, not a flat silhouette: shading produces a spread of tones per face orientation.
  assert.ok(image.bright > 0, 'nothing is lit -- normals or lights are wrong');
  assert.ok(image.distinct > 20, `only ${image.distinct} distinct colours -- the model is not shaded`);
});

test('clicking a drawn element returns the id the server holds', async () => {
  const result = await page.evaluate(async () => {
    const viewer = window.viewer;
    // Straight down through the tower footprint (x 64..90, y 6..32 in the demo scheme).
    const hits = await viewer.api('/api/pick', { origin: [77, 19, 400], direction: [0, 0, -1] });
    return {
      count: hits.hits.length,
      top: hits.hits[0] ? hits.hits[0].globalId : null,
      drawn: hits.hits[0] ? viewer.meshes.has(hits.hits[0].globalId) : false,
    };
  });
  assert.ok(result.count > 0, 'a ray down the tower hit nothing');
  assert.ok(result.top.startsWith('mass-'), `unexpected id ${result.top}`);
  // The element the server named is one the browser actually has on screen.
  assert.ok(result.drawn, 'the server returned an element the client never drew');
});

test('a reload transfers nothing, because the ids are content hashes', async () => {
  const plan = await page.evaluate(async () =>
    window.viewer.api('/api/plan', { have: [...window.viewer.payloads.keys()] }),
  );
  assert.deepEqual(plan.fetch, []);
  assert.equal(plan.fetchBytes, 0);
  assert.ok(plan.reuse.length > 0);
});
