# MassingViser

[![CI](https://github.com/MassingCloud/massingviser/actions/workflows/ci.yml/badge.svg)](https://github.com/MassingCloud/massingviser/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/MassingCloud/massingviser)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A federated **AEC platform in Python** — a plugin kernel, fifteen capability families,
content-addressed version control, a server-side geometry pipeline, and two browser front ends.

```bash
pip install -e ".[all]"
python -m massingviser --demo --web
```

Opens `http://127.0.0.1:8080`: three massing blocks on a site, a panel that sketches and extrudes,
a cost panel that prices what you drew, an issues panel that pins to it, and undo across all of it.
`--web` adds `http://127.0.0.1:8081`, a three.js client drawing the same kernel's geometry.

---

## What this is

Two projects meet here, and they are exact complements.

[`MassingCloud/massingifc`](https://github.com/MassingCloud/massingifc) is 33k lines of TypeScript:
a framework-agnostic kernel — service container, event bus, command bus with undo, versioned
persistence, project containers, plugin host, capability and UI registries — plus interface-first
contracts for fifteen capability families. It states plainly that it **contains no viewer**.

[`viser`](https://github.com/viser-project/viser) is the opposite shape: a pure-Python library whose
entire point is that `pip install` gives you a browser 3D viewer driven from Python.

MassingViser is the join, plus the things neither had: **version control**, a **server-side
geometry pipeline**, and a browser layer thin enough to be worth reading.

Five rules, checked by `tests/test_architecture.py` rather than asserted in prose:

1. **The kernel contains mechanisms, never features.** Nothing in `massingviser.kernel` knows what
   a massing story or a cost assembly is.
2. **No plugin can crash the host.** A plugin that raises is quarantined and its partial
   registrations rolled back.
3. **Interface-first.** Every capability ships contracts before implementation.
4. **Everything persisted is versioned.** A document written by a newer build is refused rather
   than misread.
5. **Dependencies live in three named layers.** `viewer/` may import viser and numpy; `geometry/`
   may import numpy; `adapters/` may import the optional extras it declares. Everything else —
   kernel, schema, SDK, storage, vcs, the HTTP layer and all fifteen families — is **standard
   library only**.

---

## The shape

```
massingviser/
  kernel/          no dependencies — DI, events, commands+undo, state, persistence, containers,
                   plugin host, capability & UI registries, permissions, telemetry
  schema/          all 13 record families, stable schema ids, migration engine, record codec
  sdk/             define_plugin, RecordStore, clock/id ports, a real-kernel test harness
  storage/         durable adapters — filesystem (atomic writes) and SQLite
  vcs/             content-addressed version control — commits, branches, diffs, merges
  plugins/         all fifteen capability families, none importing another
    massing/       planar geometry, sketch validation, story-aware masses, metrics, tessellation
    markup/        pins, GlobalId anchoring with orphan reporting, issues, threads, review
    estimating/    takeoff with a safe expression evaluator, rates, bills, estimates, cashflow
    coordination/  clash with stable signatures, validation rules, issue routing, revision diff
    planning/      schedule import and re-import, rule-based links, playback, planned-vs-actual
    procurement/   packages from the bill, vendor award, element-level field status, earned value
    families/      repository adapters, semver resolution, placement, parameters, upgrade
    authoring/     edit sessions, sketch planes, reversible history, conflict-checked publish
    twin/          captured reality, planar Procrustes alignment, observations, gated promotion
    federation/    project composition, load state, id-preserving revision replacement
    interop/       content-first format detection, import/export dispatch, connector governance
    analytics/     metric aggregation, history, reports, forecasts with bounds
    shell/         headless shell state: layout, notifications, progress, palette, status bar
    engine/        engine-neutral scene packages: semantics, geometry ladders, transfer plans
    icdd/          ISO 21597 containers, RDF/XML codec, 15 link classes, validation
  geometry/        server-side compute — BVH picking, frustum culling, clash, LOD, crease-aware
                   normals, and the content-addressed mesh payload format (numpy)
  adapters/        optional: IfcOpenShell (read and write), trimesh/manifold3d, pyproj
  web/             four HTTP routes over the engine bridge (standard library only)
  viewer/          the viser shell
  app.py           composition root, and the bridge that makes the families compose

web/               the browser layer — a zero-dependency reader for the payload format, and a
                   three.js client built on it. Tested under Node against Python-written buffers.
```

---

## Version control

Models are versioned the way [Speckle](https://github.com/specklesystems) versions them, which is
Git's shape applied to a building rather than to text. `massingviser.vcs` has **no dependencies**.

```python
repo = Repository(filesystem_storage("project.mass"))
first = (await repo.save(scheme, message="Concept", author="ada")).value
await repo.create_branch("option-b")
second = (await repo.save(taller, message="30 storeys", author="ada", branch="option-b")).value

(await repo.diff(first.id, second.id)).value  # a set difference over ids
(await repo.merge(ours=first.id, theirs=second.id, author="ada")).value
```

A model is decomposed into atomic objects, each identified by `sha256(canonical json)[:32]`.
Members prefixed `@` are **detached** — stored as their own object and referenced by id — and long
detached lists are **chunked**, so one moved vertex rewrites one chunk rather than a buffer. Each
object carries a **closure table** mapping every descendant to its depth, turning "fetch everything
this version needs" into one lookup instead of a recursive walk over a network.

Three properties fall out, and they are the reason for the design:

- **Identical content is stored once.** Two options sharing a footprint store it once; so do two
  unrelated models that happen to contain the same object.
- **Diffing is a set operation.** An object present in both versions is byte-identical in both,
  because its id *is* its content. Comparing two 400,000-element models is set arithmetic.
- **Corruption is detectable.** An id that does not match its content is a fact you can check.

Merges are three-way against the nearest common ancestor. Disjoint edits merge; edits touching the
same path are **reported as conflicts**, never resolved by preference. Tags are immutable, because
a tag names an issued state and moving one silently rewrites what somebody was handed.

---

## Where the work happens

The goal is that the server does the heavy lifting and the browser draws.
[ThatOpen](https://github.com/ThatOpen) keeps almost everything client-side —
[`engine_web-ifc`](https://github.com/ThatOpen/engine_web-ifc) parses IFC *in the browser* via
WebAssembly. This codebase goes the other way.

| Concern | Where | Dependencies |
|---|---|---|
| Sketch validation, metrics, GFA, plot ratio | `plugins/massing` | none |
| Tessellation — ear clipping with holes, per-storey solids | `plugins/massing/tessellate` | none |
| Version control, diffing, merging | `vcs/` | none |
| Quantity takeoff, rates, bills, cashflow | `plugins/estimating` | none |
| Clash triage, validation, revision diff | `plugins/coordination` | none |
| ISO 21597 containers, RDF/XML, Turtle, JSON-LD, checksums | `plugins/icdd` | none |
| P6 XER and MS Project XML programmes | `plugins/planning/formats` | none |
| Serving the manifest, payloads, picks and culls | `web/` | none |
| Picking, frustum culling, broad-phase clash, LOD | `geometry/` | numpy |
| Mesh encoding, chunking, content addressing, normals | `geometry/payload`, `geometry/normals` | numpy |
| IFC parsing, tessellation, property sets | `adapters/ifc` | ifcopenshell |
| IFC writing — tessellated solids, spatial tree | `adapters/ifc_write` | ifcopenshell |
| Narrow-phase clash — real solid intersection | `adapters/solids` | trimesh, manifold3d |
| CRS transforms, georeference validation | `adapters/crs` | pyproj |

**What the JavaScript layer does, and nothing more:** upload buffers to the GPU, run the camera,
forward input events. It sends a ray and receives GlobalIds; it sends a view-projection matrix and
receives the ids it can see; it sends the payload ids it already holds and receives only what
changed. It never parses IFC, builds a spatial index, decimates a mesh, computes a normal, or
decides what is visible. See [The browser layer](#the-browser-layer).

### One import, six capabilities

```python
await kernel.commands.execute("interop.import", {"payload": ifc_bytes, "filename": "tower.ifc"})
```

That single command parses and tessellates the file server-side, then publishes the model to
**every** capability that asked for elements — estimating gets a takeoff source, coordination gets
snapshots and a solid-accurate clash engine, markup gets an element resolver, the engine bridge
gets scene nodes, and the geometry layer gets a spatial index. All six key on **IfcGlobalId**, so
they are talking about the same element. No plugin changed to make this work; the adapters simply
registered at a higher priority than the defaults.

---

## Geometry payloads

A scene package has two halves. The **semantic** half — nodes, property sets, typed edges,
precomputed indexes — comes from a `SceneNodeSource`. The **geometry** half comes from a separate
`GeometryPayloadSource`, and the two are joined on GlobalId by the engine service. Neither source
knows the other exists, so a deployment with no mesh producer still exports a package that selects,
filters and inspects.

```python
plan = (await engine.plan(have=client_payload_ids)).value
plan.fetch  # PayloadRefs this client does not hold
plan.fetch_bytes  # what the transfer will cost, before starting it
plan.stale  # ids it can evict

data = (await kernel.commands.execute("engine.scene.payload", {"payloadId": ref.id})).value
```

The buffer is a flat little-endian format — not glTF, not FBX. A consumer needs a file handle and
the ability to read a `uint32`; there is no schema to install and no vendor object model in the
conversion path, which is the same argument the scene package itself makes.

```
0   char[4]  "MVMS"        16  uint32  vertex_count (chunk total)
4   uint32   version (2)   20  uint32  index_count
8   uint32   mesh_count    24  uint32  lod (0 = finest)
12  uint32   flags         28  uint32  reserved
32  directory  mesh_count × { vertex_offset, vertex_count, index_offset, index_count }
    positions  float32[3] × vertex_count
    normals    float32[3] × vertex_count      ← when flags & 1
    indices    uint32     × index_count       ← local to each mesh, so one element slices out clean
```

Four properties, and they are the reason for the shape:

- **Content-addressed.** A payload's id is `sha256(buffer)[:32]`, the same convention the version
  control uses. The id depends on the geometry and nothing else — not the GlobalIds, which live in
  the manifest — so two models containing the same chunk send it once.
- **Chunked.** A chunk holds whole meshes up to a vertex budget, never splitting one. Editing a
  single wall rewrites the chunk it sits in, not the building. `tests/test_geometry.py` asserts
  this: moving one element of six across three chunks leaves two byte-identical.
- **LOD'd.** Each level is decimated from the *original*, so error does not compound, and a level
  that fails to cut 30% of the faces above it is dropped rather than shipped — a client should
  never pay a whole transfer for geometry it cannot tell apart.
- **Shaded, with a crease angle.** Flat shading is right for a wall and smooth shading is right for
  a scan, and a building contains both. A corner whose face disagrees with its vertex average by
  more than 30° keeps its own normal and gets a vertex of its own; anything flatter shares. The
  same setting shades a box flat (three faces at 90° put the average 54.7° off each) and a sphere
  smooth (adjacent faces differ by a couple of degrees). Normals are recomputed per level, because
  reusing the finest level's would light a simplified surface as though it kept its detail.

Levels ascend from the finest, and a node carries the whole ladder because choosing a level is the
client's job: it is the only party that knows the camera.

---

## The browser layer

`web/` is the JavaScript. It is small on purpose — the argument the rest of this repository makes
is that it *should* be.

```
web/src/mvmesh.js    the payload reader. Zero dependencies, no three.js, ~120 lines
web/src/viewer.js    the three.js client: fetch, upload, orbit, click
web/index.html       the page
web/vendor/          three.js, pinned and checksummed — not a CDN, because a viewer that needs
                     the public internet is no use in a site office or behind a proxy
massingviser/web/    four HTTP routes, standard library only
```

The client ships **inside the wheel**, so `pip install massingviser` and `--web` gives you a working
page rather than a working API and a 404.

Four routes are the whole server-side API:

| Route | Answers |
|---|---|
| `GET /api/manifest` | the scene: nodes, property sets, relationships, indexes, LOD ladders |
| `GET /api/payload/<id>.bin` | one geometry chunk, `immutable` for a year — the id *is* the content |
| `POST /api/plan` | given the ids I hold, what do I still need |
| `POST /api/pick` / `/api/cull` | a ray or a matrix in, GlobalIds out |

Clicking is worth spelling out. three.js could intersect the meshes it holds — but only the ones it
holds, so anything culled or not yet uploaded would be unclickable. The ray goes to the server
instead, where the BVH covers the whole model whether or not it has been drawn, and comes back as
the same GlobalId that cost, markup and coordination already key on.

Loading the demo scheme in a real browser: **43 nodes, 43 drawable, one payload, 37 KB**, and on
reload the plan comes back `fetch: 0`. That is asserted, not anecdotal — `web/test/render.test.mjs`
runs it in headless Chrome on every push and reads the framebuffer back.

### Instancing

An IFC model is tessellated in **local** coordinates with the placement read separately, which is
what makes instancing possible at all: with world coordinates baked in, a window placed a thousand
times is a thousand distinct meshes, because each one has been moved. Elements sharing a
representation collapse to one buffer and many 4×4s.

Twelve identical walls: **one mesh, 768 bytes, twelve transforms.** World vertices are reconstructed
server-side, so the spatial index, clash and bounds see exactly what they always did.

---

## Writing a plugin

```python
from massingviser.kernel import CommandDefinition, create_capability_token
from massingviser.sdk import define_plugin

GreeterToken = create_capability_token("demo.greeter")


def activate(context):
    log = context.state.define_slice("log", ())
    context.capabilities.provide(GreeterToken, lambda name: f"Hello, {name}")
    context.commands.register(
        CommandDefinition(
            id="demo.greet",
            title="Greet",
            permission="demo.greet",
            handler=lambda params, ctx: log.update(lambda s: (*s, params["name"])),
        )
    )


greeter = define_plugin(
    id="demo.greeter", version="1.0.0", permissions=["demo.greet"], activate=activate
)
```

Everything registered through `context` is released automatically on deactivate. Test it against a
real kernel, not a mock:

```python
harness = create_test_harness()
await harness.load(greeter)
await harness.execute("demo.greet", {"name": "Ada"})
```

### How the families compose

No capability family imports another. `massingviser/app.py` is where that gets demonstrated:
estimating declares `ModelElementSourceToken` — "something that can list elements" — and measures
whatever satisfies it. Markup declares `ElementResolverToken`. Coordination declares
`ClashEngineToken` and `IssueDirectoryToken`, the latter by capability *id* rather than by import,
so it is coupled to the contract and not to the package.

```
sketch a 40 × 24 footprint
  → extrude 12 storeys                 →  GFA 11,520 m²
  → takeoff measures storey by storey  →  41,472 m³, provenance: rule-frame v1 @ concept-1
  → composite rate (waste, OH, profit) →  £523.56/m³
  → bill of quantities                 →  reconciles line-by-line with its own total
  → estimate + 7.5% contingency        →  £23,341,561.34
  → cashflow over a schedule basis     →  sums to the estimate, to the penny
  → pin an issue on the mass           →  orphans when the mass is deleted, never relocates
  → undo × 4                           →  back to an empty site
```

---

## Decisions worth knowing

**Money is integer minor units.** `Money(12.5, "GBP")` raises. Rounding is half-away-from-zero, not
Python's banker's rounding — an estimator checking a line by hand expects 2.5p to become 3p.

**Takeoff expressions are parsed, never `eval`'d.** A rule's expression comes out of a cost library
or a saved project — it is *data*, and handing data to `eval` is arbitrary code execution. The
evaluator is a shunting-yard parser with no attribute access, no subscripting, no imports, and only
the functions in `SAFE_FUNCTIONS`.

**Provenance is required, not optional.** `QuantityRecord.source` records the rule, its version and
the model revision measured against. `BoqLineRecord` refuses a rate without a `rate_source`.

**Anchors orphan, they never relocate.** Markup keys on IFC GlobalId, and passing an integer is
refused at the boundary. Moving a pin somewhere plausible is how a review comment silently ends up
attached to the wrong wall.

**Clash re-runs carry triage forward.** Every clash has a signature hashed from its sorted element
pair, so a weekly cycle accumulates knowledge instead of discarding it. A clash that no longer
occurs becomes `resolved` — never deleted.

**A splat cannot be promoted to geometry.** A radiance field renders convincingly and has no
surface. Promotion is gated on measurability, and the rule lives on the schema so every consumer
agrees about it. Cataloguing one as an asset stays allowed.

**Ear clipping is implemented here, not imported.** `shapely` / `mapbox_earcut` / `triangle` are
optional extras of optional extras, and a massing tool that cannot draw a floor plate unless a C
extension happens to be installed is not a massing tool.

**Decimation is vertex clustering, not quadric.** Clustering is numpy-only, order-independent (so
the same input always gives the same output, which matters when the output is content-addressed),
and behaves well on building geometry — a wall collapsed to a slab is still a wall-shaped box.

**The kernel gets one thread.** The command bus, state store and plugin host assume no two
mutations interleave; viser's callbacks arrive on websocket threads. `KernelBridge` is the single
place that boundary is crossed, reads included.

**Schema ids keep the `massingifc.` prefix.** They name a wire format, not a package. A document
written by the TypeScript implementation opens here unchanged.

---

## Tests

```bash
python -m pytest                          # 754, the Python side
cd web && npm ci && npm run test:all      # 22, the browser side
```

**776 tests.** Organised by the claim each defends:

| File | Defends |
|---|---|
| `test_kernel.py` | Isolation, rollback, quarantine, undo semantics, pure reads, forward-incompatibility, permission fail-closed |
| `test_schema.py` | Georeferencing maths, the measurability rule, 4D intent, the record codec |
| `test_storage.py` | Key encoding (reversibility, Windows device names, traversal), atomic writes, surviving a restart |
| `test_vcs.py` | Content addressing — dedup, closure depths, chunking, diffs as set operations, and a three-way merge that will not resolve a conflict by preference |
| `test_geometry.py` | BVH ray/frustum/pair queries, plane extraction, clash penetration, crease shading against the analytic answer, the wire format byte-for-byte |
| `test_massing.py` | Planar geometry, and that triangulation conserves area for holes and concave rings |
| `test_capabilities.py` | Money exactness, evaluator safety (five injection attempts), a power with no real value reported rather than raised, anchoring, issue state machine |
| `test_delivery.py` | Triage surviving a re-run, links surviving a re-issue, earned value falling on rework, P6 and MS Project identity and units |
| `test_content.py` | Semver resolution, a publish gate that fires on somebody else's change and not your own, Procrustes alignment, constraints measured against real coordinates |
| `test_platform.py` | Interop detection, bounded forecasts, shell bookkeeping, engine scene packages, the payload transfer plan |
| `test_icdd.py` | Exact IRIs, inverse pairing, three syntaxes round-tripping the same triples, checksum verification, DTD refusal, and that SHACL reports every constraint it could not evaluate |
| `test_adapters.py` | IFC parse/tessellate, IFC **write** and read back at the same size, narrow-phase volume, CRS round trip, LOD budgets |
| `test_integration.py` | The cross-plugin chain, and undo across a whole session |
| `test_web.py` | The four HTTP routes over real sockets: content types, cache headers, malformed input, path escapes, and a cache that cannot be read half-updated |
| `web/test/mvmesh.test.mjs` | The **JavaScript** reader, against buffers the Python encoder wrote |
| `web/test/render.test.mjs` | A real server, the real page, headless Chrome, and the framebuffer read back |
| `test_properties.py` | Invariants over randomised input: key encoding, semver ordering, an evaluator that cannot be made to raise, money reversibility, mesh round-trips, merge symmetry, syntax round-trips |
| `test_architecture.py` | The five rules above, by parsing imports |

The architecture checks are mutation-tested: injecting a plugin cross-import, an `import viser`
below the shell, a kernel-reaches-up import, or `scipy` in the compute layer each fails the suite.

CI runs the suite two ways, because "the extras are optional" is a claim and not a hope:

| Job | Installs | Result |
|---|---|---|
| `core` | `.[dev]` — no extras, Python 3.10–3.13 | 658 pass, 33 skip |
| `full` | `.[all,dev]` — every extra, Linux and Windows | 691 pass |
| `web` | Node 22 + headless Chrome — fixtures, readers, then pixels | 22 pass |

The `core` job asserts up front that `available() == ()`. Without that, the job would go green just
as happily if an extra crept in through a transitive dependency, and the thing it exists to prove
would quietly stop being proven.

The `web` job **regenerates** the fixtures from the Python encoder and fails if they differ from
what is checked in, then runs the Node suite against them. That crossing is the only place the two
implementations of the wire format meet, so a change to either side that breaks the contract fails
there rather than in someone's browser. It also verifies the vendored three.js against its recorded
digests, and finishes by loading the real page in headless Chrome and reading the framebuffer —
because every other test checks one half, and neither half notices if the two combine into a black
screen.

```bash
ruff check . && ruff format --check . && python -m pytest
```

---

## Not yet built

Stated plainly, in the spirit of the repository this is ported from.

- **The authoring backend is massing-backed, not a solid modeller.** It creates, deletes, restores
  and now **moves** real records through the command bus, and measures constraints against real
  coordinates. A mass is a vertical extrusion, so a `move` carrying a rotation about z and a
  translation is carried out for real -- by rewriting the footprint it is extruded from, forking
  the profile first when another option shares it. Anything else a 4x4 can express -- a tilt, a
  scale, a shear -- has no representation as a mass and is **refused by name** rather than
  approximated. `modify` has no massing equivalent at all and is refused the same way.
- **SHACL has no SPARQL.** Core constraints, the logical combinators (`sh:node`, `sh:not`,
  `sh:and`, `sh:or`, `sh:xone`), qualified value shapes and property paths (inverse, sequence,
  alternative) are all evaluated. `sh:sparql` is not, because it means an entire query language,
  and a constraint the engine cannot evaluate is *reported* -- `report.complete` goes false and
  names it -- because an engine that skips one and answers "conforms" has reported on a shape it
  never checked.
- **The vendor programme readers do not read resources.** Tasks, dates, progress, WBS, criticality,
  the dependency graph and the working calendars are read, so lag converts through the calendar the
  file names and a cashflow curve puts no money inside a shutdown. Resource assignments and
  constraint arithmetic are left in the file rather than half-parsed.
- **IFC writing is tessellated.** Solids go out as `IfcPolygonalFaceSet`, not as swept solids or
  B-reps, because triangles are what the platform holds. Materials are written; types and
  classification are not, because nothing here holds them.
- **Instancing does not cross the massing/IFC line.** Within massing, a tower's repeated floor
  plate ships once however many storeys repeat it, and a plate shared by several masses -- or by a
  tower duplicated and moved across the site -- ships once too, because the footprint is keyed at
  its own plan corner rather than at its site coordinates. An imported IFC model collapses
  identical shapes the same way, on the same one-micron grid. What the two do not do is share with
  *each other*: a massing study and an imported model that describe the same plate send two
  meshes, because they are two models and nothing joins their geometry.
- **Rotation is not normalised when deduplicating shapes.** Two identical boxes that differ only by
  a rotation still travel twice. Collapsing them means choosing a canonical orientation, and a box's
  principal axes are degenerate -- any rule for picking them merges some pair that genuinely
  differs, which is worse than sending one mesh too many.

## Relationship to `massingifc`

Complementary, not competing. `massingifc` remains the TypeScript reference and the source of the
schema ids written here. MassingViser is what that architecture looks like in Python, with the
viewer attached, version control added, the geometry pipeline moved server-side, and a
browser layer that consumes it.

MIT.
