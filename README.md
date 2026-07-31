# MassingViser

[![CI](https://github.com/MassingCloud/massingviser/actions/workflows/ci.yml/badge.svg)](https://github.com/MassingCloud/massingviser/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/MassingCloud/massingviser)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A federated **AEC platform in pure Python** — a plugin kernel, fifteen capability families,
content-addressed version control, server-side geometry, and a browser viewer.

```bash
pip install -e ".[all]"
python -m massingviser --demo
```

Opens `http://127.0.0.1:8080`: three massing blocks on a site, a panel that sketches and extrudes,
a cost panel that prices what you drew, an issues panel that pins to it, and undo across all of it.

---

## What this is

Two projects meet here, and they are exact complements.

[`MassingCloud/massingifc`](https://github.com/MassingCloud/massingifc) is 33k lines of TypeScript:
a framework-agnostic kernel — service container, event bus, command bus with undo, versioned
persistence, project containers, plugin host, capability and UI registries — plus interface-first
contracts for fifteen capability families. It states plainly that it **contains no viewer**.

[`viser`](https://github.com/viser-project/viser) is the opposite shape: a pure-Python library whose
entire point is that `pip install` gives you a browser 3D viewer driven from Python.

MassingViser is the join, plus the two things neither had: **version control** and **server-side
geometry**.

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
   kernel, schema, SDK, storage, vcs and all fifteen families — is **standard library only**.

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
    engine/        engine-neutral scene packages keyed on GlobalId
    icdd/          ISO 21597 containers, RDF/XML codec, 15 link classes, validation
  geometry/        server-side compute — BVH picking, frustum culling, clash, LOD (numpy)
  adapters/        optional: IfcOpenShell, trimesh/manifold3d, pyproj — each behind a token
  viewer/          the viser shell — the ONLY package that renders
  app.py           composition root, and the bridge that makes the families compose
```

---

## Version control

Models are versioned the way [Speckle](https://github.com/specklesystems) versions them, which is
Git's shape applied to a building rather than to text. `massingviser.vcs` has **no dependencies**.

```python
repo   = Repository(filesystem_storage("project.mass"))
first  = (await repo.save(scheme, message="Concept", author="ada")).value
await repo.create_branch("option-b")
second = (await repo.save(taller, message="30 storeys", author="ada", branch="option-b")).value

(await repo.diff(first.id, second.id)).value              # a set difference over ids
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
| ISO 21597 containers, RDF/XML | `plugins/icdd` | none |
| Picking, frustum culling, broad-phase clash, LOD | `geometry/` | numpy |
| IFC parsing, tessellation, property sets | `adapters/ifc` | ifcopenshell |
| Narrow-phase clash — real solid intersection | `adapters/solids` | trimesh, manifold3d |
| CRS transforms, georeference validation | `adapters/crs` | pyproj |

**What a JavaScript layer will eventually need to do, and nothing more:** upload buffers to the
GPU, run the camera, forward input events. It sends a ray and receives GlobalIds; it sends a
view-projection matrix and receives the ids it can see. It never parses IFC, builds a spatial
index, or decides what is visible.

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

## Writing a plugin

```python
from massingviser.kernel import CommandDefinition, create_capability_token
from massingviser.sdk import define_plugin

GreeterToken = create_capability_token("demo.greeter")

def activate(context):
    log = context.state.define_slice("log", ())
    context.capabilities.provide(GreeterToken, lambda name: f"Hello, {name}")
    context.commands.register(CommandDefinition(
        id="demo.greet",
        title="Greet",
        permission="demo.greet",
        handler=lambda params, ctx: log.update(lambda s: (*s, params["name"])),
    ))

greeter = define_plugin(id="demo.greeter", version="1.0.0",
                        permissions=["demo.greet"], activate=activate)
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
python -m pytest
```

459 tests. Organised by the claim each defends:

| File | Defends |
|---|---|
| `test_kernel.py` | Isolation, rollback, quarantine, undo semantics, pure reads, forward-incompatibility, permission fail-closed |
| `test_schema.py` | Georeferencing maths, the measurability rule, 4D intent, the record codec |
| `test_storage.py` | Key encoding (reversibility, Windows device names, traversal), atomic writes, surviving a restart |
| `test_vcs.py` | Content addressing — dedup, closure depths, chunking, diffs as set operations, three-way merge |
| `test_geometry.py` | BVH ray/frustum/pair queries, plane extraction, clash penetration |
| `test_massing.py` | Planar geometry, and that triangulation conserves area for holes and concave rings |
| `test_capabilities.py` | Money exactness, evaluator safety (five injection attempts), anchoring, issue state machine |
| `test_delivery.py` | Triage surviving a re-run, links surviving a re-issue, earned value falling on rework |
| `test_content.py` | Semver resolution, conflict-checked publish, Procrustes alignment, id-preserving replacement |
| `test_platform.py` | Interop detection, bounded forecasts, shell bookkeeping, engine scene packages |
| `test_icdd.py` | Exact IRIs, inverse pairing, RDF/XML round trip, DTD and `parseType` refusal |
| `test_adapters.py` | IFC parse/tessellate, narrow-phase volume, CRS round trip, LOD budgets |
| `test_integration.py` | The cross-plugin chain, and undo across a whole session |
| `test_architecture.py` | The five rules above, by parsing imports |

The architecture checks are mutation-tested: injecting a plugin cross-import, an `import viser`
below the shell, a kernel-reaches-up import, or `scipy` in the compute layer each fails the suite.

CI runs the suite two ways, because "the extras are optional" is a claim and not a hope:

| Job | Installs | Result |
|---|---|---|
| `core` | `.[dev]` — no extras, Python 3.10–3.13 | 441 pass, 18 skip |
| `full` | `.[all,dev]` — every extra, Linux and Windows | 459 pass |

The `core` job asserts up front that `available() == ()`. Without that, the job would go green just
as happily if an extra crept in through a transitive dependency, and the thing it exists to prove
would quietly stop being proven.

```bash
ruff check . && ruff format --check . && python -m pytest
```

---

## Not yet built

Stated plainly, in the spirit of the repository this is ported from.

- **No JavaScript layer yet.** The viewer is viser's prebuilt client. A three.js layer that consumes
  the server's culled id lists and LOD payloads is the next piece, and the server side of that
  contract is already here.
- **The 4D schedule basis and the authoring geometry backend are nominal** — a fixed S-curve and a
  port with no solid modeller behind it. Both are registered so a real implementation wins the
  moment one is installed, and both say what they are.
- **Schedule import reads CSV and JSON only.** P6 XER and MS Project XML belong behind the same
  interface as their own plugin; a half-parser that mis-reads a calendar would be worse.
- **ICDD is RDF/XML only** — no Turtle, no JSON-LD — and validation is structural rather than SHACL.
  Checksums are not verified. The parsed graphs are exposed so a host can run the published shapes.
- **The engine bridge emits the semantic half.** Nothing hands out mesh buffers yet, so scene
  packages carry nodes, properties, relationships and indexes but no geometry payloads — and
  `validate_scene_package` reports that rather than leaving a consumer to find out.
- **No IFC writing.** Reading and tessellation only.
- **Rendered output is not asserted.** The viewer tests check what the server sends, not pixels.

## Relationship to `massingifc`

Complementary, not competing. `massingifc` remains the TypeScript reference and the source of the
schema ids written here. MassingViser is what that architecture looks like in Python, with the
viewer attached, version control added, and the geometry pipeline moved server-side.

MIT.
