# Contributing

```bash
pip install -e ".[all,dev]"
pytest                                   # 652
cd web && npm ci && npm run test:all     # 22 — readers, then a real browser
```

Both suites run in CI on every push. The Node ones need Node 22 and, for the render test, Chrome —
`npm ci` fetches it.

---

## The rules that are actually enforced

`tests/test_architecture.py` parses imports rather than trusting prose, and it is mutation-tested:
injecting a violation fails the suite. These are not style preferences, they are the reason the
codebase composes.

1. **No capability family imports another.** Fifteen families live in `plugins/`, and none of them
   imports a sibling. They compose through capability tokens — a string id and a Protocol. If you
   need something another family has, declare a token for the *shape* you need and let the
   composition root satisfy it.
2. **Third-party imports live in three packages.** `viewer/` may import viser and numpy;
   `geometry/` may import numpy; `adapters/` may import the extras it declares. Everything else —
   kernel, schema, SDK, storage, vcs, `web/` and all fifteen families — is standard library only.
3. **Everything persisted carries a schema id and a version**, and a document written by a newer
   build is refused rather than misread.
4. **The kernel contains mechanisms, never features.** Nothing in `massingviser.kernel` knows what
   a storey or a cost assembly is.
5. **A plugin that raises is quarantined**, and its partial registrations rolled back.

---

## Two conventions worth adopting before you write anything

**A test name states the claim it defends.** Not `test_import_schedule`, but
`test_xer_identity_is_the_activity_code_not_the_internal_id`. When it fails a year from now, the
name should tell you what was believed and is now false. Read a few in `tests/test_delivery.py`.

**The reason lives next to the code.** Comments here explain *why*, and usually name the failure the
line prevents — "P6 stores this as 0-100 and the platform stores 0..1", "an off-by-one here does not
fail, it produces a shape that loads and is subtly the wrong solid". A comment restating what the
code does is worse than none. A PR description explaining a subtle decision is a comment in the
wrong file.

---

## What to do when something cannot be done properly

Say so, in the README's **Not yet built**, with the reason. That section is load-bearing: it is why
this codebase has no half-written SHACL engine and no schedule parser that guesses at calendars.

The same applies inside the code. When a subset is genuinely the right call, make the subset
*report its own edges*: the Turtle reader refuses blank nodes rather than dropping them, the SHACL
engine sets `report.complete = False` and names the constraint it could not evaluate, and
`adapters.missing()` says which extra is absent and why. A subset that reports what it did not check
is a tool. One that stays quiet is a lie with a green tick.

---

## Changing the wire format

`geometry/payload.py` and `web/src/mvmesh.js` are two implementations of one format, and they meet
in exactly one place — the Node suite runs against fixtures the *Python* encoder wrote.

```bash
python web/test/generate_fixtures.py     # regenerate and commit
cd web && npm test
```

CI regenerates them and fails if the checked-in copies differ, so a change on either side that
breaks the contract fails there rather than in someone's browser. If you add a block to the buffer,
bump `FORMAT_VERSION` and leave the old version readable — `READABLE_VERSIONS` exists for that.

---

## Releasing

Tag-driven, and a tag is the only thing that publishes:

```bash
# bump `version` in pyproject.toml, commit, then
git tag v0.2.0 && git push origin v0.2.0
```

The release workflow refuses to publish if the tag disagrees with `pyproject.toml`, and checks the
browser client is actually inside the wheel before it goes anywhere. Publishing uses PyPI trusted
publishing, so there is no token in this repository to leak.
