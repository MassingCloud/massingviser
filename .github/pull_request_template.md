## What this changes

<!-- The behaviour, not the diff. What can the platform do now that it could not before? -->

## Why this way

<!--
The convention here is that the reason lives next to the code. If a decision is not obvious from
reading the change, it wants a comment in the source rather than only here — comments survive, PR
descriptions get lost.
-->

## Checklist

- [ ] `ruff check . && ruff format --check .`
- [ ] `pytest`
- [ ] `cd web && npm test` — if the payload format, the manifest or the client changed
- [ ] `python web/test/generate_fixtures.py` re-run and committed, if the encoder changed
- [ ] New behaviour has a test whose **name states the claim it defends**
- [ ] Anything deliberately not done is in the README's "Not yet built", with the reason

## Architecture

- [ ] No capability family imports another — they compose through tokens
- [ ] Third-party imports stay in `viewer/`, `geometry/` or `adapters/`
- [ ] Anything persisted carries a schema id and a version
