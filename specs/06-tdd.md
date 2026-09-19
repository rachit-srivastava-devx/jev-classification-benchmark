# 06 · TDD strategy

## Bottom line

Every test in this project runs with **no API key, no network, and no spend** — because every
network response it needs was captured once by `scripts/probe.py` and checked in under
`tests/fixtures/`. That is the whole trick. Without it, TDD on a paid API degrades into
mocking your own assumptions and proving them to yourself.

## The three test tiers

| Tier | Runs | Needs a key | Gate |
|---|---|---|---|
| **Unit** | every save | no | `pytest tests/` — must stay under 2 s |
| **Contract** | every save | no | replays captured fixtures through the real parsers |
| **Live smoke** | before a full run | yes | 1 call per arm, 16 calls, ~$0.005 |

The unit and contract tiers are the same `pytest` invocation and the same command in CI. The
live smoke tier is `pytest -m live`, deselected by default, and is the only thing that spends.

## Why fixtures, not mocks

A mock asserts that the code does what its author believed the API does. The gemini truncation
at `max_tokens: 64` and the 400 on `reasoning: {"enabled": false}` (PROBE-RESULTS P2, P3) were
both invisible to any mock anyone would have written — they were discovered by sending real
requests. So the rule is: **a fixture is a body the live API actually returned.** Hand-written
fixtures are permitted only for failure shapes the API is unlikely to produce on demand
(a truncated body, malformed JSON, a 500), and each one carries a comment saying it is
synthetic and why.

`scripts/probe.py --all --write-fixtures` regenerates the real ones. A test asserts every
fixture file is either listed in `PROBE-RESULTS.md` or marked synthetic.

## The red-green sequence, per task

Each task in `TASKS.md` names its failing test first. The sequence is always:

1. Write the test. **Run it. Watch it fail for the stated reason** — not an import error, not
   a typo. A test that has never failed has proven nothing.
2. Write the minimum code to pass it.
3. Run the whole suite, not just the new test.
4. Run `bash ~/.claude/skills/l8-code/scripts/selfcheck.sh` over the changed files.

Step 1's "watch it fail for the stated reason" is the step that gets skipped, and skipping it
is how a test that asserts nothing enters the suite green.

## The cases every parser test must cover

From the l8-code input checklist, applied to `jev_arm.parse` and `chat_arm.parse`:

| Case | Expected |
|---|---|
| the real captured body | correct `Prediction` |
| empty body / zero bytes | `error="parse"`, no exception escapes |
| valid JSON, no `label` key | `error="parse"` |
| `label: "Billing"` (wrong case) | `error="invalid_label"` — not silently normalised |
| `label: "refunds"` (not in set) | `error="invalid_label"` |
| body truncated mid-JSON (the real gemini failure) | `error="parse"` |
| `usage` absent | tokens 0, cost 0, and a reconciliation warning — not a crash |
| unicode and emoji in the ticket text | request builds; encoding is UTF-8 |
| a 400 body naming reasoning | triggers the one permitted retry |
| a 500 | `error="http"`, **no retry** |

`label` case-normalisation is deliberately **not** done. Silently accepting `"Billing"` would
hide a real behavioural difference between models under a convenience, and the failure
taxonomy in spec 03 exists to surface exactly that.

## What the tests must assert that is easy to forget

- **`urllib` is imported in exactly one module.** One test greps the package. This is what
  keeps the suite offline as the code grows.
- **Cost arithmetic is integer.** A test asserts no `float` reaches `cost_micro_usd`, using
  the Jev case as the known-exact value: 395 tokens → 17 micro-dollars, reported 16.59.
- **The registry matches a captured model roster.** A retired model id fails the suite, not
  minute 30 of a paid run.
- **Chat requests differ only in `model` and `reasoning`.** Built for all 15 arms, compared
  field by field.
- **Latency metrics read only `pass_name == "latency"`.** Fed a mixed list where the bulk
  samples are deliberately absurd (5 ms), the reported P50 must ignore them.
- **The dataset gate is not vacuous.** A test feeds the validator a deliberately broken
  dataset and asserts it *fails*. A validator that has only ever seen valid input has not
  been tested — and a gate that passes on an empty set is a defect, not a pass.

## Denominators

The test report states how many cases were checkable, not just how many passed. "12 parse
cases, 12 pass" is a claim; "12 of 12 parse cases, covering 6 real fixtures and 6 synthetic
failure shapes" is the same claim with its denominator attached.

## CI

```bash
pytest tests/ -q                 # unit + contract, offline, <2s
bash ~/.claude/skills/l8-code/scripts/selfcheck.sh
```

Both must be green before any paid run. The live smoke tier is run by hand, deliberately,
because it costs money.
