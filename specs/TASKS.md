# Tasks · 14 atomic features, TDD

Each task names its failing test first. **Watch the test fail for the stated reason before
writing the code** — that step is the one that gets skipped, and skipping it is how a test
that asserts nothing enters the suite green (spec 06).

Sizes assume one engineer who has read the specs. Total **6–7 hours**, of which about 20
minutes is paid API time and about $0.46 is spend.

---

## Phase A · Foundations — offline, no key, no spend

**A1 · Labels** — `jevdemo/labels.py`
*Red:* `test_labels.py` asserts `render_for_prompt()` and `render_for_criteria()` expose the
same five keys and the same definition strings. Fails: module does not exist.
*Green:* ordered dict of 5 labels + both renderers.
*Done:* both renderers round-trip identical definitions; no label appears twice. ~20 min.

**A2 · Transport seam** — `jevdemo/transport.py`
*Red:* `test_transport.py` asserts `FixtureTransport` replays a captured body and a scripted
`wall_ms`, and that a package-wide grep finds `import urllib` in **exactly one** module.
*Green:* the `Transport` callable, `HttpTransport`, `FixtureTransport`.
*Done:* the grep test passes and nothing else in `jevdemo/` imports `urllib`. ~30 min.

**A3 · Arm registry** — `jevdemo/arms.py`
*Red:* `test_arms.py` asserts all 16 arms resolve against a captured `models.json`, and that
prices are `int` micro-dollars.
*Green:* the frozen `Arm` dataclass and 16 rows from spec 03.
*Done:* a deliberately corrupted model id makes the test fail. ~30 min.

**A4 · Pricing** — `jevdemo/pricing.py`
*Red:* `test_pricing.py` asserts the Jev case reconciles exactly — 395 tokens at 42,000
micro/Mtok → 17 micro-dollars against a reported $1.659e-05 — and that a 3-micro-dollar
discrepancy is reported, not swallowed.
*Green:* float→int conversion at the boundary, integer arithmetic after, reconciliation.
*Done:* no `float` reaches `cost_micro_usd`; the discrepancy path is exercised. ~40 min.

---

## Phase B · Arms — offline against fixtures, then one live call

**B1 · Probe and fixtures** — `scripts/probe.py`, `tests/fixtures/`
Captures one real body per arm plus a `models.json` snapshot, writes them to
`tests/fixtures/`, and regenerates the tables in `PROBE-RESULTS.md`. Everything downstream
tests against these. *Do this first in Phase B — it de-risks B2 and B3 entirely.*
*Done:* 16 fixture files exist and each is listed in `PROBE-RESULTS.md`. ~40 min, ~$0.01.

**B2 · Jev arm** — `jevdemo/jev_arm.py`
*Red:* `test_jev_arm.py` runs the ten parser cases from spec 06 against the captured P1 body
and the synthetic failure shapes. Fails: module does not exist.
*Green:* `build_request` + `parse` per spec 03.
*Done:* an out-of-set `choice` **raises** rather than coercing to `other`; `probabilities`
survives into the `Prediction`. ~45 min.

**B3 · Chat arm** — `jevdemo/chat_arm.py`
*Red:* `test_chat_arm.py` runs the same ten cases, plus: requests built for all 15 chat arms
are byte-identical except `model` and `reasoning`.
*Green:* `build_request` + `parse`, `reasoning_tokens` defaulting to 0.
*Done:* `{"label": "Billing"}` yields `invalid_label`, **not** a normalised pass; the
truncated-gemini fixture yields `parse`. ~50 min.

**B4 · Reasoning negotiation** — in `chat_arm.py`
*Red:* a test feeds the captured gemini 400 body and asserts the arm retries **once** with
`{"effort": "minimal"}`, records which setting answered, and does **not** retry a 500.
*Green:* the one permitted retry, per arm, not per ticket.
*Done:* the resolved setting appears in the arm's recorded config. ~30 min.

---

## Phase C · Data and measurement — offline

**C1 · Dataset** — `data/tickets.json` + `tests/test_dataset.py`
*Red:* write the validator's five assertions from spec 02 **first**, plus a test that feeds it
a deliberately broken dataset and asserts it fails. Then write the 100 tickets against it.
*Done:* 100 records, 20 per label, 25 boundary cases each with a `note`, every ticket 15–60
words, no ticket containing its own label. The broken-input test fails the validator. ~90 min.

**C2 · Metrics** — `jevdemo/metrics.py`
*Red:* `test_metrics.py` feeds a hand-built `Prediction` list with known answers, including
bulk-pass samples at an absurd 5 ms, and asserts the reported P50 **ignores them**.
*Green:* pure functions — accuracy, boundary accuracy, CI, P50/P95, token means, cost/1k,
failure counts by type.
*Done:* no HTTP, no printing, no clock read inside `metrics.py`. ~50 min.

**C3 · Runner** — `jevdemo/runner.py`
*Red:* `test_runner.py` drives both passes end to end on `FixtureTransport` and asserts the
latency pass ran strictly sequentially, the first call per arm was discarded, and the bulk
pass produced 1,600 records.
*Green:* two-pass orchestration per spec 01, `--dry-run`, exit 2 on a missing key.
*Done:* a full fixture-backed run completes with zero network calls. ~50 min.

---

## Phase D · Output and the real run

**D1 · stdout + results.json** — `jevdemo/report.py`
*Red:* a test asserts none of the placeholder digits from spec 05 appear as literals in any
source file, and that the JSON round-trips to identical metrics.
*Done:* the table renders under 100 columns; the JSON recomputes every summary figure. ~40 min.

**D2 · Live smoke** — `pytest -m live`
One call per arm, 16 calls, ~$0.005. Catches a retired model or a changed envelope before the
paid run.
*Done:* 16/16 arms return a parseable answer. ~15 min.

**D3 · The run**
`python3 -m jevdemo.run`. Two passes, ~20 min, ~$0.45.
*Done:* `results.json` holds 1,920 records, every arm reconciles within 2 micro-dollars, and
any arm that did not is named. ~25 min including watching it.

**D4 · HTML report** — `report.html`
Load the `devx-doctrine` skill, inline `assets/doctrine.css`, build the five sections from
spec 05 against the real numbers. Then **open it in a browser and look at it** — a rendered
page, not an HTTP 200.
*Done:* conclusion sits above methodology; one accent used 2–4 times; tied arms banded not
ranked; the spec-00 hypotheses appear with the wrong ones marked wrong. ~60 min.

---

## Dependency order

```
A1 ─┬──────────────> B2 ─┐
A2 ─┼─> B1 ─────────> B3 ─┼─> C3 ─> D1 ─> D2 ─> D3 ─> D4
A3 ─┘           └──> B4 ─┘     │
A4 ───────────────────────────>┤
C1 ───────────────────────────>┤
C2 ───────────────────────────>┘
```

A1–A4 are parallelisable. B1 gates B2/B3/B4. C1 and C2 can proceed any time after A1.

## Gate before any paid task (D2 onward)

```bash
pytest tests/ -q
bash ~/.claude/skills/l8-code/scripts/selfcheck.sh
```

Both green, with real output pasted. A suite that passes on an empty input set is a defect,
not a pass (spec 06).

## Next action

Start A1: write `tests/test_labels.py` asserting the two renderers agree, run it, and watch
it fail on the missing module. Ten minutes, no key required.
