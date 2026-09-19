# jev-demo — what does a cheap classification actually cost?

A 16-arm benchmark on one narrow task: routing a support ticket to a department. Same 100
tickets, same 5 labels, one API key. **Jev 1.13** — TypeSafe's typed-decision model — against
15 chat-model configurations, measured on latency, tokens, cost, and the number that decides
it, accuracy.

Status: **specified and probed. No implementation code written yet.**

## What the probe already established

All verified live on 2026-09-20 against the key in `.env`. Full bodies in
[specs/PROBE-RESULTS.md](specs/PROBE-RESULTS.md).

- **Jev is not a chat model.** `/v1/chat/completions` rejects it. It is served at
  `POST https://openrouter.ai/api/alpha/decisions` with a `{model, state, questions}` body and
  answers in `answers.<name>.{choice, probabilities, confidence}`.
- **One Jev classification cost $0.0000166** — 395 input tokens at $0.042/M, output billed at
  zero. Cost is a pure function of input length.
- **Reasoning is the expensive thing, not the model.** `gemini-3.8-flash` spent 171 reasoning
  tokens on a 5-token answer by default, and $0.000693. Set to minimal reasoning it cost
  $0.0000518 for the same correct answer — **13×**.
- **Reasoning cannot be turned off uniformly.** `{"enabled": false}` returns HTTP 400 on
  gemini (`Reasoning is mandatory for this endpoint`) and works on `gpt-5.3-codex`. The
  benchmark negotiates per arm and reports which setting answered.
- **`max_tokens: 64` truncates reasoning models mid-answer.** The real setting is 512.

## Read in this order

| File | What it settles |
|---|---|
| [specs/00-overview.md](specs/00-overview.md) | The task, the 16 arms, hypotheses recorded before the run, scope |
| [specs/01-architecture.md](specs/01-architecture.md) | 9 modules, the transport seam, integer money, two-pass runner |
| [specs/02-dataset.md](specs/02-dataset.md) | 100 tickets, 25 boundary cases, the validation gate |
| [specs/03-arms.md](specs/03-arms.md) | Arm registry, both wire protocols, the 4-way failure taxonomy |
| [specs/04-metrics.md](specs/04-metrics.md) | What is measured, and the 4 traps in measuring it |
| [specs/05-report.md](specs/05-report.md) | stdout table, results.json, the DevX-doctrine HTML report |
| [specs/06-tdd.md](specs/06-tdd.md) | Fixture-backed TDD: every test offline, no key, no spend |
| [specs/PROBE-RESULTS.md](specs/PROBE-RESULTS.md) | The verified evidence the specs cite |
| [specs/TASKS.md](specs/TASKS.md) | **14 atomic features**, each red-test-first, 6–7 hours |

## Cost and runtime of a full run

About **$0.46** and **20 minutes**: a strictly sequential latency pass over 20 tickets, then
an 8-wide bulk pass over all 100, across 16 arms — 1,920 calls. Derived from the measured
per-call costs in PROBE-RESULTS, not estimated.

## Explicitly out of scope

Agent CLIs — Claude Code, Codex CLI, Kiro. They authenticate separately, expose no per-call
token or cost accounting, and would need subprocess driving. Their numbers would not be
comparable to the API arms, so including them would weaken the report rather than broaden it.

## Running it, once built

```bash
pytest tests/ -q          # offline, no key, no spend
python3 -m jevdemo.run    # the paid run
```
