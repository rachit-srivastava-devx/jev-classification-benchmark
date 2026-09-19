# 00 · Overview

## Bottom line

Jev costs roughly **$0.000017 per classification** and returns a typed label that cannot be
malformed. The chat models tested cost between $0.00004 and $0.00063 for the same answer, and
one of them — `gemini-3.8-flash` — spends **13× its own minimum** on reasoning tokens for a
question it answers correctly without them (spec PROBE-RESULTS, P3). This benchmark turns
those single-call observations into 16 arms × 100 tickets of measured evidence, and reports
the one number that decides whether cheap matters: accuracy.

## The task

A free-text support ticket must be routed to exactly one department:

`billing` · `technical` · `account` · `sales` · `other`

Fixed label set, short input, no prose required in the answer. The canonical shape where a
decision model should beat a chat model, and therefore the fairest place to test the claim.

## The arms

16 arms, one provider (OpenRouter), one API key, one run. Models reasoning-capable on this
endpoint get two arms — default and lowest-reasoning — because the gap between them is a
configuration choice teams make without seeing its cost.

| # | Arm | Model | Reasoning |
|---|---|---|---|
| 1 | `jev` | `typesafe/jev-1.13` | n/a — decisions endpoint |
| 2 | `sonnet-5` | `anthropic/claude-sonnet-5` | none observed |
| 3 | `opus-5` | `anthropic/claude-opus-5` | none observed |
| 4–5 | `gpt-5.3-codex{,-low}` | `openai/gpt-5.3-codex` | default / off |
| 6–7 | `gemini-3.8-flash{,-low}` | `google/gemini-3.8-flash` | default / minimal |
| 8–9 | `glm-5.3-flash{,-low}` | `z-ai/glm-5.3-flash` | default / low |
| 10–11 | `qwen3.8-flash{,-low}` | `qwen/qwen3.8-flash` | default / low |
| 12 | `deepseek-v4.1-flash-low` | `deepseek/deepseek-v4.1-flash` | low |
| 13 | `kimi-k3-low` | `moonshotai/kimi-k3` | low |
| 14 | `grok-4.6-low` | `x-ai/grok-4.6` | low |
| 15 | `mistral-small-low` | `mistralai/mistral-small-2603` | low |
| 16 | `llama-4-scout` | `meta-llama/llama-4-scout` | not supported |

Roster verified against `GET /api/v1/models`, 2026-09-20 (spec PROBE-RESULTS, P4). The arm
table lives in `jevdemo/arms.py` and is the only place a model id appears.

## Hypotheses, recorded before the run

1. **Jev is 10–40× cheaper than the median chat arm.** From P1/P2: $0.0000166 against a
   median of roughly $0.00026.
2. **Jev is fastest at P50**, but by less than the price gap suggests — OpenRouter publishes
   0.26 s for Jev, and the network floor from this machine is 200–300 ms on every arm alike.
3. **Accuracy spreads under 8 points across the top half of the roster.** Unverified, and the
   number the whole exercise exists to produce.
4. **The `-low` arms lose under 2 points of accuracy while cutting cost 2–13×.** If this
   holds, it is a larger practical finding than the Jev comparison itself.

Hypotheses are written down now so the report can record which were wrong. A benchmark that
only confirms is not being run honestly.

## Scope

**In:** 100 hand-labelled tickets · 16 arms · a two-pass runner (clean latency, then bulk
accuracy) · offline-testable parsers · a stdout table, a results JSON, and one HTML report
built to the DevX doctrine.

**Out, deliberately:**

- **Agent CLIs** (Claude Code, Codex CLI, Kiro). They authenticate separately, expose no
  per-call token or cost accounting, and would have to be driven as subprocesses. Their
  numbers would not be comparable to the API arms, so including them would weaken the
  report rather than broaden it. Decided with the user, 2026-09-20.
- **Jev's parallel-question advantage.** Asking seven questions costs roughly what one costs
  (awesome-jev). Exercising it would make the arms structurally different tasks. One question
  each; the advantage is named in the report as unmeasured upside.
- Confidence thresholds, escalation, human review, few-shot tuning, prompt search, caching,
  batch endpoints, concurrency tuning.

## Non-goals worth naming

n=100 gives an accuracy confidence interval of about **±9 points** at 95%. It resolves a 15-point
gap and does not resolve a 3-point one. Every table that prints accuracy prints this alongside it.
