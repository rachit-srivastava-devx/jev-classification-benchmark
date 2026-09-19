# 04 · Metrics

## Bottom line

Five numbers per arm: accuracy, P50 latency, tokens in/out, cost per 1,000 classifications,
and failure count. Four of the five have a way of being measured correctly and reported as
something else. This file is mostly about those four.

## What is measured

| Metric | Definition | Source | From which pass |
|---|---|---|---|
| Accuracy | correct ÷ 100. A failure counts as incorrect. | gold labels | bulk |
| Boundary accuracy | correct ÷ 25, over `boundary: true` only | gold labels | bulk |
| Latency P50 / P95 | wall-clock around the HTTP call | `perf_counter` in `transport.py` | **latency only** |
| Tokens | input, output, reasoning — mean per ticket | provider `usage` | bulk |
| Cost / 1k | `total_micro ÷ n × 1000` | provider `usage.cost`, reconciled | bulk |
| Failures | by the four error types in spec 03 | — | both, reported separately |

## Trap 1 · Latency from this machine is not the model's latency

Wall-clock includes DNS, TLS, the trip to OpenRouter, its routing, and the trip back — a
100–300 ms floor that lands on every arm equally and therefore **flatters the slower arms**.
Jev's measured 1010 ms in PROBE-RESULTS P1 against its published 0.26 s P50 is almost
entirely this, plus a cold connection.

Required, all four:

1. Report **P50 and P95**, never the mean. One 11-second outlier — gemini produced exactly
   one in the probe — must not move the headline.
2. Discard each arm's first call as connection warm-up, and say it was discarded.
3. Take latency **only from the sequential pass**. The bulk pass runs 8-wide; its latency
   samples exist in the JSON and are excluded from every reported figure by construction, not
   by remembering to.
4. Print OpenRouter's published P50 beside the measured one, so the network floor is visible
   by subtraction rather than argued about.

## Trap 2 · Cost must be reconciled, not trusted

The provider reports `usage.cost`. `pricing.py` independently computes
`in × price_in + out × price_out` from the registry and asserts agreement within 2
micro-dollars. Jev is the worked proof: 395 × $0.042/M = $1.659e-05, the exact reported value
(PROBE-RESULTS P1).

A mismatch is not rounded away. It means the roster price is stale, or the provider billed
something not modelled — cache reads, a surcharge, a different served build. The run prints
the discrepancy per arm and the report carries it. **An unreconciled total is not published.**

## Trap 3 · Reasoning tokens are billed as output and hidden from the answer

`gemini-3.8-flash` default: 176 output tokens of which **171 were reasoning**, for a 5-token
answer (PROBE-RESULTS P3). Reporting only `completion_tokens` shows 176 and explains nothing;
reporting only the visible answer hides 97% of what was paid for.

So the tokens row is always **three numbers**: input, output-visible
(`completion_tokens − reasoning_tokens`), and reasoning. And each reasoning-capable model
gets its two arms printed adjacent, with the ratio between them, because the comparison
`gemini-3.8-flash` vs `gemini-3.8-flash-low` is a decision a team can act on this afternoon.

## Trap 4 · n=100 does not resolve small accuracy gaps

The 95% confidence interval on 100 binary trials is about **±9 points**. A 3-point gap
between two arms is noise. Mitigations:

- Every accuracy figure prints its CI.
- Arms within 9 points of each other are rendered as a **tied band**, not ranked 1-2-3-4.
  Ranking noise as if it were signal is the most common way a benchmark misleads.
- Boundary accuracy (n=25, CI ±19 points) is labelled as indicative only and never used to
  order arms.

## Derived comparisons

Three ratios, each printed with both underlying numbers on the same line so the ratio can
never travel alone:

- **Cost ratio** — arm ÷ Jev, cost per 1,000.
- **Latency ratio** — arm P50 ÷ Jev P50.
- **Reasoning tax** — default arm ÷ `-low` arm, cost and accuracy, for the six paired models.

## Not measured, and why

Per-class precision and recall (n=20 per class; one ticket moves a figure 5 points) ·
confidence calibration, which needs hundreds of labelled examples · throughput under
concurrency, a property of the provider's rate limits rather than the models · cached-input
pricing, since no arm is given a cacheable prefix.
