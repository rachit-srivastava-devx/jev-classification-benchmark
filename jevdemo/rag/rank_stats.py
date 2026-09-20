"""Confidence intervals for scores that are means, not proportions.

`jevdemo.stats` computes a Wilson interval, which is the right interval for a
proportion: n tickets, each either right or wrong. A recall score is not that
shape. Each query contributes a fraction between 0 and 1 — one query might yield
0.33, the next 1.0 — so the sample has a spread of its own that a binomial
interval cannot see, and would usually understate.

So: the bootstrap. Resample the queries with replacement a few thousand times,
recompute the mean each time, and read the 2.5th and 97.5th percentiles off the
resulting distribution. It assumes nothing about the shape of the scores.

Comparisons are *paired*. Every model answers the same queries, so the query
itself is the unit resampled, and the difference is taken within each query. An
unpaired comparison would throw that away: on a corpus where query difficulty
swamps model difference — which is this corpus — it would call every real gap
noise.

The RNG is a parameter, never module state. A number in the report that changes
when nobody changed anything is not a measurement.
"""

from __future__ import annotations

import random

ITERATIONS = 4000
LOW_PCT = 0.025
HIGH_PCT = 0.975


def _percentiles(draws: list[float]) -> tuple[float, float]:
    draws.sort()
    n = len(draws)
    # Nearest-rank, clamped. With 4000 draws the choice of percentile convention
    # moves the bound by less than a thousandth; being explicit beats being clever.
    lo = draws[min(n - 1, max(0, int(LOW_PCT * n)))]
    hi = draws[min(n - 1, max(0, int(HIGH_PCT * n)))]
    return lo, hi


def mean_interval(values: list[float], rng: random.Random,
                  iterations: int = ITERATIONS) -> tuple[float, float] | None:
    """95% interval for the mean of per-query scores. None on an empty sample.

    O(iterations x len(values)). At the report's sizes — 4000 x ~100 — that is a
    few milliseconds per arm.
    """
    n = len(values)
    if not n:
        return None
    draws = [sum(rng.choices(values, k=n)) / n for _ in range(iterations)]
    return _percentiles(draws)


def paired_difference(a: list[float], b: list[float], rng: random.Random,
                      iterations: int = ITERATIONS) -> tuple[float, float] | None:
    """95% interval for mean(a) - mean(b), resampling queries not scores.

    Raises ValueError on unequal lengths rather than truncating: a truncated
    pairing compares one model's query 5 with another's query 6 and reports the
    result as if it were paired.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    n = len(a)
    if not n:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    draws = [sum(rng.choices(diffs, k=n)) / n for _ in range(iterations)]
    return _percentiles(draws)


def verdict(interval: tuple[float, float] | None) -> str:
    """'better', 'worse' or 'tie' — where tie means the sample cannot tell."""
    if interval is None:
        return "tie"
    lo, hi = interval
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "tie"
