"""Whether a gap between two rankers is real.

Report one used a Wilson interval, which is correct for a proportion: n tickets,
each right or wrong. A recall score is not that. Each query yields a fraction
between 0 and 1 and the scores vary in size as well as in mean, so the binomial
interval would be the wrong shape and usually too narrow.

Two models also see the *same* queries, so the comparison is paired. Treating it
as two independent samples throws away the pairing and makes real differences
look like noise.
"""

from __future__ import annotations

import random

import pytest

from jevdemo.rag.rank_stats import mean_interval, paired_difference


def rng():
    return random.Random(20260920)


def test_interval_brackets_the_mean():
    lo, hi = mean_interval([0.2, 0.4, 0.6, 0.8], rng())
    assert lo < 0.5 < hi


def test_a_wider_spread_gives_a_wider_interval():
    """The whole reason a proportion interval is wrong here: these two samples
    share a mean, and only one of them supports a confident claim."""
    tight = mean_interval([0.5] * 20 + [0.45, 0.55], rng())
    loose = mean_interval([0.0, 1.0] * 11, rng())
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_more_queries_narrow_the_interval():
    few = mean_interval([0.0, 0.5, 1.0] * 4, rng())
    many = mean_interval([0.0, 0.5, 1.0] * 60, rng())
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_interval_stays_inside_zero_and_one():
    lo, hi = mean_interval([1.0] * 30, rng())
    assert 0.0 <= lo <= hi <= 1.0


def test_empty_sample_has_no_interval():
    assert mean_interval([], rng()) is None


def test_a_consistent_small_win_on_every_query_is_detected():
    """Paired is the point. Unpaired, a 0.05 edge inside a 0-to-1 spread is
    invisible; paired, it is the same sign on every single query."""
    a = [0.1, 0.4, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8] * 5
    b = [x - 0.05 for x in a]
    lo, hi = paired_difference(a, b, rng())
    assert lo > 0, "a consistent per-query win should exclude zero"


def test_noise_around_zero_is_not_called_a_difference():
    a = [0.1, 0.9, 0.4, 0.6, 0.5, 0.3] * 5
    b = [0.9, 0.1, 0.6, 0.4, 0.3, 0.5] * 5
    lo, hi = paired_difference(a, b, rng())
    assert lo < 0 < hi


def test_difference_is_antisymmetric():
    a, b = [0.2, 0.8, 0.5, 0.4], [0.1, 0.6, 0.5, 0.3]
    lo1, hi1 = paired_difference(a, b, rng())
    lo2, hi2 = paired_difference(b, a, rng())
    assert lo1 == pytest.approx(-hi2, abs=1e-9)
    assert hi1 == pytest.approx(-lo2, abs=1e-9)


def test_unequal_lengths_are_refused_rather_than_silently_truncated():
    """Truncating would compare model A on query 5 with model B on query 6 and
    report the result as a paired test."""
    with pytest.raises(ValueError):
        paired_difference([0.1, 0.2], [0.1], rng())


def test_the_same_seed_gives_the_same_interval_twice():
    """A number in the report that moves when nobody changed anything is not a
    measurement. The RNG is injected so the report is reproducible."""
    a = [0.1, 0.4, 0.9, 0.3, 0.7]
    assert mean_interval(a, random.Random(7)) == mean_interval(a, random.Random(7))
