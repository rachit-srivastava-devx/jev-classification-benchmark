"""Statistics behind the report's claims.

Every number the report calls a proof is computed here, so the proof is executable
rather than typeset. Reference values are taken from the published formulae, not
from this implementation's own output.
"""

from __future__ import annotations

import pytest

from jevdemo.stats import (
    Z_95,
    break_even_micro,
    difference_interval,
    distinguishable,
    wilson_interval,
)


# --- Wilson score interval -------------------------------------------------

def test_wilson_centre_is_pulled_toward_one_half():
    """The Wilson centre is not p-hat. That shrinkage is the whole point of using it."""
    lo, hi = wilson_interval(90, 100)
    centre = (lo + hi) / 2
    assert 0.83 < centre < 0.90
    assert centre < 0.90


def test_wilson_at_n_100_p_90_matches_the_formula_worked_longhand():
    """Reference computed here from the formula, not copied from a table.

    An earlier version of this test asserted (0.8238, 0.9420) on the strength of
    a remembered citation. The remembered numbers were wrong and the
    implementation was right. Deriving in the test removes the memory from the
    loop: if the formula below is wrong, it is wrong visibly.
    """
    import math as _m

    n, successes, z = 100, 90, 1.959963984540054
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * _m.sqrt(p * (1 - p) / n + z2 / (4 * n * n))

    lo, hi = wilson_interval(successes, n)
    assert lo == pytest.approx(centre - half)
    assert hi == pytest.approx(centre + half)
    # And the interval a reader of the report would see, to 3 places.
    assert (round(lo, 3), round(hi, 3)) == (0.826, 0.945)


def test_wilson_never_leaves_the_unit_interval_at_the_extremes():
    """The normal approximation goes negative here; Wilson is why we do not use it."""
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05

    lo, hi = wilson_interval(100, 100)
    assert 0.95 < lo < 1.0
    assert hi == 1.0


def test_wilson_width_shrinks_as_n_grows():
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(101, 100)
    with pytest.raises(ValueError):
        wilson_interval(-1, 100)


def test_wilson_on_an_empty_sample_is_the_whole_unit_interval():
    """No data means no constraint. Returning (0, 1) keeps 'unknown' honest."""
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --- Difference of two proportions -----------------------------------------

def test_difference_interval_contains_zero_for_a_small_gap():
    """93 vs 90 out of 100 is the case the report must refuse to call a win."""
    lo, hi = difference_interval(93, 100, 90, 100)
    assert lo < 0 < hi
    assert not distinguishable(93, 100, 90, 100)


def test_difference_interval_excludes_zero_for_a_large_gap():
    lo, hi = difference_interval(95, 100, 70, 100)
    assert lo > 0
    assert distinguishable(95, 100, 70, 100)


def test_difference_interval_is_antisymmetric():
    a_lo, a_hi = difference_interval(95, 100, 70, 100)
    b_lo, b_hi = difference_interval(70, 100, 95, 100)
    assert a_lo == pytest.approx(-b_hi)
    assert a_hi == pytest.approx(-b_lo)


def test_identical_arms_are_never_distinguishable():
    assert not distinguishable(88, 100, 88, 100)


# --- Break-even cost of a misroute -----------------------------------------

def test_break_even_is_the_cost_gap_divided_by_the_accuracy_gap():
    """cheap: 10 micro, 90%. dear: 260 micro, 95%.
    Gap 250 micro over 0.05 accuracy -> a misroute must cost 5000 micro to justify the dear arm."""
    assert break_even_micro(10, 0.90, 260, 0.95) == 5000


def test_break_even_is_none_when_the_dearer_arm_is_not_more_accurate():
    """A dearer arm that is no better is dominated at every misroute cost."""
    assert break_even_micro(10, 0.92, 260, 0.90) is None
    assert break_even_micro(10, 0.90, 260, 0.90) is None


def test_break_even_is_zero_when_the_better_arm_is_also_cheaper():
    """Zero means 'wins even if misroutes are free' — strict dominance, not a tie."""
    assert break_even_micro(260, 0.90, 10, 0.95) == 0


def test_break_even_rejects_accuracies_outside_the_unit_interval():
    with pytest.raises(ValueError):
        break_even_micro(10, 1.4, 260, 0.95)


def test_break_even_rejects_float_money():
    """Money is integer micro-dollars everywhere in this codebase."""
    with pytest.raises(TypeError):
        break_even_micro(10.5, 0.90, 260, 0.95)


def test_z_95_is_the_two_sided_normal_quantile():
    assert Z_95 == pytest.approx(1.959964, abs=1e-6)


# --- Rank correlation ------------------------------------------------------

def test_spearman_of_a_perfect_ranking_is_one():
    from jevdemo.stats import spearman
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_spearman_of_a_reversed_ranking_is_minus_one():
    from jevdemo.stats import spearman
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_monotone_not_linear():
    """Spearman is chosen over Pearson because only the ordering matters:
    a judge that ranks arms correctly but is miscalibrated is still useful."""
    from jevdemo.stats import spearman
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_handles_ties_with_midranks():
    from jevdemo.stats import spearman
    # xs all tied -> zero variance in ranks -> undefined correlation.
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_spearman_worked_example_with_a_tie():
    from jevdemo.stats import spearman
    # xs ranks: 1, 2.5, 2.5, 4   ys ranks: 1, 2, 3, 4
    value = spearman([10, 20, 20, 30], [1, 2, 3, 4])
    assert value == pytest.approx(0.9486832980505138, abs=1e-9)


def test_spearman_rejects_mismatched_lengths():
    from jevdemo.stats import spearman
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_spearman_needs_at_least_two_points():
    from jevdemo.stats import spearman
    assert spearman([1], [1]) is None
    assert spearman([], []) is None
