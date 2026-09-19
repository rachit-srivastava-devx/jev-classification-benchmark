"""The two tables a CEO reads: what is actually distinguishable, and what each
arm's accuracy must be worth before its price is justified."""

from __future__ import annotations

import pytest

from jevdemo.decision import break_even_table, distinguishability_matrix


def _arm(name, correct, attempted, cost_per_1000_micro):
    """cost_per_1000_micro is micro-dollars per 1,000 calls, which is numerically
    the per-call cost in nano-dollars. Jev at $0.000017/call is 17_000."""
    return {
        "arm": name,
        "correct": correct,
        "attempted": attempted,
        "cost_per_1000_micro": cost_per_1000_micro,
    }


ROSTER = [
    _arm("cheap", 88, 100, 17_000),    # $0.000017 per call
    _arm("mid", 90, 100, 40_000_000),
    _arm("dear", 96, 100, 630_000_000),
]


# --- distinguishability ----------------------------------------------------

def test_matrix_marks_an_unresolvable_gap_as_a_tie():
    m = distinguishability_matrix(ROSTER)
    assert m["cheap"]["mid"]["verdict"] == "tie"
    assert m["cheap"]["mid"]["resolved"] is False


def test_matrix_marks_a_resolvable_gap_with_a_direction():
    m = distinguishability_matrix(ROSTER)
    assert m["cheap"]["dear"]["verdict"] == "worse"
    assert m["dear"]["cheap"]["verdict"] == "better"
    assert m["cheap"]["dear"]["resolved"] is True


def test_matrix_diagonal_is_self():
    m = distinguishability_matrix(ROSTER)
    assert m["mid"]["mid"]["verdict"] == "self"


def test_matrix_carries_the_difference_interval():
    m = distinguishability_matrix(ROSTER)
    lo, hi = m["dear"]["cheap"]["interval"]
    assert lo > 0
    assert hi > lo


def test_matrix_on_a_single_arm_has_only_a_diagonal():
    m = distinguishability_matrix([_arm("solo", 90, 100, 10_000)])
    assert m == {"solo": {"solo": {"verdict": "self", "resolved": False,
                                   "interval": [0.0, 0.0]}}}


def test_matrix_on_an_empty_roster_is_empty():
    assert distinguishability_matrix([]) == {}


# --- break-even ------------------------------------------------------------

def test_break_even_is_measured_against_the_cheapest_arm():
    rows = break_even_table(ROSTER)
    baseline = [r for r in rows if r["is_baseline"]]
    assert len(baseline) == 1
    assert baseline[0]["arm"] == "cheap"


def test_break_even_reports_a_threshold_per_dearer_arm():
    rows = {r["arm"]: r for r in break_even_table(ROSTER)}
    # dear: (630_000_000 - 17_000) nano over (0.96 - 0.88) = 0.08
    #     -> 7_874_787_875 nano per misroute -> $7.87
    assert rows["dear"]["break_even_nano"] == 7_874_787_500
    assert rows["dear"]["break_even_usd"] == pytest.approx(7.8747875, abs=1e-7)


def test_a_dearer_arm_with_an_unresolvable_gain_is_flagged():
    """mid is nominally +2 points on cheap, but the gap is inside the interval.
    Reporting a break-even number for it would dress noise as a threshold."""
    rows = {r["arm"]: r for r in break_even_table(ROSTER)}
    assert rows["mid"]["resolved"] is False
    assert rows["mid"]["break_even_nano"] is not None
    assert "not resolvable" in rows["mid"]["note"].lower()


def test_a_dominated_arm_has_no_threshold():
    roster = [_arm("cheap", 90, 100, 17_000), _arm("waste", 80, 100, 500_000_000)]
    rows = {r["arm"]: r for r in break_even_table(roster)}
    assert rows["waste"]["break_even_nano"] is None
    assert "dominated" in rows["waste"]["note"].lower()


def test_break_even_table_is_empty_for_an_empty_roster():
    assert break_even_table([]) == []


def test_break_even_skips_arms_with_no_measured_cost():
    roster = [_arm("cheap", 90, 100, 17_000), _arm("unknown", 95, 100, None)]
    rows = {r["arm"]: r for r in break_even_table(roster)}
    assert rows["unknown"]["break_even_nano"] is None
    assert "no measured cost" in rows["unknown"]["note"].lower()
