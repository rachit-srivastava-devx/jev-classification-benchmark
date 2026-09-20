"""Turning per-call rows into the numbers the report prints.

The traps this guards are the ones that make a comparison quietly unfair: a
failed call scored as a zero, two models compared over different query sets, and
a query with no gold in the candidate list counted against the reranker.
"""

from __future__ import annotations

import pytest

from jevdemo.rag.aggregate import align, summarise

SEED = 20260920


def row(arm, qid, r5, *, failure=None, cost=100, gold_in_view=1, ms=500.0):
    # Mirrors run_rag._score: with no gold in the candidate list there is nothing
    # the reranker could have surfaced, so reachable@k is None, not 0.0.
    reach = r5 if gold_in_view else None
    return {
        "arm": arm, "query_id": qid, "source": "fiqa", "depth": 100,
        "elapsed_ms": ms, "input_tokens": 1000, "output_tokens": 20,
        "reasoning_tokens": 0, "reported_cost_micro": cost, "dropped_ids": 0,
        "failure": failure, "detail": "", "ranking": ["p0"],
        "gold_in_view": gold_in_view, "gold_total": 2,
        "recall@5": r5, "reachable@5": reach, "ndcg@5": r5, "mrr": r5,
        "recall@1": r5, "reachable@1": reach, "ndcg@1": r5,
        "recall@3": r5, "reachable@3": reach, "ndcg@3": r5,
        "recall@10": r5, "reachable@10": reach, "ndcg@10": r5,
    }


def test_a_failed_call_is_not_scored_as_a_zero():
    """Scoring an HTTP 500 as 0.0 blends a reliability problem into a quality
    number, and the report can no longer tell a bad ranker from a flaky one."""
    rows = [row("a", "q1", 1.0), row("a", "q2", 0.0, failure="http_error")]
    s = summarise(rows, SEED)["a"]
    assert s["scored"] == 1
    assert s["failures"] == 1
    assert s["recall@5"] == 1.0


def test_a_query_with_no_gold_in_the_candidate_list_is_excluded_from_reachable():
    """Nothing the reranker does could surface a passage that was never
    retrieved. Counting it charges the reranker for the retriever's miss."""
    rows = [row("a", "q1", 1.0, gold_in_view=1), row("a", "q2", 0.0, gold_in_view=0)]
    s = summarise(rows, SEED)["a"]
    assert s["reachable_n"] == 1
    assert s["reachable@5"] == 1.0
    # recall@5 keeps both, because that is what the whole pipeline delivers.
    assert s["recall@5"] == pytest.approx(0.5)


def test_every_headline_score_carries_an_interval():
    rows = [row("a", f"q{i}", i / 10) for i in range(10)]
    s = summarise(rows, SEED)["a"]
    lo, hi = s["recall@5_ci95"]
    assert lo < s["recall@5"] < hi


def test_cost_is_summed_in_integer_micro_dollars():
    rows = [row("a", "q1", 1.0, cost=333), row("a", "q2", 1.0, cost=334)]
    s = summarise(rows, SEED)["a"]
    assert s["cost_micro"] == 667
    assert isinstance(s["cost_micro"], int)


def test_cost_per_thousand_queries_projects_from_the_calls_that_happened():
    rows = [row("a", f"q{i}", 1.0, cost=500) for i in range(4)]
    s = summarise(rows, SEED)["a"]
    assert s["cost_per_1k_micro"] == 500_000


def test_latency_percentiles_come_from_successful_calls_only():
    """A call that 500s in 20ms is not a fast call."""
    rows = [row("a", "q1", 1.0, ms=1000.0),
            row("a", "q2", 0.0, failure="http_error", ms=5.0)]
    assert summarise(rows, SEED)["a"]["p50_ms"] == 1000.0


def test_align_compares_two_arms_only_on_queries_both_answered():
    """Model A scoring 1.0 on a query model B never answered is not evidence
    about A versus B."""
    rows = [row("a", "q1", 1.0), row("a", "q2", 0.5),
            row("b", "q1", 0.0), row("b", "q2", 0.0, failure="http_error")]
    xs, ys = align(rows, "a", "b", "recall@5")
    assert (xs, ys) == ([1.0], [0.0])


def test_align_is_order_independent_of_how_rows_arrived():
    """Rows come back from a thread pool, so their order is not the query order.
    Pairing by position rather than by query id would silently misalign them."""
    rows = [row("a", "q2", 0.5), row("a", "q1", 1.0),
            row("b", "q1", 0.25), row("b", "q2", 0.75)]
    xs, ys = align(rows, "a", "b", "recall@5")
    assert list(zip(xs, ys)) == [(1.0, 0.25), (0.5, 0.75)]


def test_an_arm_with_no_successful_call_reports_no_score_rather_than_zero():
    rows = [row("a", "q1", 0.0, failure="http_error")]
    s = summarise(rows, SEED)["a"]
    assert s["recall@5"] is None
    assert s["recall@5_ci95"] is None


def test_summary_is_reproducible_from_the_same_seed():
    rows = [row("a", f"q{i}", (i % 7) / 7) for i in range(30)]
    assert summarise(rows, SEED)["a"]["recall@5_ci95"] == summarise(rows, SEED)["a"]["recall@5_ci95"]


def test_substitute_refuses_a_token_the_report_cannot_produce():
    """The whole point of the token system: a figure typed into prose is a figure
    that can go stale. An unknown token has to stop the build, not render."""
    from scripts.build_report import substitute
    with pytest.raises(KeyError):
        substitute("cost was {{made_up_number}}", {"real": "1"}, {"BLOCK"})


def test_substitute_leaves_this_reports_own_block_tokens_for_the_caller():
    from scripts.build_report import substitute
    assert substitute("{{HEADLINE_TABLE}}", {}, {"HEADLINE_TABLE"}) == "{{HEADLINE_TABLE}}"


def test_substitute_still_defaults_to_report_ones_blocks():
    """Report one passes no block set, so its behaviour must be untouched."""
    from scripts.build_report import BLOCKS, substitute
    name = next(iter(BLOCKS))
    assert substitute("{{%s}}" % name, {}) == "{{%s}}" % name


# --- failure reporting ------------------------------------------------------
# These guard two claims the report makes about real, named products. Getting
# either wrong publishes a falsehood about a vendor, which no score can undo.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_rag_report import _failure_tokens  # noqa: E402


def _summary(arm, *, failures=0, invented=0):
    return {"arm": arm, "failures": failures, "queries_with_dropped_ids": invented}


def test_no_model_is_named_an_inventor_when_none_invented_an_id():
    """`max` over an all-zero column still returns an arm. Printing that arm's
    name beside 'invented the most' would libel a product that invented none."""
    summary = {"jev": _summary("jev"), "sonnet-5": _summary("sonnet-5")}
    out = _failure_tokens(summary, [])
    assert "no model invented one" in out["invention_finding"]
    assert "sonnet" not in out["invention_finding"].lower()
    assert "jev" not in out["invention_finding"].lower()


def test_the_worst_inventor_is_named_when_one_exists():
    summary = {"jev": _summary("jev", invented=1),
               "sonnet-5": _summary("sonnet-5", invented=7)}
    out = _failure_tokens(summary, [])
    assert "sonnet-5" in out["invention_finding"]
    assert "7 questions" in out["invention_finding"]


def test_a_402_is_reported_as_our_own_credit_limit_not_a_model_defect():
    summary = {"jev": _summary("jev", failures=3)}
    rows = [{"arm": "jev", "detail": 'HTTP 402: {"error":"exceed your credits"}'}]
    note = _failure_tokens(summary, rows)["credit_note"]
    assert "402" in note and "not the model" in note


def test_there_is_no_credit_note_when_nothing_hit_a_402():
    summary = {"jev": _summary("jev", failures=1)}
    rows = [{"arm": "jev", "detail": "HTTP 500: upstream"}]
    assert _failure_tokens(summary, rows)["credit_note"] == ""


def test_a_row_with_no_detail_field_does_not_crash_the_credit_scan():
    """Successful rows carry no detail; `None` and a missing key both occur."""
    summary = {"jev": _summary("jev")}
    rows = [{"arm": "jev"}, {"arm": "jev", "detail": None}]
    assert _failure_tokens(summary, rows)["credit_note"] == ""


# --- the baseline claim -----------------------------------------------------
# This sentence is the report's second-biggest claim. A draft of it was typed by
# hand and asserted the opposite of what the run measured, so all three branches
# are pinned here.

from scripts.build_rag_report import _lift_tokens  # noqa: E402


def _lift_rows(arm, lift):
    """Rows where `arm` beats the baseline by `lift` on every query."""
    out = []
    for i in range(40):
        out.append({"arm": arm, "query_id": f"q{i}", "recall@5": lift, "failure": None})
        out.append({"arm": "bm25-baseline", "query_id": f"q{i}", "recall@5": 0.0,
                    "failure": None})
    return out


def test_when_every_model_beats_the_baseline_the_claim_says_so():
    rows = _lift_rows("jev", 0.2) + [
        dict(r, arm="sonnet-5") for r in _lift_rows("jev", 0.2) if r["arm"] == "jev"]
    summary = {"jev": {}, "sonnet-5": {}, "bm25-baseline": {}}
    out = _lift_tokens(summary, rows)
    assert out["models_beating_bm25"] == "2"
    assert "Every model beat the free option" in out["baseline_finding"]
    assert "cost money and changed nothing" not in out["baseline_finding"]


def test_when_no_model_beats_the_baseline_the_claim_says_that_instead():
    rows = _lift_rows("jev", 0.0)
    out = _lift_tokens({"jev": {}, "bm25-baseline": {}}, rows)
    assert out["models_beating_bm25"] == "0"
    assert "No model beat the free option" in out["baseline_finding"]


def test_a_mixed_field_names_how_many_beat_it_and_how_many_did_not():
    rows = _lift_rows("jev", 0.3)
    rows += [{"arm": "sonnet-5", "query_id": f"q{i}", "recall@5": 0.0, "failure": None}
             for i in range(40)]
    out = _lift_tokens({"jev": {}, "sonnet-5": {}, "bm25-baseline": {}}, rows)
    assert out["models_beating_bm25"] == "1"
    assert "Only 1 of the 2" in out["baseline_finding"]


# --- source counting ---------------------------------------------------------
# The report once printed "4 real collections" directly above a list of two,
# because the only count it had was of topic sets and BRIGHT contributes three
# of those. Two counts now exist and both are derived, so neither can be typed
# in by hand and drift.

from scripts.build_rag_report import SOURCE_LABEL  # noqa: E402


def collections(sources):
    return len({SOURCE_LABEL[s].split()[0] for s in sources})


def test_bright_subsets_are_one_collection():
    subsets = ["bright-biology", "bright-economics", "bright-psychology"]
    assert collections(subsets) == 1
    assert len(subsets) == 3


def test_the_shipped_roster_is_two_collections_over_four_topics():
    every = sorted(SOURCE_LABEL)
    assert collections(every) == 2
    assert len(every) == 4


def test_a_single_source_counts_as_one_of_each():
    assert collections(["fiqa"]) == 1
