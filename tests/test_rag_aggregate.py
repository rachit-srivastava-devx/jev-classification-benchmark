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


def test_the_shipped_roster_is_three_collections_over_five_topics():
    # BRIGHT contributes three topic sets and one collection; FiQA and WANDS one
    # each. The two counts are deliberately different numbers and the report
    # prints both, so this test pins the pair rather than either alone.
    every = sorted(SOURCE_LABEL)
    assert collections(every) == 3
    assert len(every) == 5


def test_a_single_source_counts_as_one_of_each():
    assert collections(["fiqa"]) == 1


# --- the chart must paint itself, not rely on the stylesheet ------------------
# WeasyPrint does not cascade the document's CSS into an inline SVG, so a chart
# styled only by `class=` renders in the PDF with no axes, gridlines or leaders
# — which is exactly how this report reached GitHub. The classes stay, for the
# HTML; these tests hold the line that every element also carries its paint as a
# presentation attribute, which is the only thing the PDF reads.

import pathlib  # noqa: E402
import re  # noqa: E402

from scripts.build_rag_report import chart  # noqa: E402


def two_point_summary():
    def s(arm, cost, recall):
        return {"arm": arm, "cost_per_1k_micro": cost, "recall@5": recall,
                "scored": 100, "attempted": 100}
    return {"jev": s("jev", 500_000, 0.245),
            "glm-5.3-flash-low": s("glm-5.3-flash-low", 1_200_000, 0.244)}


def test_every_chart_line_carries_a_stroke_attribute():
    svg = chart(two_point_summary())
    lines = re.findall(r"<line [^>]*>", svg)
    assert lines, "no lines in the chart at all"
    assert all("stroke=" in ln for ln in lines), \
        [ln for ln in lines if "stroke=" not in ln]


def test_every_chart_glyph_carries_a_fill_attribute():
    svg = chart(two_point_summary())
    glyphs = re.findall(r"<(?:text|circle) [^>]*>", svg)
    assert glyphs, "no points or labels in the chart at all"
    assert all("fill=" in g for g in glyphs), [g for g in glyphs if "fill=" not in g]


def test_the_jev_point_is_painted_with_the_accent():
    svg = chart(two_point_summary())
    jev = re.search(r"<circle class='pt jev' [^>]*>", svg)
    assert jev and "#1E6FFF" in jev.group(0), jev and jev.group(0)


# --- the worked example must not be chosen after seeing the results ----------
# The appendix walks one query end to end, and an appendix chosen because it
# flattered the product is worse than no appendix. The pick is therefore made
# from queries that are hard, reachable and answered by every model, and from
# that set by a seeded shuffle — never by score.

import re  # noqa: E402

from scripts.build_rag_report import (  # noqa: E402
    EXAMPLE_CHARS, JEV_ARMS, STRATA, _encoding_tokens, _example_task,
    _gold_table, _picks_table, encoding_table, strata_table,
)


def _task(qid, stratum="hard", reachable=2, gold=("g1", "g2")):
    return {
        "query_id": qid, "source": "fiqa", "query": "q " + qid,
        "gold_ids": list(gold), "stratum": stratum, "gold_reachable": reachable,
        "candidates": [{"id": "g1", "text": "gold one " * 40},
                       {"id": "g2", "text": "gold two " * 40},
                       {"id": "c3", "text": "filler " * 40}],
    }


def _sum(*arms):
    return {a: {"arm": a, "recall@5": 0.5, "cost_micro": 1} for a in arms}


def _row(arm, qid, ranking=("g1", "c3"), failure=None, stratum="hard"):
    r = {"arm": arm, "query_id": qid, "source": "fiqa", "depth": 3,
         "elapsed_ms": 1.0, "input_tokens": 100, "output_tokens": 0,
         "reasoning_tokens": 0, "reported_cost_micro": 10, "dropped_ids": 0,
         "failure": failure, "detail": "", "ranking": list(ranking),
         "gold_in_view": 2, "gold_total": 2, "subcalls": 1,
         "subcall_failures": 0, "stratum": stratum, "mrr": 0.5}
    for k in (1, 3, 5, 10):
        r[f"recall@{k}"] = None if failure else 0.5
        r[f"reachable@{k}"] = None if failure else 0.5
        r[f"ndcg@{k}"] = None if failure else 0.5
    return r


def test_the_example_query_is_hard_reachable_and_answered_by_everyone():
    tasks = {"tasks": [_task("easy1", stratum="easy"),
                       _task("thin", reachable=1),
                       _task("good")]}
    rows = [_row("jev", "good"), _row("sonnet-5", "good"),
            _row("jev", "easy1"), _row("sonnet-5", "easy1"),
            _row("jev", "thin"), _row("sonnet-5", "thin")]
    assert _example_task(tasks, rows)["query_id"] == "good"


def test_a_query_one_model_failed_is_not_eligible():
    # sonnet-5 answered q2 and failed q1, so it has an opinion and q1 is the one
    # query it cannot speak to. The example must be q2. Both queries are equally
    # hard and equally reachable, so nothing but the failure separates them.
    tasks = {"tasks": [_task("q1"), _task("q2")]}
    rows = [_row("jev", "q1"), _row("jev", "q2"),
            _row("sonnet-5", "q1", failure="http_error"), _row("sonnet-5", "q2")]
    assert _example_task(tasks, rows)["query_id"] == "q2"


def test_no_eligible_query_is_an_error_not_a_quiet_fallback():
    # Every query failed for somebody who succeeded elsewhere. There is no
    # honest example to show, and the appendix must stop rather than quietly
    # pick one the reader would assume was clean.
    tasks = {"tasks": [_task("q1"), _task("q2")]}
    rows = [_row("jev", "q1"), _row("jev", "q2"),
            _row("sonnet-5", "q1", failure="http_error"),
            _row("sonnet-5", "q2", failure="http_error"),
            _row("sonnet-5", "q3")]
    with pytest.raises(ValueError):
        _example_task(tasks, rows)


def test_the_example_pick_is_stable_across_runs():
    tasks = {"tasks": [_task(f"q{i}") for i in range(12)]}
    rows = [_row(a, f"q{i}") for i in range(12) for a in ("jev", "sonnet-5")]
    first = _example_task(tasks, rows)["query_id"]
    assert all(_example_task(tasks, rows)["query_id"] == first for _ in range(3))


def test_the_example_pick_does_not_follow_the_best_score():
    # Same eligible set, opposite scores. If the pick tracked the numbers these
    # two calls would disagree.
    tasks = {"tasks": [_task(f"q{i}") for i in range(8)]}
    good = [_row(a, f"q{i}") for i in range(8) for a in ("jev", "sonnet-5")]
    flipped = [dict(r, **{"recall@5": 1.0 - r["recall@5"]}) for r in good]
    assert (_example_task(tasks, good)["query_id"]
            == _example_task(tasks, flipped)["query_id"])


# --- the picks table is laid out down the page, not across it ----------------
# Five chunk ids as five columns overflowed the A4 text block and clipped the
# last two silently, in a table whose entire claim is that nothing was hidden.

def test_every_pick_survives_into_one_cell_per_model():
    task = _task("good")
    rows = [_row("jev", "good", ranking=("g1", "c3", "g2"))]
    summary = _sum("jev")
    html = _picks_table(task, rows, summary, {"g1", "g2"},
                        {"g1": 1, "g2": 2, "c3": 3})
    assert html.count("<td") == 3 * html.count("<tr><td")   # 3 columns, never 7
    for cid in ("g1", "c3", "g2"):
        assert cid in html


def test_a_model_that_failed_the_example_query_says_so():
    task = _task("good")
    rows = [_row("jev", "good", failure="http_error")]
    html = _picks_table(task, rows, _sum("jev"), {"g1"}, {"g1": 1})
    assert "no answer" in html and "http_error" in html


def test_gold_outside_the_candidate_list_is_not_listed_as_missed():
    task = _task("good", gold=("g1", "never-retrieved"))
    rows = [_row("jev", "good", ranking=("g1",))]
    html = _gold_table(task, task["candidates"], rows, _sum("jev"),
                       set(task["gold_ids"]), {"g1": 1})
    assert "g1" in html and "never-retrieved" not in html


def test_a_query_with_no_reachable_gold_says_so_instead_of_an_empty_table():
    task = _task("good", gold=("nowhere",))
    html = _gold_table(task, task["candidates"], [], {}, {"nowhere"}, {})
    assert "<table" not in html and "nothing for any model to find" in html


# --- the difficulty split is a read of a stored field, not a new rule --------

def test_every_stratum_the_fetcher_writes_has_a_column():
    # Read as text rather than imported: the fetcher pulls in fsspec at module
    # scope, and a test of a naming contract must not need the download stack.
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "fetch_rag_corpus.py").read_text()
    body = src[src.index('"stratum":'):src.index('"gold_reachable":')]
    written = set(re.findall(r'"(hard|easy|unreachable)"', body))
    assert written == {k for k, _ in STRATA}


def test_strata_table_counts_questions_and_scored_answers_separately():
    rows = [_row("jev", "a", stratum="hard"),
            _row("jev", "b", stratum="hard", failure="http_error"),
            _row("jev", "c", stratum="easy")]
    html = strata_table({"rows": rows, "summary": _sum("jev")})
    assert "2 questions" in html      # both hard queries counted
    assert "1 scored" in html         # only one of them produced a number


# --- "Jev" is three encodings, and the sentence under the table says which ---

def test_the_encoding_sentence_names_both_the_best_and_the_cheapest():
    summary = {
        "jev": {"recall@5": 0.20, "cost_per_1k_micro": 1000},
        "jev-noul": {"recall@5": 0.34, "cost_per_1k_micro": 5000},
        "jev-score": {"recall@5": 0.30, "cost_per_1k_micro": 5000},
    }
    out = _encoding_tokens(summary, [])["encoding_finding"]
    assert "Noul" in out and "Choice" in out
    assert "14.0%" in out and "5×" in out
    assert "not the same one" in out


def test_one_encoding_alone_is_reported_as_nothing_to_compare():
    out = _encoding_tokens({"jev": {"recall@5": 0.2, "cost_per_1k_micro": 1}}, [])
    assert "nothing to compare" in out["encoding_finding"]


def test_the_encoding_table_prints_calls_per_question_not_total_calls():
    rows = [dict(_row("jev-noul", f"q{i}"), subcalls=100, input_tokens=1000)
            for i in range(3)]
    html = encoding_table({
        "rows": rows,
        "summary": {"jev-noul": {"arm": "jev-noul", "recall@5": 0.3,
                                 "cost_micro": 1,
                                 "reachable@5": 0.4, "cost_per_1k_micro": 1000}}})
    assert ">100<" in html          # 100 per question, not 300
    assert "1,000" in html          # 1000 input tokens per question, not 3000


def test_the_three_jev_arms_are_the_ones_the_runner_ships():
    from jevdemo.arms import ARMS
    assert JEV_ARMS == [a.name for a in ARMS if a.kind.startswith("jev")]


def test_chunk_preview_is_short_enough_that_a_hundred_rows_fit():
    assert 80 <= EXAMPLE_CHARS <= 160


def test_a_model_that_failed_every_query_does_not_block_the_example():
    # A key that ran out of budget mid-run leaves an arm with 350 failures and no
    # successes. Demanding a result row from it would make every query ineligible
    # and the appendix impossible to build, while proving nothing about any query:
    # the arm has no opinion to agree or disagree with. An arm that failed only
    # *some* queries is a different case and is still pinned, by the test above.
    tasks = {"tasks": [_task("q1"), _task("q2")]}
    rows = [_row("jev", "q1"), _row("jev", "q2"),
            _row("jev-noul", "q1"), _row("jev-noul", "q2"),
            _row("broke", "q1", failure="http_error"),
            _row("broke", "q2", failure="http_error")]
    assert _example_task(tasks, rows)["query_id"] in {"q1", "q2"}


# --- The common-denominator table -------------------------------------------
#
# The main leaderboard scores each arm over the questions it personally
# answered. That is a fair answer to "what will I get", and an unfair answer to
# "which is better", because two arms with different failure patterns then sit
# different exams. These guard the section that puts them on one exam.

def _report():
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_rag_report", root / "scripts" / "build_rag_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_common_subset_is_only_the_questions_everyone_answered():
    """If an arm's failures leak into the shared set, the whole point of the
    section is lost: it would be the same unequal comparison with a new title."""
    m = _report()
    rows = ([row("a", f"q{i}", 1.0) for i in range(3)]
            + [row("b", "q0", 0.0), row("b", "q1", 1.0),
               row("b", "q2", 0.0, failure="http_error")])
    common, scores = m._common_subset(rows, m.summarise(rows, SEED))
    assert common == ["q0", "q1"]
    assert dict(scores)["a"] == 1.0
    assert dict(scores)["b"] == 0.5


def test_an_arm_that_answered_nothing_does_not_empty_the_shared_set():
    """A model that failed every call carries no information about any question.
    Intersecting its empty set would silently delete the whole section."""
    m = _report()
    rows = ([row("a", f"q{i}", 1.0) for i in range(2)]
            + [row("b", f"q{i}", 1.0) for i in range(2)]
            + [row("dead", f"q{i}", 0.0, failure="http_error") for i in range(2)])
    common, scores = m._common_subset(rows, m.summarise(rows, SEED))
    assert common == ["q0", "q1"]
    assert "dead" not in dict(scores)


def test_movement_is_counted_against_the_same_arms_not_the_full_roster():
    """An arm missing from the shared set shifts every rank below it. Counting
    that as models 'changing place' would manufacture a finding out of nothing."""
    m = _report()
    # The scoreable arms keep their order; only `sonnet-5` is absent. It sorts
    # above the baseline in the main table, so dropping it shifts the baseline
    # up one rank -- the exact way this miscounted before.
    rows = ([row("jev-score", "q0", 1.0), row("jev-score", "q1", 1.0)]
            + [row("jev-noul", "q0", 0.0), row("jev-noul", "q1", 1.0)]
            + [row("sonnet-5", "q0", 0.0, failure="http_error"),
               row("sonnet-5", "q1", 0.0, failure="http_error")]
            + [row("bm25-baseline", "q0", 0.0, cost=0),
               row("bm25-baseline", "q1", 0.0, cost=0)])
    out = m._common_subset_tokens(rows, m.summarise(rows, SEED))
    assert "The order does not change" in out["common_finding"]


def test_the_spread_sentence_ignores_the_free_baseline():
    """The baseline is the thing being beaten, not a competitor. Including it
    would report the field as far wider apart than the models actually are."""
    m = _report()
    rows = ([row("jev-score", "q0", 1.0), row("jev-score", "q1", 1.0)]
            + [row("jev-noul", "q0", 1.0), row("jev-noul", "q1", 0.0)]
            + [row("bm25-baseline", "q0", 0.0, cost=0),
               row("bm25-baseline", "q1", 0.0, cost=0)])
    out = m._common_subset_tokens(rows, m.summarise(rows, SEED))
    # a=100%, b=50%, baseline=0%. The models are 50 points apart, not 100.
    assert "50.0 percentage points" in out["common_finding"]


def test_jev_headline_never_quotes_a_score_the_table_withholds():
    """Printing Choice's number in the opening line while the leaderboard calls
    it not comparable is the contradiction that made this report untrustworthy."""
    m = _report()
    rows = ([row("jev", "q0", 1.0)]
            + [row("jev", f"q{i}", 0.0, failure="http_error") for i in range(1, 10)]
            + [row("jev-score", f"q{i}", 0.4) for i in range(10)]
            + [row("jev-noul", f"q{i}", 0.4) for i in range(10)])
    text = m.jev_headline(m.summarise(rows, SEED))
    assert "100.0%" not in text          # Choice's score over its one answer
    assert "answered only 1 of 10" in text


def test_a_failed_call_is_not_a_free_call():
    """You pay for a call that comes back unusable; it just buys you nothing.

    Dividing total spend by attempts makes a model with a high failure rate look
    cheap in proportion to how often it failed. The honest unit is spend per
    *answer you can actually use*, so the denominator is the calls that scored.
    """
    rows = ([row("a", f"q{i}", 1.0, cost=500) for i in range(2)]
            + [row("a", f"f{i}", 0.0, cost=500, failure="http_error")
               for i in range(2)])
    s = summarise(rows, SEED)["a"]
    assert s["cost_micro"] == 2000, "every attempt was billed, including failures"
    # 2000 micro over the 2 answers it actually produced -> $1.00 per thousand.
    assert s["cost_per_1k_micro"] == 1_000_000


def test_an_arm_that_answered_nothing_reports_no_price_rather_than_zero():
    """Spend / 0 answers is not $0.00 — it is undefined, and must not print cheap."""
    rows = [row("a", f"q{i}", 0.0, cost=500, failure="http_error") for i in range(3)]
    s = summarise(rows, SEED)["a"]
    assert s["cost_per_1k_micro"] is None


def test_a_per_passage_arm_records_that_it_made_a_hundred_calls():
    """Its per-question price and latency are a hundred calls' worth, not one's.

    Without this the table silently compares one HTTP call against a hundred in
    the same 'typical speed' and 'cost' columns.
    """
    rows = [row("jev-score", f"q{i}", 1.0) for i in range(3)]
    assert summarise(rows, SEED)["jev-score"]["calls_per_query"] == 100
    rows = [row("jev", f"q{i}", 1.0) for i in range(3)]
    assert summarise(rows, SEED)["jev"]["calls_per_query"] == 1
    rows = [row("glm-5.3-flash-low", f"q{i}", 1.0) for i in range(3)]
    assert summarise(rows, SEED)["glm-5.3-flash-low"]["calls_per_query"] == 1
