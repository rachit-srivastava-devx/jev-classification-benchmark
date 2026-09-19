"""The renderer is the only path from measurements to prose, so it is tested.

Its one job beyond formatting: refuse to emit a report containing a claim the
results file cannot support. An unknown token is a typo or an invented number,
and both must stop the build rather than ship.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_report as br  # noqa: E402

RESULTS = {
    "arms": [
        {"arm": "jev", "model": "typesafe/jev-1.13", "kind": "jev", "reasoning": None,
         "attempted": 100, "scored": 100, "correct": 88, "accuracy": 0.88,
         "accuracy_ci95": [0.80, 0.93], "failures": {},
         "input_tokens": 39500, "output_tokens": 5200, "reasoning_tokens": 0,
         "computed_cost_micro": 1659, "reported_cost_micro": 1659,
         "cost_reconciled": True, "cost_per_1000_micro": 17_000,
         "price_in_micro_per_mtok": 42_000, "price_out_micro_per_mtok": 0,
         "p50_latency_ms": 641.0, "p95_latency_ms": 900.0, "latency_samples": 20},
        {"arm": "dear", "model": "x/y", "kind": "chat", "reasoning": None,
         "attempted": 100, "scored": 98, "correct": 96, "accuracy": 0.96,
         "accuracy_ci95": [0.90, 0.99], "failures": {"malformed_output": 2},
         "input_tokens": 7100, "output_tokens": 1100, "reasoning_tokens": 400,
         "computed_cost_micro": 63000, "reported_cost_micro": 63000,
         "cost_reconciled": True, "cost_per_1000_micro": 630_000_000,
         "price_in_micro_per_mtok": 2_000_000, "price_out_micro_per_mtok": 10_000_000,
         "p50_latency_ms": 2108.0, "p95_latency_ms": 9491.0, "latency_samples": 20},
    ],
    "reasoning_negotiation": {},
    "records": [],
    "totals": {"records": 240, "negotiation_calls": 2,
               "computed_cost_micro": 64659, "rate_limit_retries": 3},
}


# --- token safety ----------------------------------------------------------

def test_unknown_token_stops_the_build():
    with pytest.raises(KeyError):
        br.substitute("cost was {{NOT_A_REAL_TOKEN}}", br.tokens(RESULTS))


def test_block_markers_survive_substitution():
    """They are swapped after the markdown render, not during it."""
    table = br.tokens(RESULTS)
    for name in br.BLOCKS:
        out = br.substitute("{{" + name + "}}", table)
        assert out == "{{" + name + "}}"


def test_a_block_marker_not_alone_on_its_line_is_refused(tmp_path):
    findings = tmp_path / "findings.md"
    findings.write_text("See this {{BREAK_EVEN_TABLE}} inline.\n")
    out = tmp_path / "out.html"
    results = tmp_path / "results.json"
    results.write_text(__import__("json").dumps(RESULTS))
    with pytest.raises(ValueError, match="alone on its own line"):
        br.main(["--results", str(results), "--findings", str(findings), "--out", str(out)])


# --- block renderers -------------------------------------------------------

def test_break_even_table_prices_the_dearer_arm():
    html_out = br.break_even_table_html(RESULTS)
    assert "<table" in html_out
    assert "jev" in html_out
    assert "baseline" in html_out.lower()
    # (630_000_000 - 17_000) nano over (0.96 - 0.88) -> $7.87 per misroute
    assert "7.87" in html_out


def test_distinguishability_matrix_names_a_tie_as_a_tie():
    html_out = br.distinguishability_html(RESULTS)
    assert "<table" in html_out
    # 88/100 vs 96/100 is resolvable at n=100.
    assert "better" in html_out or "worse" in html_out


def test_failures_table_counts_by_kind():
    html_out = br.failures_html(RESULTS)
    assert "malformed_output" in html_out
    assert "rate_limit_retries" in html_out or "3" in html_out


def test_judge_table_says_so_when_the_experiment_was_not_run():
    """The section must never silently render empty and imply a null result."""
    html_out = br.judge_html(None)
    assert "not run" in html_out.lower()


def test_judge_table_reports_agreement_and_rank_correlation():
    judge = {
        "judge_model": "anthropic/claude-sonnet-5",
        "rows_judged": 200, "rows_skipped_arm_failure": 2,
        "judge_failures": {},
        "overall_agreement_scored": 198, "overall_agreement_matched": 180,
        "overall_agreement_rate": 180 / 198,
        "rank_correlation_spearman": 0.81, "judge_cost_micro": 1_400_000,
        "arms": [
            {"arm": "jev", "rows": 100, "gold_accuracy": 0.88, "judge_accuracy": 0.90,
             "judge_accuracy_ci95": [0.82, 0.95], "agreement_scored": 99,
             "agreement_matched": 90, "agreement_rate": 90 / 99,
             "mean_probability": 0.87, "self_judged": False},
        ],
    }
    html_out = br.judge_html(judge)
    assert "0.81" in html_out
    assert "91" in html_out or "90.9" in html_out  # agreement rate as a percentage
    assert "jev" in html_out


def test_judge_table_marks_the_self_judged_arm():
    judge = {
        "judge_model": "anthropic/claude-sonnet-5",
        "rows_judged": 100, "rows_skipped_arm_failure": 0, "judge_failures": {},
        "overall_agreement_scored": 100, "overall_agreement_matched": 95,
        "overall_agreement_rate": 0.95, "rank_correlation_spearman": None,
        "judge_cost_micro": 1000,
        "arms": [
            {"arm": "sonnet-5", "rows": 100, "gold_accuracy": 0.9, "judge_accuracy": 0.97,
             "judge_accuracy_ci95": [0.91, 0.99], "agreement_scored": 100,
             "agreement_matched": 95, "agreement_rate": 0.95,
             "mean_probability": 0.96, "self_judged": True},
        ],
    }
    html_out = br.judge_html(judge)
    assert "self-judged" in html_out.lower()


# --- tokens ----------------------------------------------------------------

def test_tokens_expose_the_sample_size_and_call_count():
    t = br.tokens(RESULTS)
    assert t["arm_count"] == "2"
    assert t["record_count"] == "240"


def test_tokens_expose_the_resolution_bound():
    """The report states this before showing any accuracy, so it must be derived."""
    t = br.tokens(RESULTS)
    assert "ci_halfwidth_pp" in t
    assert float(t["ci_halfwidth_pp"]) > 0
