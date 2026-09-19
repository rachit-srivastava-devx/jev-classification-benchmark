"""The renderer is the only path from measurements to prose, so it is tested.

Its one job beyond formatting: refuse to emit a report containing a claim the
results file cannot support. An unknown token is a typo or an invented number,
and both must stop the build rather than ship.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_report as br  # noqa: E402

from jevdemo.decision import distinguishability_matrix  # noqa: E402

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


# --- unmeasured arms -------------------------------------------------------

UNMEASURED = {
    **RESULTS,
    "arms": RESULTS["arms"] + [
        {"arm": "scout", "model": "m/scout", "kind": "chat", "reasoning": None,
         "attempted": 100, "scored": 0, "correct": 0, "accuracy": None, "measured": False,
         "accuracy_ci95": [0.0, 0.04], "failures": {"http_error": 100},
         "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
         "computed_cost_micro": 0, "reported_cost_micro": 0, "cost_reconciled": True,
         "cost_per_1000_micro": 0, "price_in_micro_per_mtok": 1, "price_out_micro_per_mtok": 1,
         "p50_latency_ms": None, "p95_latency_ms": None, "latency_samples": 0},
    ],
}


def test_an_unmeasured_arm_is_kept_out_of_the_matrix():
    """0 of 100 is resolvable against everything; leaving it in invents findings."""
    out = br.distinguishability_html(UNMEASURED)
    assert "Excluded as unmeasured" in out
    rows = out.count("<tr>")
    assert rows == 1 + len(RESULTS["arms"])  # header plus the measured arms only


def test_an_unmeasured_arm_is_kept_out_of_the_break_even_baseline():
    """Its zero cost would otherwise become the baseline every arm is priced against."""
    out = br.break_even_table_html(UNMEASURED)
    assert "Excluded as unmeasured" in out
    assert "scout</code></td>" not in out


def test_an_unmeasured_arm_cannot_be_the_best_arm():
    assert br.tokens(UNMEASURED)["best_arm"] == "dear"


# --- indented code blocks --------------------------------------------------
#
# The proofs are the centrepiece of this document and they are written as
# indented formula blocks. Rendering them as ordinary paragraphs reflows the
# algebra into prose, which is worse than not showing it.


def test_an_indented_block_renders_as_preformatted_text():
    out = br.markdown("Then:\n\n    C = (p_in * T_in) / 10^6\n\nTherefore.")
    assert "<pre>" in out
    assert "C = (p_in * T_in) / 10^6" in out


def test_a_multi_line_formula_keeps_its_line_breaks():
    out = br.markdown("    a = 1\n    b = 2\n")
    assert out.count("\n") >= 1
    assert "a = 1" in out and "b = 2" in out
    assert out.count("<pre>") == 1


def test_a_list_item_is_not_mistaken_for_an_indented_block():
    out = br.markdown("- first\n- second\n")
    assert "<pre>" not in out
    assert "<ul>" in out


def test_code_inside_a_block_is_escaped_not_interpreted():
    out = br.markdown("    <script>alert(1)</script>\n")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


# --- wrapped list items -----------------------------------------------------
# A numbered item that wraps onto a second line is one item. Before this, the
# loop stopped at the first continuation line and the remaining items reflowed
# into a paragraph, silently deleting the numbering from the corollaries.


def test_wrapped_list_item_stays_one_item():
    html_out = br.markdown(
        "1. first item that\n   wraps onto a second line\n2. second item\n"
    )
    assert html_out.count("<li>") == 2
    assert "first item that wraps onto a second line" in html_out


def test_wrapped_list_does_not_leak_into_a_paragraph():
    html_out = br.markdown(
        "1. alpha\n   continued\n2. beta\n\nAfterwards prose.\n"
    )
    assert "<p>Afterwards prose.</p>" in html_out
    assert "<p>alpha" not in html_out


def test_bullet_list_wraps_too():
    html_out = br.markdown("- one\n  more of one\n- two\n")
    assert html_out.count("<li>") == 2
    assert "one more of one" in html_out


def test_list_stops_at_a_blank_line():
    html_out = br.markdown("- one\n\n- two\n")
    assert html_out.count("<ul>") == 2


# --- code spans are literal -------------------------------------------------
# `K*` is a real symbol in the proofs. Its asterisk once opened an emphasis run
# that swallowed a sentence and a half of the corollaries into blue italic.


def test_asterisk_inside_a_code_span_is_not_emphasis():
    out = br.inline("`K*` scales with cost and inversely with `K*` again")
    assert "<em>" not in out
    assert out.count("<code>K*</code>") == 2


def test_emphasis_outside_code_spans_still_works():
    out = br.inline("the arm is *dominated* here")
    assert "<em>dominated</em>" in out


def test_bold_outside_code_spans_still_works():
    out = br.inline("**Claim.** something")
    assert "<strong>Claim.</strong>" in out


def test_code_span_content_is_escaped_not_marked_up():
    out = br.inline("`a < b` and `**x**`")
    assert "&lt;" in out
    assert "<strong>" not in out


# --- the headline table -----------------------------------------------------
# The full results table carries ten columns. That is the working table, not the
# one a reader should meet first. This one answers three questions — is it
# accurate, what does it cost, is it fast — and prices every row as a multiple
# of the cheapest, because "132 times" lands where "$2.44" does not.


def _two_arm_results():
    return {"arms": [
        {"arm": "jev", "kind": "jev", "reasoning": None, "accuracy": 0.92,
         "measured": True, "correct": 92, "attempted": 100, "scored": 100,
         "cost_per_1000_micro": 18500, "p50_latency_ms": 385.0, "p95_latency_ms": 558.0,
         "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0, "failures": {},
         "price_in_micro_per_mtok": 1000, "price_out_micro_per_mtok": 1000,
         "cost_reconciled": True, "accuracy_ci95": [0.85, 0.96]},
        {"arm": "opus-5", "kind": "chat", "reasoning": None, "accuracy": 0.94,
         "measured": True, "correct": 94, "attempted": 100, "scored": 100,
         "cost_per_1000_micro": 2441400, "p50_latency_ms": 3253.0, "p95_latency_ms": 6025.0,
         "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0, "failures": {},
         "price_in_micro_per_mtok": 1000, "price_out_micro_per_mtok": 1000,
         "cost_reconciled": True, "accuracy_ci95": [0.85, 0.96]},
    ]}


def test_headline_table_prices_every_row_against_the_cheapest():
    out = br.headline_table(_two_arm_results())
    assert "1&times;" in out or "1×" in out
    assert "132" in out  # opus-5 is 132x the price of jev


def test_headline_table_is_sorted_cheapest_first():
    out = br.headline_table(_two_arm_results())
    assert out.index("jev") < out.index("opus-5")


def test_headline_table_says_model_not_arm():
    out = br.headline_table(_two_arm_results())
    assert "<th>Model</th>" in out
    assert ">Arm<" not in out


def test_headline_table_excludes_an_unmeasured_model():
    results = _two_arm_results()
    results["arms"].append(
        {"arm": "llama-4-scout", "kind": "chat", "reasoning": None, "accuracy": None,
         "measured": False, "correct": 0, "attempted": 100, "scored": 0,
         "cost_per_1000_micro": 0, "p50_latency_ms": None, "p95_latency_ms": None,
         "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
         "failures": {"http_error": 100}})
    out = br.headline_table(results)
    assert "llama-4-scout" not in out


# --- cost vs effectiveness chart --------------------------------------------
# The spread is three orders of magnitude on cost and eighteen points on
# accuracy. A linear x-axis would pile eleven models into the left margin, so
# cost is logarithmic and the chart is drawn from results.json, never by hand.


def test_chart_plots_every_measured_model():
    out = br.cost_effectiveness_chart(_two_arm_results())
    assert out.startswith("<svg")
    assert "jev" in out and "opus-5" in out


def test_chart_omits_an_unmeasured_model():
    results = _two_arm_results()
    results["arms"].append(
        {"arm": "llama-4-scout", "kind": "chat", "reasoning": None, "accuracy": None,
         "measured": False, "correct": 0, "attempted": 100, "scored": 0,
         "cost_per_1000_micro": 0, "p50_latency_ms": None, "p95_latency_ms": None,
         "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
         "failures": {"http_error": 100}})
    assert "llama-4-scout" not in br.cost_effectiveness_chart(results)


def test_chart_labels_both_axes():
    out = br.cost_effectiveness_chart(_two_arm_results())
    assert "Cost per 1,000 tickets" in out
    assert "Accuracy" in out


def test_chart_separates_overlapping_labels():
    """Eighteen points inside an eighteen-point accuracy band collide by default."""
    results = {"arms": [
        dict(arm=f"m{i}", kind="chat", reasoning=None, accuracy=0.90, measured=True,
             correct=90, attempted=100, scored=100,
             cost_per_1000_micro=20000 + i, p50_latency_ms=100.0, p95_latency_ms=200.0,
             input_tokens=1, output_tokens=1, reasoning_tokens=0, failures={})
        for i in range(8)]}
    out = br.cost_effectiveness_chart(results)
    ys = [float(m) for m in re.findall(r"class='pt-label'[^>]*? y='([\d.]+)'", out)]
    assert len(ys) == 8
    assert all(abs(a - b) >= 9 for a, b in zip(sorted(ys), sorted(ys)[1:]))


def test_chart_survives_a_single_model():
    results = {"arms": [_two_arm_results()["arms"][0]]}
    assert br.cost_effectiveness_chart(results).startswith("<svg")


def test_chart_with_no_measured_model_says_so_rather_than_dividing_by_zero():
    out = br.cost_effectiveness_chart({"arms": []})
    assert "<svg" not in out
    assert out.strip() != ""


# --- categories and worked examples -----------------------------------------
# Both are rendered from the same files the run used. A definition retyped into
# the report is a definition that can drift from the one the models were given.


def test_category_table_renders_every_label_definition():
    out = br.category_table()
    from jevdemo.labels import LABELS
    for name, definition in LABELS.items():
        assert name in out
        assert html.escape(definition) in out


def test_examples_table_shows_one_ticket_per_category():
    from jevdemo.labels import LABELS
    tickets = [{"id": f"t{i}", "text": f"ticket text {i}", "label": name}
               for i, name in enumerate(LABELS)]
    out = br.examples_table(tickets)
    assert out.count("<tr>") == len(LABELS) + 1  # +1 for the head row
    for name in LABELS:
        assert name in out


def test_examples_table_escapes_ticket_text():
    out = br.examples_table([{"id": "t0", "text": "a <script> & b", "label": "billing"}])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_examples_table_with_no_tickets_says_so():
    out = br.examples_table([])
    assert "<table" not in out


def test_headline_tokens_are_present():
    results = _two_arm_results()
    results["totals"] = {"records": 200, "computed_cost_micro": 0}
    results["records"] = []
    results["reasoning_negotiation"] = {}
    table = br.tokens(results, None)
    for key in ("jev_acc_cost", "dearest_model", "dearest_ratio", "unrecon_count"):
        assert key in table, key


def test_chart_paints_with_presentation_attributes_not_only_css():
    """WeasyPrint's CSS engine rejects `fill`, so a chart styled only by class
    prints as undifferentiated black. Every painted element carries its own
    attribute; the classes stay for the screen."""
    svg = br.cost_effectiveness_chart(_two_arm_results())
    for frag in ("<circle", "<text", "<line"):
        for el in re.findall(frag + r"[^>]*>", svg):
            assert "fill=" in el or "stroke=" in el, el
    assert br.ACCENT in svg          # the Jev point is the one accent on the chart


def _judge_fixture():
    return {"judge_model": "m", "rows_judged": 84, "rows_skipped_arm_failure": 16,
            "overall_agreement_scored": 84, "overall_agreement_matched": 83,
            "overall_agreement_rate": 0.988, "rank_correlation_spearman": 0.84,
            "judge_cost_micro": 1_533_532,
            "arms": [{"arm": "mistral-small-low", "rows": 84,
                      "gold_accuracy": 0.9405, "judge_accuracy": 0.9286,
                      "judge_accuracy_ci95": [0.85, 0.97], "agreement_rate": 0.988,
                      "mean_probability": 0.917, "self_judged": False}]}


def test_judge_table_publishes_the_denominator_it_scored_over():
    """Its gold column is computed over judged rows only, so a model that lost calls
    reads 94% here and 79% in the measurement table. Without the row count and a
    stated denominator that looks like the report contradicting itself."""
    out = br.judge_html(_judge_fixture())
    assert ">84<" in out                      # the denominator is on the row
    assert "judged rows" in out.lower()       # and named in the header or caption


def test_judge_caption_says_model_not_arm():
    # A word-boundary match, not a substring one: "Spearman" contains "arm".
    assert not re.search(r"\barms?\b", br.judge_html(_judge_fixture()), re.I)


def _real_results():
    """The live measurements. These assertions are about the shipped document, so
    a synthetic fixture would prove nothing about the numbers it prints."""
    import json
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "results.json"
    if not p.exists():
        import pytest
        pytest.skip("results.json not present; run the experiment first")
    return json.loads(p.read_text())


def test_cost_divergence_range_is_computed_not_typed():
    """The report quoted a hand-typed 10%-53% range. Both ends were wrong. The
    range is now derived from the same fields that set `unrecon_count`."""
    results = _real_results()
    t = br.tokens(results, None)
    unrecon = [a for a in results["arms"] if not a.get("cost_reconciled", True)]
    div = sorted(
        abs(a["computed_cost_micro"] - a["reported_cost_micro"]) / a["reported_cost_micro"]
        for a in unrecon
    )
    assert t["divergence_lo"] == f"{div[0] * 100:.0f}%"
    assert t["divergence_hi"] == f"{div[-1] * 100:.0f}%"
    assert t["unrecon_count"] == str(len(unrecon))


def test_jev_row_counts_come_from_the_matrix_that_is_printed():
    """Appendix B claimed Jev's row was all ties while the matrix below it showed
    a win. The claim is now the same computation as the matrix."""
    results = _real_results()
    t = br.tokens(results, None)
    arms = br._measured(results)
    row = distinguishability_matrix(arms)["jev"]
    verdicts = [v["verdict"] for k, v in row.items() if k != "jev"]
    assert t["jev_better_than"] == str(verdicts.count("better"))
    assert t["jev_worse_than"] == str(verdicts.count("worse"))
    assert t["jev_ties"] == str(verdicts.count("tie"))


def test_each_model_carries_how_many_models_beat_it():
    """Section 3 called one model 'the only model this sample can call worse than
    the rest'. Two models are separable, not one."""
    results = _real_results()
    t = br.tokens(results, None)
    arms = br._measured(results)
    m = distinguishability_matrix(arms)
    for a in arms:
        slug = a["arm"].replace(".", "_").replace("-", "_")
        beaten_by = sum(
            1 for b in arms if b["arm"] != a["arm"]
            and m[b["arm"]][a["arm"]]["verdict"] == "better"
        )
        assert t[f"{slug}_lost_to"] == str(beaten_by), a["arm"]
    assert t["separable_count"] == str(
        sum(1 for a in arms
            if any(m[b["arm"]][a["arm"]]["verdict"] == "better"
                   for b in arms if b["arm"] != a["arm"]))
    )
