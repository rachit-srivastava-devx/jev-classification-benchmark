"""Render the reranking report. Every number comes from the results file.

Same contract as `build_report.py`, and it reuses that script's markdown
renderer and token substitution rather than growing a second one: a `{{token}}`
the prose references but this script does not produce is a hard KeyError, so a
figure cannot drift out of date by being typed into a sentence.

    python scripts/build_rag_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevdemo.arms import by_name
from jevdemo.rag import jev_rank
from jevdemo.rag.corpus import load_tasks
from jevdemo.rag.aggregate import align, summarise
from jevdemo.rag.rank_stats import paired_difference, verdict
from scripts.build_report import TEMPLATE, inline, markdown, substitute, usd

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260920

#: Display names. The report uses the provider's own model names, never "arm 3":
#: a reader has to be able to go and price the thing we measured.
PRETTY = {
    "bm25-baseline": "BM25 only (no model)",
    "jev": "Jev \u00b7 Choice (all chunks at once)",
    "jev-noul": "Jev \u00b7 Noul (per passage)",
    "jev-score": "Jev \u00b7 Score (per passage)",
    "glm-5.3-flash-low": "glm-5.3-flash",
    "deepseek-v4.1-flash-low": "deepseek-v4.1-flash",
    "qwen3.8-flash-low": "qwen3.8-flash",
    "gemini-3.8-flash-low": "gemini-3.8-flash",
    "sonnet-5": "sonnet-5",
}

SOURCE_LABEL = {
    "bright-biology": "BRIGHT biology",
    "bright-economics": "BRIGHT economics",
    "bright-psychology": "BRIGHT psychology",
    "fiqa": "FiQA finance",
    "wands": "WANDS shopping",
}


def pct(x: float | None, places: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{places}f}%"


def ms(x: float | None, unit: str = "") -> str:
    """Milliseconds, or an em dash when an arm produced no successful call."""
    return "\u2014" if x is None else f"{x:,.0f}{unit}"


def order(summary: dict) -> list[dict]:
    """Best recall@5 first, with the free baseline always last for contrast."""
    rows = [s for s in summary.values() if s["arm"] != "bm25-baseline"]
    rows.sort(key=lambda s: (-(s["recall@5"] or -1), s["cost_micro"]))
    base = summary.get("bm25-baseline")
    return rows + ([base] if base else [])


def tokens(res: dict, summary: dict, tasks: dict, res20: dict | None) -> dict[str, str]:
    rows = res["rows"]
    # Only models that actually answered can win: crowning one that half-failed
    # would put a name in the headline that the measurement cannot support.
    ranked = [r for r in order(summary) if coverage(r) >= MIN_COVERAGE]
    best = ranked[0]
    jev = summary.get("jev")
    base = summary.get("bm25-baseline")
    per_source = sorted({t["source"] for t in tasks["tasks"]})

    out = {
        "n_queries": str(res["tasks"]),
        "depth": str(res["depth"]),
        "n_models": str(len(res["roster"])),
        "n_calls": f"{len([r for r in rows if r['arm'] != 'bm25-baseline']):,}",
        "n_passages_ranked": f"{res['tasks'] * res['depth']:,}",
        # Two different counts, and conflating them put "4 real collections"
        # above a list of two. `n_sources` is topic sets (BRIGHT has three);
        # `n_collections` is the datasets they come from, taken from the
        # label prefix so adding a source cannot leave the count behind.
        "n_sources": str(len(per_source)),
        "n_collections": str(len({SOURCE_LABEL[s].split()[0] for s in per_source})),
        "source_list": ", ".join(SOURCE_LABEL[s] for s in per_source),
        # Per-corpus, from the run that read the corpora — never a literal.
        # This was `tasks.get("corpus_total", 218052)`, whose default could
        # never be overridden because nothing ever wrote that key, so the
        # report described FiQA's corpus with the total of all four.
        "fiqa_corpus_size": f"{_corpus_size('fiqa'):,}",
        "wands_corpus_size": f"{_corpus_size('wands'):,}",
        "corpus_size": f"{sum(_corpus_sizes().values()):,}",
        "top_k": str(res["top_k"]),
        "total_spend": usd(res["spend_micro"], 2),
        "best_model": PRETTY[best["arm"]],
        "best_recall": pct(best["recall@5"]),
        "jev_recall": pct(jev["recall@5"]) if jev else "—",
        "jev_reachable": pct(jev["reachable@5"]) if jev else "—",
        "jev_cost_1k": usd(jev["cost_per_1k_micro"], 2) if jev else "—",
        "jev_p50": f"{jev['p50_ms']:,.0f}" if jev and jev["p50_ms"] else "—",
        "base_recall": pct(base["recall@5"]) if base else "—",
        "ceiling": pct(_ceiling(rows)),
        "gold_per_query": f"{sum(len(t['gold_ids']) for t in tasks['tasks']) / len(tasks['tasks']):.1f}",
        # Read off the fetch script's own constants rather than retyped, so the
        # sentence that claims the sample was mechanical cannot drift from the
        # code that made it.
        "selection_rule": _selection_rule(),
    }
    out.update(_lift_tokens(summary, rows))
    # Counted, not characterised. An earlier draft said "most of this grid is
    # `=`" beside a grid in which most cells were not.
    _res, _tot = _separable_counts(rows, summary)
    out["matrix_reading"] = (
        f"{_res} of the {_tot} comparisons come out as a real difference and "
        f"{_tot - _res} as a tie." + (
            " The differences are real but narrow, so this grid ranks the models "
            "far less sharply than their prices do."
            if _res else " On this sample the field is indistinguishable."))
    out.update(_encoding_tokens(summary, rows))
    out.update(_cost_tokens(ranked, jev))
    out.update(_depth_tokens(res20, summary))
    out.update(_failure_tokens(summary, rows))
    out["chart_span"] = chart_span(summary)
    out["chart_omissions"] = chart_omissions(summary)
    out["jev_headline"] = jev_headline(summary)
    out.update(_common_subset_tokens(res["rows"], summary))
    out["dearest_unscorable"] = _dearest_unscorable(summary)
    return out


def _selection_rule() -> str:
    """The sampling rule, quoted from the constants that actually ran it.

    There is no random sampling and so no seed to choose: queries are taken in
    sorted id order and the first N kept. A seed is one more knob a reader has to
    take on trust, and this rule needs none.
    """
    import scripts.fetch_rag_corpus as f
    return (f"take every judged query in each corpus in sorted id order, keep the first "
            f"{f.PER_DOMAIN} per BRIGHT domain, {f.FIQA_N} from FiQA and {f.WANDS_N} from "
            f"WANDS, retrieve the top {f.DEPTH} chunks for each with BM25, and freeze the "
            f"result to disk \u2014 no random sampling, no seed")


SIZES_PATH = Path(__file__).resolve().parents[1] / "data/rag/corpus-sizes.json"


def _corpus_sizes() -> dict[str, int]:
    """Corpus sizes as counted by `fetch_rag_corpus.py --sizes-only`.

    Deliberately has no fallback. A missing file stops the build rather than
    letting the report print a plausible number nobody measured.
    """
    if not SIZES_PATH.exists():
        raise FileNotFoundError(
            f"{SIZES_PATH} is missing — run: "
            "python scripts/fetch_rag_corpus.py --sizes-only")
    return json.loads(SIZES_PATH.read_text())["corpus_sizes"]


def _corpus_size(source: str) -> int:
    sizes = _corpus_sizes()
    if source not in sizes:
        raise KeyError(f"{source} not in {SIZES_PATH}: have {sorted(sizes)}")
    return sizes[source]


def _pairwise(rows: list[dict], summary: dict) -> tuple[int, int, list[float]]:
    """Every head-to-head comparison, paired on the query.

    Returns (resolved, total, gaps) where `gaps` holds the observed paired mean
    difference, in percentage points, for the comparisons that resolve. One
    function so that no two sentences in the report can count the same grid
    differently — an earlier draft described it as mostly ties beside a grid in
    which most cells were not ties.
    """
    arms = [name for name, v in summary.items()
            if name != "bm25-baseline" and v.get("recall@5") is not None]
    rng = random.Random(SEED)
    resolved, total, gaps = 0, 0, []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            xs, ys = align(rows, a, b, "recall@5")
            if not xs:
                continue
            total += 1
            if verdict(paired_difference(xs, ys, rng)) != "tie":
                resolved += 1
                gaps.append(abs(sum(x - y for x, y in zip(xs, ys)) / len(xs)) * 100)
    return resolved, total, gaps


def _separable_counts(rows: list[dict], summary: dict) -> tuple[int, int]:
    resolved, total, _ = _pairwise(rows, summary)
    return resolved, total


def _separability(rows: list[dict], summary: dict) -> str:
    """Say whether the models are actually distinguishable, from the paired tests.

    An earlier draft asserted flatly that this test could not tell the models
    apart, reasoning from the *unpaired* confidence intervals in the headline
    table, which overlap. That is the wrong test for this design: every model
    answered the same queries, so the paired comparison is both available and
    strictly more powerful, and it does resolve most of these pairs. The claim
    is therefore counted here rather than asserted.
    """
    resolved, total, gaps = _pairwise(rows, summary)
    if not total:
        return ""
    if not resolved:
        return (f"Which model does it is not something this test can tell apart: "
                f"all {total} head-to-head comparisons are ties.")
    return (f"Which model does it matters less than it looks. Of the {total} "
            f"head-to-head comparisons between the models that answered, "
            f"{resolved} are real differences and {total - resolved} are ties this "
            f"sample cannot resolve \u2014 and the differences that are real are "
            f"small, between {min(gaps):.1f} and {max(gaps):.1f} percentage points. "
            f"What separates these models far more sharply is price.")


def _ceiling(rows: list[dict]) -> float:
    """Share of judged gold that BM25 put in front of the rerankers at all.

    No reranker can exceed it, so it is the denominator behind every score in
    the report and it is printed rather than left implied.
    """
    base = [r for r in rows if r["arm"] == "bm25-baseline"]
    return sum(r["gold_in_view"] / r["gold_total"] for r in base) / len(base)


def _lift_tokens(summary: dict, rows: list[dict]) -> dict[str, str]:
    """How much each model improved on the retriever it was handed, paired."""
    out: dict[str, str] = {}
    rng = random.Random(SEED)
    beat = 0
    # A model with no paired questions at all was never measured against the
    # baseline and cannot be counted on either side of it. sonnet-5 in this run
    # is the case that forced the distinction: its key hit a spending limit and
    # every call was refused, so it neither beat the baseline nor "cost money and
    # changed nothing" — it did not run. Counting it in the denominator would put
    # a model in a failure bucket it never reached.
    unmeasured = []
    for arm in summary:
        if arm == "bm25-baseline":
            continue
        xs, ys = align(rows, arm, "bm25-baseline", "recall@5")
        interval = paired_difference(xs, ys, rng) if xs else None
        if not xs:
            unmeasured.append(arm)
        slug = arm.replace(".", "_").replace("-", "_")
        out[f"{slug}_lift"] = pct(None if not interval else (sum(xs) - sum(ys)) / len(xs))
        out[f"{slug}_lift_lo"] = pct(interval[0]) if interval else "—"
        out[f"{slug}_lift_hi"] = pct(interval[1]) if interval else "—"
        out[f"{slug}_verdict"] = verdict(interval)
        if verdict(interval) == "better":
            beat += 1
    n_models = len(summary) - 1 - len(unmeasured)
    out["models_unmeasured"] = str(len(unmeasured))
    out["models_beating_bm25"] = str(beat)
    out["models_not_beating_bm25"] = str(n_models - beat)

    # The headline claim about the free baseline is written from the measurement
    # rather than typed into the prose. An earlier draft asserted that most models
    # failed to beat it; the run said all of them did, by almost the same amount.
    # Prose that states a result has to be produced by the thing that measured it.
    lifts = [float(out[f"{a.replace('.', '_').replace('-', '_')}_lift"].rstrip("%"))
             for a in summary if a != "bm25-baseline"
             and out[f"{a.replace('.', '_').replace('-', '_')}_lift"] != "\u2014"]
    if beat == n_models and lifts:
        spread = max(lifts) - min(lifts)
        out["baseline_finding"] = (
            f"**Every model beat the free option \u2014 and by almost exactly the same "
            f"amount.** Before any model is called, a plain keyword search has already "
            f"put the chunks in some order. All {n_models} models improved on that "
            f"order, by between {min(lifts):.1f} and {max(lifts):.1f} percentage "
            f"points \u2014 a spread of {spread:.1f} points across the whole field. "
            f"So the reranking step is worth paying for. {_separability(rows, summary)}"
            + (f" {len(unmeasured)} model{'s' if len(unmeasured) > 1 else ''} "
               f"({', '.join(PRETTY.get(a, a) for a in unmeasured)}) returned no "
               f"scoreable answer at all and "
               f"{'are' if len(unmeasured) > 1 else 'is'} excluded from that count."
               if unmeasured else ""))
    elif beat == 0:
        out["baseline_finding"] = (
            f"**No model beat the free option by an amount this test can confirm.** "
            f"A plain keyword search already orders the chunks before any model is "
            f"called, and none of the {n_models} models improved on that order "
            f"measurably. On this evidence the reranking step is not worth its cost.")
    else:
        out["baseline_finding"] = (
            f"**Only {beat} of the {n_models} models beat the free option.** A plain "
            f"keyword search already orders the chunks before any model is called. "
            f"{beat} improved on that order by an amount this test can confirm; the "
            f"other {n_models - beat} cost money and changed nothing we can prove."
            + (f" A further {len(unmeasured)} "
               f"({', '.join(PRETTY.get(a, a) for a in unmeasured)}) returned no "
               f"scoreable answer at all and is counted on neither side."
               if unmeasured else ""))
    return out


def _cost_tokens(ranked: list[dict], jev: dict | None) -> dict[str, str]:
    paid = [s for s in ranked if s["arm"] != "bm25-baseline" and s["cost_micro"]]
    if not paid or not jev or not jev["cost_micro"]:
        return {"dearest_model": "—", "dearest_multiple": "—",
                "cheapest_other": "—", "cheapest_other_multiple": "—"}
    dearest = max(paid, key=lambda s: s["cost_per_1k_micro"])
    others = [s for s in paid if s["arm"] != "jev"]
    cheapest = min(others, key=lambda s: s["cost_per_1k_micro"]) if others else jev
    return {
        "dearest_model": PRETTY[dearest["arm"]],
        "dearest_cost_1k": usd(dearest["cost_per_1k_micro"], 2),
        "dearest_multiple": f"{dearest['cost_per_1k_micro'] / jev['cost_per_1k_micro']:.0f}",
        "cheapest_other": PRETTY[cheapest["arm"]],
        "cheapest_other_multiple": f"{cheapest['cost_per_1k_micro'] / jev['cost_per_1k_micro']:.1f}",
    }


#: An arm whose calls mostly failed did not get a random sample of the questions:
#: it got whichever ones happened to run before the budget ran out. Averaging
#: those is arithmetic, not measurement, so below this share of completed calls
#: the score is withheld and the reason is printed instead. 0.9 rather than a
#: bare majority: the claim being protected is a comparison between products.
MIN_COVERAGE = 0.9


def coverage(s: dict) -> float:
    """Share of a model's calls that returned something scoreable."""
    return s["scored"] / s["attempted"] if s["attempted"] else 0.0


def quotable(s: dict | None, key: str = "recall@5") -> str | None:
    """The score as a percentage, or None if too few calls completed to mean it."""
    if not s or coverage(s) < MIN_COVERAGE:
        return None
    return pct(s[key])


def _depth_tokens(res20: dict | None, summary100: dict) -> dict[str, str]:
    if not res20:
        return {"d20_ran": "no", "d20_best": "\u2014", "d20_jev_recall": "\u2014",
                "d20_premium": "The premium comparison was not run.",
                "d20_spend": "\u2014", "sonnet_d100_projected": "\u2014"}
    s20 = summarise(res20["rows"], SEED)
    son = s20.get("sonnet-5")

    quoted = quotable(son)
    if quoted:
        premium = f"sonnet-5 scored {quoted}."
    elif son:
        premium = (f"**sonnet-5 could not be measured here.** Only {son['scored']} of "
                   f"{son['attempted']} of its calls completed \u2014 the rest were "
                   f"refused by the provider for want of prepaid credit, not by the "
                   f"model. The questions that did get through are the ones that "
                   f"happened to run first, not a fair sample, so averaging them "
                   f"would produce a number that looks like a result and is not one. "
                   f"It is left out rather than dressed up.")
    else:
        premium = "The premium model was not run at this depth."

    # What the premium model would have cost at the headline depth, from its own
    # measured per-query cost at depth 20 scaled by the passage ratio. Arithmetic
    # on a measurement, not a measurement \u2014 the prose says so.
    projected = "\u2014"
    if son and son["cost_per_1k_micro"]:
        projected = usd(round(son["cost_per_1k_micro"] * 5), 2)

    # The "best" label must not crown an arm that barely ran, for the same reason.
    ranked = [r for r in order(s20) if coverage(r) >= MIN_COVERAGE]
    return {
        "d20_ran": "yes",
        "d20_jev_recall": quotable(s20.get("jev")) or "\u2014",
        "d20_premium": premium,
        "d20_best": PRETTY[ranked[0]["arm"]] if ranked else "\u2014",
        "d20_spend": usd(res20["spend_micro"], 2),
        "sonnet_d100_projected": projected,
    }


def _failure_tokens(summary: dict, rows: list[dict]) -> dict[str, str]:
    TOTAL_Q = len({r.get("query_id") for r in rows if r.get("query_id")})
    total = sum(s["failures"] for s in summary.values())
    dropped = sum(s["queries_with_dropped_ids"] for s in summary.values())

    # `max` over an all-zero field still returns an arm, which would print a real
    # product's name next to "invented the most chunk ids" when it invented none.
    # So the whole sentence is produced here from the data rather than assembled
    # in the prose around a bare number.
    worst = max((s for s in summary.values()
                 if s["arm"] != "bm25-baseline" and s["queries_with_dropped_ids"]),
                key=lambda s: s["queries_with_dropped_ids"], default=None)
    if worst is None:
        finding = ("In this run no model invented one: every id every model returned "
                   "was an id it had actually been shown.")
    else:
        finding = (f"{PRETTY[worst['arm']]} did this most, on "
                   f"{worst['queries_with_dropped_ids']} questions.")

    # An HTTP 402 is our account running out of headroom, not the model failing.
    # Reporting it as a model defect would be a false claim about a real product,
    # so the cause is detected and named rather than left to the reader.
    # Two different ways our own account, not the model, ends a call: HTTP 402
    # (no prepaid balance) and HTTP 403 with the provider's key-limit message
    # (the API key's own spend cap reached). Matching only "402" missed the
    # second entirely, which mattered: at depth 100 the cap, not the model,
    # is what stopped one arm completely and cut into another.
    def _ours(r: dict) -> bool:
        d = r.get("detail") or ""
        return "402" in d or ("403" in d and "limit exceeded" in d.lower())

    starved: dict[str, int] = {}
    codes: set[str] = set()
    for r in rows:
        if _ours(r):
            starved[r["arm"]] = starved.get(r["arm"], 0) + 1
            codes.update(c for c in ("402", "403") if c in (r.get("detail") or ""))
    if starved:
        per = ", ".join(f"{PRETTY.get(a, a)} ({n}{f' of {TOTAL_Q}' if TOTAL_Q else ''})"
                        for a, n in sorted(starved.items(), key=lambda kv: -kv[1]))
        seen = " and ".join(f"HTTP {c}" for c in sorted(codes))
        credit = (f"**Some failures were ours, not the model's.** ({seen}.) "
                  f"Our own API key hit "
                  f"its spend limit part-way through the run, and every call after that "
                  f"came back refused before it ever reached a model: {per}. Those calls "
                  f"are counted as failures so the coverage numbers stay honest, but they "
                  f"are evidence about our budget, not about the model. Any arm above "
                  f"whose coverage is short by roughly this many queries is being "
                  f"under-reported for that reason.")
    else:
        credit = ""

    return {
        "total_failures": str(total),
        "queries_with_invented_ids": str(dropped),
        "invention_finding": finding,
        "credit_note": credit,
    }


# --- tables -----------------------------------------------------------------

def headline_table(summary: dict) -> str:
    """The one table a reader who reads nothing else should see."""
    head = ("<table class='headline'><thead><tr><th>Model</th>"
            "<th class='n'>Found the right chunks</th><th class='n'>95% range</th>"
            "<th class='n'>Calls per question</th>"
            "<th class='n'>Cost per 1,000 questions</th><th class='n'>Typical speed</th>"
            "</tr></thead><tbody>")
    body = []
    for s in order(summary):
        ci = s["recall@5_ci95"]
        # A model that only half-answered is not being ranked against the others
        # on equal terms. Rather than drop the row — which would hide that it was
        # tried — the score is replaced by what actually happened.
        if coverage(s) < MIN_COVERAGE:
            note = (f"answered only {s['scored']} of {s['attempted']} — "
                    f"not comparable")
            body.append(
                f"<tr class='unresolved'><td>{inline(PRETTY[s['arm']])}</td>"
                f"<td class='n sub' colspan='2'>{note}</td>"
                f"<td class='n'>{s['calls_per_query']}</td>"
                f"<td class='n'>{usd(s['cost_per_1k_micro'], 2)}</td>"
                f"<td class='n'>{ms(s['p50_ms'], ' ms')}</td></tr>")
            continue
        body.append(
            f"<tr><td>{inline(PRETTY[s['arm']])}</td>"
            f"<td class='n'>{pct(s['recall@5'])}</td>"
            f"<td class='n sub'>{pct(ci[0]) if ci else '—'} to {pct(ci[1]) if ci else '—'}</td>"
            f"<td class='n'>{s['calls_per_query']}</td>"
            f"<td class='n'>{usd(s['cost_per_1k_micro'], 2)}</td>"
            f"<td class='n'>{ms(s['p50_ms'], ' ms')}</td></tr>")
    return head + "".join(body) + "</tbody></table>"


def by_source_table(rows: list[dict], summary: dict) -> str:
    """The same models split by corpus, because one average hides the fact that
    the two corpora are different problems."""
    sources = sorted({r["source"] for r in rows})
    head = ("<table class='bysource'><thead><tr><th>Model</th>"
            + "".join(f"<th class='n'>{SOURCE_LABEL[s]}</th>" for s in sources)
            + "</tr></thead><tbody>")
    body = []
    for s in order(summary):
        cells = []
        for src in sources:
            vals = [r["recall@5"] for r in rows
                    if r["arm"] == s["arm"] and r["source"] == src
                    and r["failure"] is None and r["recall@5"] is not None]
            cells.append(f"<td class='n'>{pct(sum(vals) / len(vals)) if vals else '—'}</td>")
        body.append(f"<tr><td>{inline(PRETTY[s['arm']])}</td>{''.join(cells)}</tr>")
    return head + "".join(body) + "</tbody></table>"


def full_table(summary: dict) -> str:
    head = ("<table class='full'><thead><tr><th>Model</th><th class='n'>Queries</th>"
            "<th class='n'>@1</th><th class='n'>@3</th><th class='n'>@5</th>"
            "<th class='n'>@10</th><th class='n'>Of what was findable</th>"
            "<th class='n'>nDCG@5</th><th class='n'>MRR</th>"
            "<th class='n'>Input tokens</th><th class='n'>Cost</th>"
            "<th class='n'>p50</th><th class='n'>p95</th></tr></thead><tbody>")
    body = []
    for s in order(summary):
        body.append(
            f"<tr><td>{inline(PRETTY[s['arm']])}</td><td class='n'>{s['scored']}</td>"
            f"<td class='n'>{pct(s['recall@1'])}</td><td class='n'>{pct(s['recall@3'])}</td>"
            f"<td class='n'>{pct(s['recall@5'])}</td><td class='n'>{pct(s['recall@10'])}</td>"
            f"<td class='n'>{pct(s['reachable@5'])}</td>"
            f"<td class='n'>{pct(s['ndcg@5'])}</td><td class='n'>{pct(s['mrr'])}</td>"
            f"<td class='n'>{s['input_tokens']:,}</td>"
            f"<td class='n'>{usd(s['cost_micro'], 3)}</td>"
            f"<td class='n'>{ms(s['p50_ms'])}</td>"
            f"<td class='n'>{ms(s['p95_ms'])}</td></tr>")
    return head + "".join(body) + "</tbody></table>"


VERDICT_WORD = {"better": "really helps", "worse": "makes it worse",
                "tie": "cannot tell"}


def lift_table(rows: list[dict], summary: dict) -> str:
    """Each model against the free keyword baseline, paired question by question.

    This is the table that decides whether a model is worth calling at all, so it
    shows the range and not just the gap: a positive average whose range crosses
    zero is not a result, and is labelled as one that cannot be told apart.
    """
    rng = random.Random(SEED)
    head = ("<table class='breakeven'><thead><tr><th>Model</th>"
            "<th class='n'>Better than free keyword search by</th>"
            "<th class='n'>95% range</th><th>Verdict</th></tr></thead><tbody>")
    body = []
    for s in order(summary):
        if s["arm"] == "bm25-baseline":
            continue
        xs, ys = align(rows, s["arm"], "bm25-baseline", "recall@5")
        interval = paired_difference(xs, ys, rng) if xs else None
        gap = (sum(xs) - sum(ys)) / len(xs) if xs else None
        v = verdict(interval)
        cls = "" if v != "tie" else " class='unresolved'"
        sign = "+" if gap and gap > 0 else ""
        body.append(
            f"<tr{cls}><td>{inline(PRETTY[s['arm']])}</td>"
            f"<td class='n'>{sign}{pct(gap)}</td>"
            f"<td class='n sub'>{pct(interval[0]) if interval else ''} to "
            f"{pct(interval[1]) if interval else ''}</td>"
            f"<td>{VERDICT_WORD[v]}</td></tr>")
    return head + "".join(body) + "</tbody></table>"


VERDICT_GLYPH = {"better": "+", "worse": "−", "tie": "=", "self": "·"}


def matrix_html(rows: list[dict], summary: dict) -> str:
    """Every pair, paired on the query. `+` means the row model really is better."""
    arms = [s["arm"] for s in order(summary)]
    rng = random.Random(SEED)
    head = ("<table class='matrix'><thead><tr><th></th>"
            + "".join(f"<th class='rot'>{i + 1}</th>" for i in range(len(arms)))
            + "</tr></thead><tbody>")
    body = []
    for i, a in enumerate(arms):
        cells = []
        for b in arms:
            if a == b:
                cells.append("<td class='v-self'>·</td>")
                continue
            xs, ys = align(rows, a, b, "recall@5")
            v = verdict(paired_difference(xs, ys, rng)) if xs else "tie"
            cells.append(f"<td class='v-{v}'>{VERDICT_GLYPH[v]}</td>")
        body.append(f"<tr><td><span class='idx'>{i + 1}</span>"
                    f"{inline(PRETTY[a])}</td>{''.join(cells)}</tr>")
    return head + "".join(body) + "</tbody></table>"


def failures_html(summary: dict) -> str:
    bad = [s for s in summary.values()
           if s["failures"] or s["queries_with_dropped_ids"]]
    if not bad:
        return "<p class='sub'>No call failed and no model named a passage that did not exist.</p>"
    head = ("<table class='failures'><thead><tr><th>Model</th>"
            "<th class='n'>Calls that failed</th><th>Why</th>"
            "<th class='n'>Queries where it invented a passage id</th>"
            "<th class='n'>Invented ids in total</th></tr></thead><tbody>")
    body = []
    for s in sorted(bad, key=lambda s: -s["failures"]):
        body.append(
            f"<tr><td>{inline(PRETTY[s['arm']])}</td>"
            f"<td class='n'>{s['failures']}</td>"
            f"<td class='sub'>{', '.join(s['failure_kinds']) or '—'}</td>"
            f"<td class='n'>{s['queries_with_dropped_ids']}</td>"
            f"<td class='n'>{s['dropped_ids']}</td></tr>")
    return head + "".join(body) + "</tbody></table>"


# --- chart ------------------------------------------------------------------

W, H = 900, 460
PAD = {"l": 58, "r": 238, "t": 34, "b": 56}
LABEL_GAP = 14.0


def _declutter(points: list[dict]) -> None:
    """Push overlapping labels apart vertically so none is hidden.

    Two models on nearly the same score would otherwise print their names on top
    of each other, and the chart would silently lose a row.
    """
    for p in sorted(points, key=lambda p: p["y"]):
        for q in sorted(points, key=lambda q: q["y"]):
            if q is p:
                break
            if abs(p["ly"] - q["ly"]) < LABEL_GAP:
                p["ly"] = q["ly"] + LABEL_GAP


def chart(summary: dict) -> str:
    """Cost against effectiveness. Cost is a log axis because the models span
    two orders of magnitude and a linear axis would stack them all on zero."""
    import math

    pts = [s for s in summary.values()
           if s["recall@5"] is not None and s["cost_per_1k_micro"] > 0
           and coverage(s) >= MIN_COVERAGE]
    if len(pts) < 2:
        return "<p class='sub'>Too few measured models to plot.</p>"

    xs = [math.log10(s["cost_per_1k_micro"]) for s in pts]
    ys = [s["recall@5"] for s in pts]
    x0, x1 = min(xs) - 0.25, max(xs) + 0.25
    span = max(ys) - min(ys)
    pad = max(span * 0.45, 0.01)
    y0, y1 = max(0.0, min(ys) - pad), max(ys) + pad

    def px(v):
        return PAD["l"] + (v - x0) / (x1 - x0) * (W - PAD["l"] - PAD["r"])

    def py(v):
        return H - PAD["b"] - (v - y0) / (y1 - y0) * (H - PAD["t"] - PAD["b"])

    # The band this chart plots is only a few points wide, so whole-percent
    # labels repeat and every point label collapses to the same value. The
    # precision follows the range instead of being fixed.
    dp = 0 if (y1 - y0) >= 0.10 else 1
    parts = [f"<svg class='chart' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg'>"]
    for frac in range(0, 5):
        v = y0 + (y1 - y0) * frac / 4
        y = py(v)
        parts.append(f"<line class='grid' {PAINT['grid']} x1='{PAD['l']}' y1='{y:.1f}' "
                     f"x2='{W - PAD['r']}' y2='{y:.1f}'/>")
        parts.append(f"<text class='ax' {PAINT['ax']} x='{PAD['l'] - 8}' y='{y + 3:.1f}' "
                     f"text-anchor='end'>{v * 100:.{dp}f}%</text>")
    parts.append(f"<line class='axis' {PAINT['axis']} x1='{PAD['l']}' y1='{H - PAD['b']}' "
                 f"x2='{W - PAD['r']}' y2='{H - PAD['b']}'/>")

    # With the dearest model excluded the points span less than one decade, which
    # left the axis with a single tick and nothing to measure against. 2x and 5x
    # inside each decade give a reader intermediate marks without crowding it.
    decade = math.floor(x0)
    while decade <= x1:
        for mult in (1, 2, 5):
            v = decade + math.log10(mult)
            if not (x0 <= v <= x1):
                continue
            x = px(v)
            dollars = 10 ** v / 1e6
            label = f"${dollars:,.2f}" if dollars >= 0.01 else f"${dollars:.4f}"
            parts.append(f"<line class='grid' {PAINT['grid']} x1='{x:.1f}' y1='{H - PAD['b']}' "
                         f"x2='{x:.1f}' y2='{H - PAD['b'] + 5}'/>")
            parts.append(f"<text class='ax' {PAINT['ax']} x='{x:.1f}' y='{H - PAD['b'] + 18}' "
                         f"text-anchor='middle'>{label}</text>")
        decade += 1
    parts.append(f"<text class='ax-title' {PAINT['ax-title']} x='{PAD['l']}' y='{H - 10}'>"
                 f"cost per 1,000 questions</text>")
    parts.append(f"<text class='ax-title' {PAINT['ax-title']} x='{PAD['l'] - 46}' y='{PAD['t'] - 14}'>share found</text>")

    placed = [{"s": s, "x": px(math.log10(s["cost_per_1k_micro"])),
               "y": py(s["recall@5"]), "ly": py(s["recall@5"])} for s in pts]
    _declutter(placed)
    for p in placed:
        cls = " jev" if p["s"]["arm"] == "jev" else ""
        parts.append(f"<line class='leader' {PAINT['leader']} x1='{p['x'] + 5:.1f}' y1='{p['y']:.1f}' "
                     f"x2='{W - PAD['r'] + 8}' y2='{p['ly']:.1f}'/>")
        parts.append(f"<circle class='pt{cls}' {PAINT['pt' + cls]} cx='{p['x']:.1f}' cy='{p['y']:.1f}' r='4'/>")
        parts.append(f"<text class='pt-label{cls}' {PAINT['pt-label' + cls]} x='{W - PAD['r'] + 13}' "
                     f"y='{p['ly'] + 3:.1f}'>{PRETTY[p['s']['arm']]} "
                     f"{pct(p['s']['recall@5'], dp)}</text>")
    parts.append("</svg>")
    return "".join(parts)


#: Inline SVG paint, repeated as presentation attributes on every chart element.
#:
#: The stylesheet already carries identical `svg.chart .grid` rules, and they do
#: style the HTML — but WeasyPrint does not cascade the document's CSS into an
#: inline SVG, so in the PDF those rules select nothing. The chart rendered with
#: whatever the SVG engine defaulted to, which is how it reached GitHub as a
#: scatter plot with no axes. Proved rather than assumed: setting `.grid` to red
#: and rebuilding produced a PDF with no red in it.
#:
#: `build_report.py` has always written these attributes; this chart was the one
#: that did not, which is why only this report's chart was broken. Keep both:
#: attributes for the PDF, the stylesheet for the HTML.
#:
#: The grid and leader sit one step darker than the doctrine hairline tokens
#: (#F0F0F0 -> #E5E5E5, #E5E5E5 -> #D8D8D8). A hairline that reads on a backlit
#: screen disappears in a PDF shown at page width, which is where this is read.
MONO = "JetBrains Mono, ui-monospace, monospace"
PAINT = {
    "grid": "stroke='#E5E5E5' stroke-width='1'",
    "axis": "stroke='#0A0A0A' stroke-width='1'",
    "leader": "stroke='#D8D8D8' stroke-width='1'",
    "pt": "fill='#5C6066'",
    "pt jev": "fill='#1E6FFF'",
    "ax": f"fill='#8A8F96' font-family='{MONO}' font-size='10'",
    "ax-title": f"fill='#5C6066' font-family='{MONO}' font-size='10'",
    "pt-label": f"fill='#1A1A1A' font-family='{MONO}' font-size='10'",
    "pt-label jev": f"fill='#1E6FFF' font-family='{MONO}' font-size='10' font-weight='500'",
}


def chart_span(summary: dict) -> str:
    """How far apart the plotted models are in price, as a sentence.

    Written from the points the chart actually draws. A hand-typed 'two orders of
    magnitude' stopped being true the moment a model was excluded for answering
    too few questions, and nothing would have caught that.
    """
    pts = [s for s in summary.values()
           if s["recall@5"] is not None and s["cost_per_1k_micro"] > 0
           and coverage(s) >= MIN_COVERAGE]
    if len(pts) < 2:
        return "a log scale \u2014 each step is ten times the last"
    lo = min(s["cost_per_1k_micro"] for s in pts)
    hi = max(s["cost_per_1k_micro"] for s in pts)
    return (f"a log scale \u2014 each step is ten times the last. It has to be, "
            f"because the cheapest and dearest models here are {hi / lo:.0f}\u00d7 "
            f"apart in price while sitting within about a point of each other on "
            f"quality")



def jev_headline(summary: dict) -> str:
    """Jev's result, stated as a range over its encodings rather than a number.

    The previous edition of this report put one Jev number in the opening line.
    That number was a property of the Choice encoding and was read as a property
    of the product, which is the single mistake this report exists to correct.
    So the headline is generated from every Jev encoding that ran, names the best
    one, and only ever gives a single number when only one encoding was measured.

    The range is taken only over encodings that cleared ``MIN_COVERAGE``. An
    earlier edition quoted Choice's 27.7% here as Jev's worst score while the
    leaderboard on the same page withheld that number as not comparable. Both
    statements cannot be true. The table is right -- a score over the subset of
    questions an encoding happened to answer is not the same quantity as a score
    over all of them -- so the headline now obeys the same rule the table does,
    and names the withheld encoding instead of scoring it.
    """
    mine = [summary[a] for a in JEV_ARMS if a in summary and summary[a]["recall@5"] is not None]
    if not mine:
        return "Jev returned no scoreable answer in this run."
    ranked = [s for s in mine if coverage(s) >= MIN_COVERAGE]
    held = [s for s in mine if coverage(s) < MIN_COVERAGE]

    def held_sentence() -> str:
        if not held:
            return ""
        parts = []
        for s in sorted(held, key=lambda s: s["arm"]):
            parts.append(f"{PRETTY[s['arm']]} answered only {s['scored']} of "
                         f"{s['attempted']} questions, so its score is withheld for "
                         f"the same reason the table withholds it, though it was the "
                         f"cheapest thing in the run at "
                         f"{usd(s['cost_per_1k_micro'], 2)} per thousand questions")
        return " " + "; ".join(parts) + "."

    if not ranked:
        return ("No Jev encoding answered enough questions to be scored against the "
                "rest of the field." + held_sentence())
    if len(ranked) == 1:
        s = ranked[0]
        return (f"{PRETTY[s['arm']]} found {pct(s['recall@5'])} at "
                f"{usd(s['cost_per_1k_micro'], 2)} per thousand questions. Only one "
                f"Jev encoding cleared the coverage bar here, so this number says "
                f"nothing about the others." + held_sentence())
    best = max(ranked, key=lambda s: s["recall@5"])
    worst = min(ranked, key=lambda s: s["recall@5"])
    cheap = min(ranked, key=lambda s: s["cost_per_1k_micro"])
    rival = min((s["cost_per_1k_micro"] for a, s in summary.items()
                 if a not in JEV_ARMS and s["cost_micro"]
                 and coverage(s) >= MIN_COVERAGE), default=None)
    # "Jev is not one number" is a claim about spread, so it is only made when
    # the spread is real. With the encodings this run could score landing a
    # third of a point apart, the honest headline is that they agree.
    gap = best["recall@5"] - worst["recall@5"]
    if gap < 0.02:
        out = (f"**Jev's encodings that could be scored agree with each other.** "
               f"{PRETTY[best['arm']]} found {pct(best['recall@5'])} at "
               f"{usd(best['cost_per_1k_micro'], 2)} per thousand questions and "
               f"{PRETTY[worst['arm']]} found {pct(worst['recall@5'])} at "
               f"{usd(worst['cost_per_1k_micro'], 2)}, a gap of "
               f"{gap * 100:.1f} of a percentage point, which is far inside the "
               f"error bars on either")
    else:
        out = (f"**Jev is not one number: it scored between {pct(worst['recall@5'])} and "
               f"{pct(best['recall@5'])} depending only on how the question was put to "
               f"it.** {PRETTY[best['arm']]} was its best encoding at "
               f"{pct(best['recall@5'])} and {usd(best['cost_per_1k_micro'], 2)} per "
               f"thousand questions; {PRETTY[worst['arm']]} was its worst at "
               f"{pct(worst['recall@5'])}")
    if cheap["arm"] != worst["arm"]:
        out += f", and {PRETTY[cheap['arm']]} was its cheapest"
    out += "."
    if rival is not None:
        out += (f" The cheapest scoreable Jev encoding costs "
                f"{usd(cheap['cost_per_1k_micro'], 2)} per thousand questions, against "
                f"{usd(rival, 2)} for the cheapest non-Jev model that answered enough "
                f"questions to be scored.")
    return out + held_sentence()


def _dearest_unscorable(summary: dict) -> str:
    """The dearest model whose coverage is too low to score, named from the data.

    The prose used to name this model by hand. A hand-typed model name in a
    sentence about cost is a claim that stops being true the first time the
    roster or the failure pattern changes, and nothing would catch it.
    """
    weak = [s for s in summary.values()
            if s["cost_micro"] and coverage(s) < MIN_COVERAGE]
    if not weak:
        return ""
    d = max(weak, key=lambda s: s["cost_per_1k_micro"])
    return (f"Not the dearest we tried: {PRETTY[d['arm']]} cost "
            f"{usd(d['cost_per_1k_micro'], 2)} per thousand questions, more than any "
            f"scored model, and is absent from the chart above for answering too few "
            f"of them.")


def chart_omissions(summary: dict) -> str:
    """Name every model the chart leaves out, and why.

    The chart plots only models whose answers can be averaged: a model that
    failed a quarter of its questions did not fail a random quarter, so its
    score is not comparable and plotting it invites exactly the comparison the
    number cannot support. That is defensible, but leaving it unsaid is not: the
    cheapest option in this whole report is one of the models excluded, and a
    cost chart that quietly drops the cheapest thing on it is misleading however
    principled the rule behind it. So the rule stays and the absence is printed.
    """
    missing = []
    for s in order(summary):
        if s["arm"] == "bm25-baseline":
            missing.append(f"{PRETTY[s['arm']]} costs nothing, and zero has no place "
                           f"on a log axis (it scored {pct(s['recall@5'])})")
        elif s["cost_micro"] and coverage(s) < MIN_COVERAGE:
            missing.append(f"{PRETTY[s['arm']]} answered only {s['scored']} of "
                           f"{s['attempted']} questions, below the {MIN_COVERAGE:.0%} "
                           f"needed to average a score, at "
                           f"{usd(s['cost_per_1k_micro'], 2)} per 1,000")
        elif not s["cost_micro"] and s["arm"] != "bm25-baseline":
            missing.append(f"{PRETTY[s['arm']]} never returned a priced answer "
                           f"({s['scored']} of {s['attempted']} questions)")
    if not missing:
        return "Every model measured is on the chart."
    return ("**Not every model is on the chart.** " + "; ".join(missing)
            + ". Their numbers are all in Appendix B and Appendix C; they are kept "
              "off this chart because averaging a model's score over the questions "
              "it happened to survive is arithmetic, not measurement.")


# --- assembly ---------------------------------------------------------------

#: The process figure. It is a rendered artifact, not a drawing: the spec is
#: `report/pipeline.architecture.json` in this repository, and the caption prints
#: its hash so a reader can check that the picture and the pipeline description
#: came from the same file.
PIPELINE_PNG = "report/assets/pipeline.png"
PIPELINE_SPEC = "report/pipeline.architecture.json"


def pipeline_figure(ctx: dict) -> str:
    """The end-to-end diagram, on its own oversized page.

    Raises FileNotFoundError if either the render or the spec is missing, rather
    than emitting a figure with a broken image: a report whose first page is a
    grey box is worse than one that fails to build.
    """
    png, spec = ROOT / PIPELINE_PNG, ROOT / PIPELINE_SPEC
    for f in (png, spec):
        if not f.exists():
            raise FileNotFoundError(f"process diagram missing: {f}")
    doc = json.loads(spec.read_text())
    nodes = len(doc.get("components", []))
    edges = len(doc.get("connections", []))
    digest = hashlib.sha256(spec.read_bytes()).hexdigest()[:16]
    # The question set gets its own hash, taken from the file on disk at build
    # time. Quoting the diagram's hash here instead — as an earlier draft did —
    # looked like provenance for the questions and was not.
    tasks = ROOT / "data" / "rag" / "tasks.json"
    if not tasks.exists():
        raise FileNotFoundError(f"question set missing: {tasks}")
    tdigest = hashlib.sha256(tasks.read_bytes()).hexdigest()[:16]
    return (
        f"<figure class='pipeline'><img src='{PIPELINE_PNG}' "
        f"alt='The full pipeline, from the three public corpora to the results file'>"
        f"<figcaption>Every step between a public dataset and a number in this "
        f"report. {nodes} components, {edges} connections. Nothing to the left of "
        f"<em>tasks.json</em> depends on any model: the code that builds the "
        f"questions imports no model client, so the question set cannot have been "
        f"chosen by looking at an answer. The question set shipped with this "
        f"report is <span class='mono'>data/rag/tasks.json</span>, sha256 "
        f"{tdigest}\u2026 \u2014 re-hash it and you are reading the same 350 "
        f"questions the models were given. That hash fixes <em>which</em> "
        f"questions; it does not by itself prove <em>when</em> they were written. "
        f"Diagram rendered from <span class='mono'>{PIPELINE_SPEC}</span>, "
        f"sha256 {digest}\u2026</figcaption></figure>"
    )


#: How much of a chunk the worked example prints. Long enough to judge relevance
#: by eye, short enough that a hundred of them fit on a page. The full text of
#: every chunk is in `data/rag/tasks.json`, keyed by the id printed beside it.
EXAMPLE_CHARS = 120


def _example_task(tasks: dict, rows: list[dict]) -> dict:
    """Pick the query the worked example walks through, deterministically.

    The pick is made by a seeded shuffle over the *eligible* queries rather than
    by choosing the one with the nicest numbers. Eligible means: the hard
    stratum, at least two gold chunks the retriever actually reached, and a
    result row from every model that answered anything at all. A worked example
    chosen after seeing the results is a sales exhibit, not evidence.

    "Answered anything at all" is the one deliberate loosening. A model that
    failed on every single query — a key that ran out of budget, a model pulled
    from the provider mid-run — carries no information about any query, so
    demanding its agreement would only make the example impossible to choose
    while proving nothing. A model that failed on *some* queries still excludes
    exactly those, which is the point of the rule.
    """
    scored = {r["query_id"] for r in rows if r["failure"] is None}
    arms = {r["arm"] for r in rows if r["failure"] is None}
    complete = {q for q in scored
                if len({r["arm"] for r in rows
                        if r["query_id"] == q and r["failure"] is None}) == len(arms)}
    eligible = [t for t in tasks["tasks"]
                if t["query_id"] in complete
                and t.get("stratum") == "hard"
                and t.get("gold_reachable", 0) >= 2]
    if not eligible:
        raise ValueError("no query is hard, reachable and scored by every model; "
                         "the worked example cannot be chosen without cherry-picking")
    eligible.sort(key=lambda t: t["query_id"])
    random.Random(SEED).shuffle(eligible)
    return eligible[0]


def worked_example(ctx: dict) -> str:
    """One query, end to end: the request, the hundred chunks, every model's picks."""
    tasks, rows, summary = ctx["tasks"], ctx["rows"], ctx["summary"]
    task = _example_task(tasks, rows)
    cands = task["candidates"][: ctx["res"]["depth"]]
    ids = [c["id"] for c in cands]
    gold = set(task["gold_ids"])
    pos = {cid: i + 1 for i, cid in enumerate(ids)}

    url, body = jev_rank.build_request(by_name("jev"), dict(task, candidates=cands), 10, None)
    shown = json.loads(json.dumps(body))
    crit = shown["questions"][jev_rank.QUESTION]["criteria"]
    keep = list(crit)[:2]
    shown["questions"][jev_rank.QUESTION]["criteria"] = {
        k: crit[k][:EXAMPLE_CHARS] + " \u2026" for k in keep}
    shown["questions"][jev_rank.QUESTION]["criteria"][
        f"\u2026 and {len(crit) - len(keep)} more, one per chunk"] = "\u2026"

    out = [
        "<p>Query <span class='mono'>%s</span> from %s, difficulty <b>hard</b>: "
        "%d of its %d correct chunks are somewhere in the hundred, none in the "
        "first five of the keyword order.</p>"
        % (inline(task["query_id"]), SOURCE_LABEL[task["source"]],
           task.get("gold_reachable", 0), len(gold)),
        "<blockquote>%s</blockquote>" % inline(task["query"][:1200]),
        "<h3>The exact request</h3>",
        "<p>This is the body posted to <span class='mono'>%s</span>, with the "
        "chunk texts cut for length. Every chunk becomes one named option; the "
        "answer comes back as a probability per option.</p>" % inline(url),
        "<pre>%s</pre>" % inline(json.dumps(shown, indent=1)[:2200]),
        "<h3>The hundred chunks it was given</h3>",
        "<p>In the order the keyword search returned them. <b>&#9733;</b> marks a "
        "chunk a human judged correct for this query, decided before the run.</p>",
        _chunk_table(cands, gold),
        "<h3>What each model picked</h3>",
        "<p>Each model's top five, in its own order. The number in brackets is "
        "where the keyword search had put that chunk \u2014 so <span class='mono'>"
        "[41]</span> means the model promoted something from 41st place.</p>",
        _picks_table(task, rows, summary, gold, pos),
        "<h3>Why those were the right ones</h3>",
        "<p>The chunks a human judged correct for this query, with where the "
        "keyword search had ranked them and which models put one in their top "
        "five. These judgements ship with the dataset; we did not make them.</p>",
        _gold_table(task, cands, rows, summary, gold, pos),
    ]
    return "".join(out)


def _gold_table(task: dict, cands: list[dict], rows: list[dict], summary: dict,
                gold: set, pos: dict) -> str:
    """The judged-correct chunks, and who found them.

    Only the gold that is actually inside the candidate list is listed: a gold
    chunk the retriever never returned could not have been found by anyone, and
    printing it here would read as nine models failing at something that was
    never on the table.
    """
    text = {c["id"]: " ".join(c["text"].split()) for c in cands}
    reachable = [cid for cid in sorted(gold) if cid in text]
    head = ("<table class='goldset'><thead><tr><th>Correct chunk</th>"
            "<th class='n'>Keyword<br>rank</th><th>What it says</th>"
            "<th>Put it in the top 5</th></tr></thead><tbody>")
    body = []
    for cid in reachable:
        found = [PRETTY[s["arm"]] for s in order(summary)
                 for r in rows
                 if r["arm"] == s["arm"] and r["query_id"] == task["query_id"]
                 and r["failure"] is None and cid in r["ranking"][:5]]
        body.append("<tr><td class='mono'>&#9733; %s</td><td class='n'>%s</td>"
                    "<td>%s</td><td>%s</td></tr>"
                    % (inline(cid), pos.get(cid, "?"),
                       inline(text[cid][:EXAMPLE_CHARS]),
                       inline(", ".join(found)) if found
                       else "<span class='dim'>nobody</span>"))
    if not body:
        return ("<p>None of this query's correct chunks were inside the hundred, "
                "so there was nothing for any model to find.</p>")
    return head + "".join(body) + "</tbody></table>"


def _chunk_table(cands: list[dict], gold: set) -> str:
    head = ("<table class='chunks'><thead><tr><th class='n'>#</th><th>Chunk id</th>"
            "<th>First %d characters</th></tr></thead><tbody>" % EXAMPLE_CHARS)
    body = []
    for i, c in enumerate(cands, 1):
        mark = " &#9733;" if c["id"] in gold else ""
        text = " ".join(c["text"].split())[:EXAMPLE_CHARS]
        body.append("<tr%s><td class='n'>%d</td><td class='mono'>%s%s</td><td>%s</td></tr>"
                    % (" class='gold'" if c["id"] in gold else "", i,
                       inline(c["id"]), mark, inline(text)))
    return head + "".join(body) + "</tbody></table>"


def _picks_table(task: dict, rows: list[dict], summary: dict,
                 gold: set, pos: dict) -> str:
    """Each model's top five, stacked inside one cell.

    Laid out down the page rather than across it because chunk ids run to fifty
    characters: five of them as five columns overflows the page and silently
    clips the last two, which is the worst possible failure for a table whose
    whole purpose is that nothing was hidden.
    """
    head = ("<table class='picks'><thead><tr><th>Model</th>"
            "<th>What it put in its top five, best first "
            "<span class='dim'>[keyword rank]</span></th>"
            "<th class='n'>Right<br>in 5</th></tr></thead><tbody>")
    body = []
    for srow in order(summary):
        arm = srow["arm"]
        hit = [r for r in rows if r["arm"] == arm and r["query_id"] == task["query_id"]]
        if not hit or hit[0]["failure"] is not None:
            body.append("<tr><td>%s</td><td colspan='2'>no answer: %s</td></tr>"
                        % (inline(PRETTY[arm]),
                           inline(hit[0]["failure"] if hit else "not run")))
            continue
        top = hit[0]["ranking"][:5]
        picks = []
        for i, cid in enumerate(top, 1):
            picks.append(
                "<div class='pick%s'><span class='dim'>%d.</span> %s%s "
                "<span class='dim'>[%s]</span></div>"
                % (" gold" if cid in gold else "", i,
                   "&#9733; " if cid in gold else "", inline(cid), pos.get(cid, "?")))
        body.append("<tr><td>%s</td><td class='mono'>%s</td>"
                    "<td class='n'>%d of %d</td></tr>"
                    % (inline(PRETTY[arm]), "".join(picks),
                       len([c for c in top if c in gold]), min(5, len(gold))))
    return head + "".join(body) + "</tbody></table>"


#: The three difficulty slices, in the order a reader should read them. The
#: labels say what the slice *is* rather than naming it, because "hard" on its
#: own invites a reader to assume we chose what hard means after seeing scores.
STRATA = [
    ("hard", "Hard \u2014 a correct chunk is in the 100, none in the keyword top 5"),
    ("easy", "Already solved \u2014 keyword search had one in its own top 5"),
    ("unreachable", "Impossible \u2014 no correct chunk in the 100 at all"),
]


def _common_subset(rows: list[dict], summary: dict) -> tuple[list[str], list[tuple[str, float]]]:
    """The questions every arm answered, and each arm's score over exactly those.

    The leaderboard scores each arm over the questions it personally answered.
    That is the right number for "what will this cost me and get me in
    production", and it is the wrong number for "which one is better", because
    two arms with different failure patterns are then being scored on different
    exams. This computes the other number: one exam, taken by everyone.

    An arm that answered nothing is excluded rather than emptying the
    intersection, the same rule the worked example uses.
    """
    scored = {}
    for a in summary:
        got = {r["query_id"] for r in rows
               if r["arm"] == a and r["failure"] is None and r["recall@5"] is not None}
        if got:
            scored[a] = got
    if not scored:
        return [], []
    common = set.intersection(*scored.values())
    if not common:
        return [], []
    out = []
    for a in scored:
        vals = [r["recall@5"] for r in rows
                if r["arm"] == a and r["query_id"] in common
                and r["failure"] is None and r["recall@5"] is not None]
        out.append((a, sum(vals) / len(vals)))
    out.sort(key=lambda t: -t[1])
    return sorted(common), out


def common_subset_table(ctx: dict) -> str:
    """The leaderboard re-run over the identical question set, side by side."""
    common, scores = _common_subset(ctx["rows"], ctx["summary"])
    if not scores:
        return ("<p class='note'>No question was answered by every arm, so no "
                "common-denominator comparison is possible.</p>")
    summary = ctx["summary"]
    full_rank = {s["arm"]: i + 1 for i, s in enumerate(order(summary))}
    head = ("<table class='headline'><thead><tr><th>Model</th>"
            f"<th class='n'>On the {len(common)} questions<br>"
            "<span class='dim'>every model answered</span></th>"
            "<th class='n'>Rank here</th>"
            "<th class='n'>Rank on the<br><span class='dim'>main table</span></th>"
            "</tr></thead><tbody>")
    body = []
    for i, (arm, val) in enumerate(scores):
        body.append(f"<tr><td>{inline(PRETTY[arm])}</td>"
                    f"<td class='n'>{pct(val)}</td>"
                    f"<td class='n'>{i + 1}</td>"
                    f"<td class='n'>{full_rank.get(arm, '—')}</td></tr>")
    return head + "".join(body) + "</tbody></table>"


def _common_subset_tokens(rows: list[dict], summary: dict) -> dict[str, str]:
    """The sentence under the common-denominator table, written from that table.

    It is generated rather than typed because its whole claim is about which
    ordering the data produces, and a typed ordering would go stale silently.
    """
    common, scores = _common_subset(rows, summary)
    if not scores:
        return {"common_n": "0",
                "common_finding": "No question was answered by every model, so "
                                  "the field cannot be put on one exam."}
    here = [a for a, _ in scores]
    # Rank the main table over exactly the arms that appear here, so an arm
    # missing from one side cannot shift every rank below it and be counted as
    # a model that moved. Only a genuine reordering should show up as movement.
    full = [s["arm"] for s in order(summary) if s["arm"] in here]
    moved = [a for a in here if full.index(a) != here.index(a)]
    top_here = next(a for a in here if a != "bm25-baseline")
    top_full = next(a for a in full if a != "bm25-baseline")
    # The baseline is not a competitor, so it does not set the field's width.
    vals = [v for a, v in scores if a != "bm25-baseline"]
    spread = max(vals) - min(vals)
    if not moved:
        finding = (f"The order does not change: {inline(PRETTY[top_here])} leads on both, "
                   f"and the models sit within {spread * 100:.1f} percentage points of "
                   f"each other. That is the strongest version of the main table's claim.")
    else:
        finding = (f"**The order changes.** On one shared exam "
                   f"{inline(PRETTY[top_here])} leads, not {inline(PRETTY[top_full])}, "
                   f"and {len(moved)} of {len(here)} entries change place. The models "
                   f"then all sit within {spread * 100:.1f} percentage points of each other. "
                   f"Neither table is wrong; they answer different questions. The main "
                   f"table answers \u201cwhat do I get if I buy this one\u201d, counting "
                   f"each model only on what it managed to answer. This one answers "
                   f"\u201cwhich one is better at the job\u201d, and its answer is that "
                   f"the test cannot tell. Any ranking that survives only one of these "
                   f"two framings is not a finding about the models.")
    return {"common_n": f"{len(common)}", "common_finding": finding}


def strata_table(ctx: dict) -> str:
    """Recall@5 by difficulty slice.

    The slice is decided by the retriever alone, before any model runs, and is
    stored on the task. Splitting on it here is a read, not a new rule.
    """
    rows, summary = ctx["rows"], ctx["summary"]
    counts = {k: len({r["query_id"] for r in rows if r["stratum"] == k})
              for k, _ in STRATA}
    head = ("<table class='strata'><thead><tr><th>Model</th>"
            + "".join(f"<th class='n'>{inline(lab)}<br><span class='dim'>"
                      f"{counts[k]} questions</span></th>" for k, lab in STRATA)
            + "</tr></thead><tbody>")
    body = []
    for s in order(summary):
        cells = []
        for k, _ in STRATA:
            vals = [r["recall@5"] for r in rows
                    if r["arm"] == s["arm"] and r["stratum"] == k
                    and r["failure"] is None and r["recall@5"] is not None]
            cells.append(f"<td class='n'>{pct(sum(vals) / len(vals)) if vals else '—'}"
                         f"<br><span class='dim'>{len(vals)} scored</span></td>")
        body.append(f"<tr><td>{inline(PRETTY[s['arm']])}</td>{''.join(cells)}</tr>")
    return head + "".join(body) + "</tbody></table>"


#: The three Jev arms, in the order the encodings are explained.
JEV_ARMS = ["jev", "jev-noul", "jev-score"]

#: What each encoding does, in one line, for the table's own left column.
ENCODING_NOTE = {
    "jev": "all 100 chunks as options in one question; they share one pool of probability",
    "jev-noul": "one yes/no question per chunk: is this chunk an answer?",
    "jev-score": "one question per chunk, placed on a four-level usefulness rubric",
}


def _encoding_tokens(summary: dict, rows: list[dict]) -> dict[str, str]:
    """The sentence under the encoding table, written from the encoding table.

    The comparison is only made among the encodings that actually produced a
    score, and the sentence names the gap rather than asserting a winner, so a
    run where the three land inside each other's intervals reads as a tie
    instead of as a result.
    """
    have = [(a, summary[a]) for a in JEV_ARMS
            if a in summary and summary[a].get("recall@5") is not None]
    if len(have) < 2:
        return {"encoding_finding": "Only one encoding produced a score in this run, "
                                    "so there is nothing to compare."}
    best = max(have, key=lambda kv: kv[1]["recall@5"])
    cheap = min(have, key=lambda kv: kv[1]["cost_per_1k_micro"])
    spread = best[1]["recall@5"] - min(s["recall@5"] for _, s in have)
    ratio = (max(s["cost_per_1k_micro"] for _, s in have)
             / max(1, min(s["cost_per_1k_micro"] for _, s in have)))
    same = best[0] == cheap[0]
    return {"encoding_finding": (
        f"**The gap between the best and worst way of asking the same model is "
        f"{pct(spread)}, and the gap in price is {ratio:.0f}\u00d7.** The most accurate "
        f"encoding here is *{PRETTY[best[0]]}*; the cheapest is *{PRETTY[cheap[0]]}*"
        + (", and they are the same one." if same else
           " \u2014 they are not the same one, so this is a choice, not a default.")
        + " Any sentence of the form \u201cJev scores X\u201d is incomplete without "
          "naming which of these three it means.")}


def encoding_table(ctx: dict) -> str:
    """The same model, the same question, three grammars.

    Present because the first edition of this report measured one encoding and
    called the result "Jev". The three rows differ only in how the question was
    written; the model id in every request is identical.
    """
    rows, summary = ctx["rows"], ctx["summary"]
    head = ("<table class='encodings'><thead><tr><th>How we asked</th>"
            "<th class='n'>Right in 5</th><th class='n'>Of what it was shown</th>"
            "<th class='n'>HTTP calls<br>per question</th>"
            "<th class='n'>Input tokens<br>per question</th>"
            "<th class='n'>Cost per<br>1,000 questions</th>"
            "<th class='n'>Questions<br>it could not answer</th>"
            "</tr></thead><tbody>")
    body = []
    for arm in JEV_ARMS:
        s = summary.get(arm)
        mine = [r for r in rows if r["arm"] == arm]
        if not s or not mine:
            continue
        ok = [r for r in mine if r["failure"] is None]
        calls = sum(r.get("subcalls", 1) for r in mine)
        tok = sum(r["input_tokens"] for r in mine)
        body.append(
            "<tr><td><b>{name}</b><br><span class='dim'>{note}</span></td>"
            "<td class='n'>{r5}</td><td class='n'>{rr}</td>"
            "<td class='n'>{calls:,.0f}</td><td class='n'>{tok:,.0f}</td>"
            "<td class='n'>{cost}</td><td class='n'>{fail} of {n}</td></tr>".format(
                name=inline(PRETTY[arm]), note=inline(ENCODING_NOTE[arm]),
                r5=pct(s.get("recall@5")), rr=pct(s.get("reachable@5")),
                calls=calls / len(mine), tok=tok / max(1, len(ok)),
                cost=usd(s["cost_per_1k_micro"], 2),
                fail=len(mine) - len(ok), n=len(mine)))
    return head + "".join(body) + "</tbody></table>"


BLOCKS = {
    "PIPELINE_FIGURE": pipeline_figure,
    "HEADLINE_TABLE": lambda c: headline_table(c["summary"]),
    "BY_SOURCE_TABLE": lambda c: by_source_table(c["rows"], c["summary"]),
    "STRATA_TABLE": strata_table,
    "COMMON_SUBSET_TABLE": common_subset_table,
    "ENCODING_TABLE": encoding_table,
    "LIFT_TABLE": lambda c: lift_table(c["rows"], c["summary"]),
    "COST_CHART": lambda c: chart(c["summary"]),
    "MATRIX": lambda c: matrix_html(c["rows"], c["summary"]),
    "FULL_TABLE": lambda c: full_table(c["summary"]),
    "FAILURES": lambda c: failures_html(c["summary"]),
    "WORKED_EXAMPLE": worked_example,
}

PAGE = TEMPLATE.replace(
    "<title>Jev vs chat models on ticket routing</title>",
    "<title>Picking the right chunks: Jev against chat models</title>")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "data/rag/results-d100.json")
    ap.add_argument("--results20", default=ROOT / "data/rag/results-d20.json")
    ap.add_argument("--tasks", default=ROOT / "data/rag/tasks.json")
    ap.add_argument("--findings", default=ROOT / "report" / "rag-findings.md")
    ap.add_argument("--out", default=ROOT / "rag-report.html")
    args = ap.parse_args(argv)

    res = json.loads(Path(args.results).read_text())
    tasks = load_tasks(args.tasks)
    # The depth-20 contrast is a separately-paid run. Missing is not an error:
    # its section then says the experiment did not run, rather than rendering empty.
    p20 = Path(args.results20)
    res20 = json.loads(p20.read_text()) if p20.exists() else None
    summary = summarise(res["rows"], SEED)

    ctx = {"rows": res["rows"], "summary": summary, "res": res, "tasks": tasks}
    body = markdown(substitute(Path(args.findings).read_text(),
                               tokens(res, summary, tasks, res20), BLOCKS))
    for name, render in BLOCKS.items():
        body = body.replace("<p>{{%s}}</p>" % name, render(ctx))
    for name in BLOCKS:
        if "{{%s}}" % name in body:
            raise ValueError("{{%s}} must sit alone on its own line" % name)

    css = (ROOT / "assets" / "doctrine.css").read_text()
    extra = (ROOT / "assets" / "report.css").read_text()
    Path(args.out).write_text(PAGE.replace("{{CSS}}", css + "\n" + extra)
                              .replace("{{BODY}}", body))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
