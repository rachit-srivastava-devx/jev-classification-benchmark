"""Render the reranking report. Every number comes from the results file.

Same contract as `build_report.py`, and it reuses that script's markdown
renderer and token substitution rather than growing a second one: a `{{token}}`
the prose references but this script does not produce is a hard KeyError, so a
figure cannot drift out of date by being typed into a sentence.

    python scripts/build_rag_report.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    "jev": "Jev (TypeSafe)",
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
        "corpus_size": f"{tasks.get('corpus_total', 218052):,}",
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
    }
    out.update(_lift_tokens(summary, rows))
    out.update(_cost_tokens(ranked, jev))
    out.update(_depth_tokens(res20, summary))
    out.update(_failure_tokens(summary, rows))
    out["chart_span"] = chart_span(summary)
    return out


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
    for arm in summary:
        if arm == "bm25-baseline":
            continue
        xs, ys = align(rows, arm, "bm25-baseline", "recall@5")
        interval = paired_difference(xs, ys, rng) if xs else None
        slug = arm.replace(".", "_").replace("-", "_")
        out[f"{slug}_lift"] = pct(None if not interval else (sum(xs) - sum(ys)) / len(xs))
        out[f"{slug}_lift_lo"] = pct(interval[0]) if interval else "—"
        out[f"{slug}_lift_hi"] = pct(interval[1]) if interval else "—"
        out[f"{slug}_verdict"] = verdict(interval)
        if verdict(interval) == "better":
            beat += 1
    n_models = len(summary) - 1
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
            f"So the reranking step is worth paying for, but *which* model does it is "
            f"not something this test can tell apart. What separates them is price.")
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
            f"other {n_models - beat} cost money and changed nothing we can prove.")
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
    starved = sorted({r["arm"] for r in rows
                      if "402" in (r.get("detail") or "")})
    if starved:
        names = ", ".join(PRETTY.get(a, a) for a in starved)
        credit = (f"**Some failures were ours, not the model's.** {names} hit "
                  f"HTTP 402 — our own prepaid balance had too little headroom left "
                  f"for the provider to accept the request. Those calls are counted "
                  f"as failures for honesty, but they say nothing about the model.")
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
                f"<td class='n'>{usd(s['cost_per_1k_micro'], 2)}</td>"
                f"<td class='n'>{ms(s['p50_ms'], ' ms')}</td></tr>")
            continue
        body.append(
            f"<tr><td>{inline(PRETTY[s['arm']])}</td>"
            f"<td class='n'>{pct(s['recall@5'])}</td>"
            f"<td class='n sub'>{pct(ci[0]) if ci else '—'} to {pct(ci[1]) if ci else '—'}</td>"
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


# --- assembly ---------------------------------------------------------------

BLOCKS = {
    "HEADLINE_TABLE": lambda c: headline_table(c["summary"]),
    "BY_SOURCE_TABLE": lambda c: by_source_table(c["rows"], c["summary"]),
    "LIFT_TABLE": lambda c: lift_table(c["rows"], c["summary"]),
    "COST_CHART": lambda c: chart(c["summary"]),
    "MATRIX": lambda c: matrix_html(c["rows"], c["summary"]),
    "FULL_TABLE": lambda c: full_table(c["summary"]),
    "FAILURES": lambda c: failures_html(c["summary"]),
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
