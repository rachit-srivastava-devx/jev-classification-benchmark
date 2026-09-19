#!/usr/bin/env python3
"""Render report.html from results.json plus an authored findings file.

The numbers come from results.json. The prose comes from findings.md. Nothing
in this script invents a figure: every `{{token}}` it substitutes is read out
of the results file, and an unknown token raises rather than rendering blank.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevdemo.pricing import TOLERANCE_MICRO
from jevdemo.decision import NANO_PER_USD, break_even_table, distinguishability_matrix
from jevdemo.stats import Z_95, difference_interval, wilson_interval

ROOT = Path(__file__).resolve().parent.parent


def usd(micro: int | None, places: int = 4) -> str:
    return "—" if micro is None else f"${micro / 1_000_000:.{places}f}"


def tokens(results: dict, judge: dict | None = None) -> dict[str, str]:
    arms = results["arms"]
    by = {a["arm"]: a for a in arms}
    jev = by["jev"]
    chat = sorted((a for a in arms if a["kind"] == "chat"), key=lambda a: a["cost_per_1000_micro"])
    median = chat[len(chat) // 2]
    measured = _measured(results)
    best = max(measured, key=lambda a: (a["accuracy"] or 0))
    out = {
        "arm_count": str(len(arms)),
        "record_count": f"{results['totals']['records']:,}",
        "total_cost": usd(results["totals"]["computed_cost_micro"], 2),
        "jev_cost_1k": usd(jev["cost_per_1000_micro"], 2),
        "jev_acc": f"{jev['accuracy'] * 100:.0f}%",
        "jev_p50": f"{jev['p50_latency_ms']:.0f}",
        "median_chat_cost_1k": usd(median["cost_per_1000_micro"], 2),
        "median_chat_arm": median["arm"],
        "cost_ratio": f"{median['cost_per_1000_micro'] / jev['cost_per_1000_micro']:.0f}",
        "best_arm": best["arm"],
        "best_acc": f"{best['accuracy'] * 100:.0f}%",
    }
    jev = next((a for a in measured if a["kind"] == "jev"), None)
    dearest = max(measured, key=lambda a: a["cost_per_1000_micro"]) if measured else None
    out["jev_acc_cost"] = (
        "\u2014" if jev is None else usd(jev["cost_per_1000_micro"], 4)
    )
    # Ratio against Jev, not against the cheapest: Jev IS the cheapest here, and
    # writing it as "against the cheapest" would stop being true the moment it
    # is not.
    out["dearest_ratio"] = (
        "\u2014" if jev is None or dearest is None or not jev["cost_per_1000_micro"]
        else f"{dearest['cost_per_1000_micro'] / jev['cost_per_1000_micro']:,.0f}"
    )
    out["dearest_model"] = "\u2014" if dearest is None else dearest["arm"]
    out["unrecon_count"] = str(
        sum(1 for a in results["arms"] if not a.get("cost_reconciled", True))
    )
    out.update(_stat_tokens(results))
    out.update(_judge_tokens(judge))
    for a in arms:
        slug = a["arm"].replace(".", "_").replace("-", "_")
        out[f"{slug}_acc"] = "—" if a["accuracy"] is None else f"{a['accuracy'] * 100:.0f}%"
        out[f"{slug}_cost_1k"] = usd(a["cost_per_1000_micro"], 2)
        out[f"{slug}_p50"] = "—" if a["p50_latency_ms"] is None else f"{a['p50_latency_ms']:.0f}"
        out[f"{slug}_think"] = f"{a['reasoning_tokens']:,}"
    return out


def _dataset_n(results: dict) -> int:
    """The bulk-pass sample size. Every arm sees the same tickets, so the max is it."""
    return max((a["attempted"] for a in results["arms"]), default=0)


def _worst_case_halfwidth(n: int) -> float:
    """Half the widest Wilson interval at this n, which occurs at p = 1/2.

    This is the number the report quotes before showing any accuracy, so that no
    reader reaches a table without already knowing what it can and cannot resolve.
    """
    if n <= 0:
        return 0.0
    lo, hi = wilson_interval(n // 2, n)
    return (hi - lo) / 2


def _resolution_pp(results: dict) -> float:
    """The smallest accuracy gap this run can actually call, near the observed top.

    Found by search rather than by formula: the Newcombe interval's width depends
    on where on [0,1] the two proportions sit, so the honest answer is the gap that
    is resolvable *at the accuracies these arms actually reached*.
    """
    n = _dataset_n(results)
    if n < 2:
        return 0.0
    top = max((a["correct"] for a in results["arms"]), default=n)
    for gap in range(1, n + 1):
        lower = top - gap
        if lower < 0:
            break
        lo, _ = difference_interval(top, n, lower, n)
        if lo > 0:
            return 100 * gap / n
    return 100.0


def _stat_tokens(results: dict) -> dict[str, str]:
    arms = _measured(results)
    n = _dataset_n(results)
    reconciled = sum(1 for a in arms if a.get("cost_reconciled"))
    jev = next(a for a in arms if a["arm"] == "jev")
    rows = break_even_table(arms)
    baseline = next((r["arm"] for r in rows if r["is_baseline"]), "\u2014")
    priced = [r for r in rows if r["break_even_nano"]]
    cheapest_upgrade = min(priced, key=lambda r: r["break_even_nano"], default=None)
    pairs = [(a, b) for a in arms for b in arms if a["arm"] < b["arm"]]
    resolved = sum(
        1 for a, b in pairs
        if distinguishability_matrix(arms)[a["arm"]][b["arm"]]["resolved"]
    ) if len(arms) <= 40 else 0
    return {
        "dataset_n": str(n),
        "ci_halfwidth_pp": f"{_worst_case_halfwidth(n) * 100:.1f}",
        "resolution_pp": f"{_resolution_pp(results):.0f}",
        "z_95": f"{Z_95:.3f}",
        "recon_arms": f"{reconciled}/{len(arms)}",
        "recon_tolerance": str(TOLERANCE_MICRO),
        "jev_price_in": f"{jev['price_in_micro_per_mtok'] / 1_000_000:.3f}",
        "jev_price_out": f"{jev['price_out_micro_per_mtok'] / 1_000_000:.3f}",
        "jev_recon": "exact" if jev.get("cost_reconciled") else "not reconciled",
        "baseline_arm": baseline,
        "pair_count": str(len(pairs)),
        "resolved_pairs": str(resolved),
        "unresolved_pairs": str(len(pairs) - resolved),
        "cheapest_upgrade_arm": "\u2014" if cheapest_upgrade is None else cheapest_upgrade["arm"],
        "cheapest_upgrade_usd": (
            "\u2014" if cheapest_upgrade is None
            else f"${cheapest_upgrade['break_even_usd']:,.2f}"
        ),
    }


def _judge_tokens(judge: dict | None) -> dict[str, str]:
    """Absent judge data renders as an explicit dash, never as a blank or a zero."""
    if judge is None:
        return {k: "\u2014" for k in (
            "judge_model", "judge_rows", "judge_agreement", "judge_rho", "judge_cost")}
    rho = judge["rank_correlation_spearman"]
    return {
        "judge_model": judge["judge_model"],
        "judge_rows": f"{judge['rows_judged']:,}",
        "judge_agreement": f"{judge['overall_agreement_rate'] * 100:.1f}%",
        "judge_rho": "\u2014" if rho is None else f"{rho:.2f}",
        "judge_cost": usd(judge["judge_cost_micro"], 2),
    }


def substitute(text: str, table: dict[str, str]) -> str:
    def swap(match: re.Match) -> str:
        key = match.group(1)
        if key in BLOCKS:
            return match.group(0)
        if key not in table:
            raise KeyError(f"findings.md references unknown token {{{{{key}}}}}")
        return table[key]

    return re.sub(r"\{\{(\w+)\}\}", swap, text)


def inline(text: str) -> str:
    # Code spans are pulled out before emphasis runs. `K*` is a real symbol in the
    # proofs, and leaving its asterisk in the stream opened an emphasis run that
    # swallowed a sentence and a half of the corollaries into accent italic.
    text = html.escape(text)
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(match.group(1))
        return "\x00%d\x00" % (len(spans) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: "<code>%s</code>" % spans[int(m.group(1))], text)


LIST_MARKER = re.compile(r"^(?:-|\d+\.)\s")


def markdown(src: str) -> str:
    """The small subset the findings file uses. Anything else is a bug, not a feature."""
    out: list[str] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("EYEBROW "):
            out.append(f'<p class="eyebrow">{inline(line[8:])}</p>')
            i += 1
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
            i += 1
        elif line.startswith("## "):
            out.append(f'<h2>{inline(line[3:])}</h2><div class="accent-rule"></div>')
            i += 1
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
            i += 1
        elif line.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:])
                i += 1
            out.append(f'<blockquote>{inline(" ".join(block))}</blockquote>')
        elif line.startswith("    ") and line.strip():
            # An indented block is a formula, not prose. Escaped, never inlined:
            # the algebra must survive verbatim and nothing in it is markup.
            block = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                if lines[i].strip():
                    block.append(lines[i][4:])
                elif block:
                    block.append("")
                i += 1
            while block and not block[-1]:
                block.pop()
            out.append("<pre>" + html.escape("\n".join(block)) + "</pre>")
        elif LIST_MARKER.match(line.lstrip()):
            tag = "ul" if line.lstrip().startswith("- ") else "ol"
            items: list[list[str]] = []
            while i < len(lines) and lines[i].strip():
                stripped = lines[i].lstrip()
                if LIST_MARKER.match(stripped):
                    items.append([stripped.split(" ", 1)[1].strip()])
                elif items:
                    # A wrapped line belongs to the item above it. Without this the
                    # loop stopped at the first continuation and every later item
                    # reflowed into a paragraph, deleting the numbering.
                    items[-1].append(stripped)
                else:
                    break
                i += 1
            body = "".join(f"<li>{inline(' '.join(it))}</li>" for it in items)
            out.append(f"<{tag}>{body}</{tag}>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(
                    ("#", "- ", "> ", "EYEBROW ", "    ")):
                para.append(lines[i])
                i += 1
            out.append(f'<p>{inline(" ".join(para))}</p>')
    return "\n".join(out)


def results_table(results: dict) -> str:
    head = ("<tr><th>Model</th><th>Reasoning</th><th class='n'>Accuracy</th>"
            "<th class='n'>Cost / 1k</th><th class='n'>P50 ms</th><th class='n'>P95 ms</th>"
            "<th class='n'>In</th><th class='n'>Out</th><th class='n'>Thinking</th>"
            "<th class='n'>Failures</th></tr>")
    body = []
    for a in results["arms"]:
        reasoning = "n/a" if a["kind"] == "jev" else (
            "default" if a["reasoning"] is None else json.dumps(a["reasoning"]))
        fails = sum(a["failures"].values())
        acc = "\u2014" if a["accuracy"] is None else f"{a['accuracy'] * 100:.0f}%"
        p50 = "\u2014" if a["p50_latency_ms"] is None else f"{a['p50_latency_ms']:.0f}"
        p95 = "\u2014" if a["p95_latency_ms"] is None else f"{a['p95_latency_ms']:.0f}"
        cells = [
            f"<code>{html.escape(a['arm'])}</code>",
            f"<span class='sub'>{html.escape(reasoning)}</span>",
            acc, usd(a["cost_per_1000_micro"], 2), p50, p95,
            f"{a['input_tokens']:,}", f"{a['output_tokens']:,}",
            f"{a['reasoning_tokens']:,}", str(fails) if fails else "\u2014",
        ]
        row = "".join(
            f"<td class='n'>{c}</td>" if i >= 2 else f"<td>{c}</td>"
            for i, c in enumerate(cells)
        )
        body.append(f"<tr>{row}</tr>")
    return f"<table class='results'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def _measured(results: dict) -> list[dict]:
    """Arms this run actually observed.

    An arm that produced no usable answer contributes nothing to an accuracy
    comparison and would corrupt one: 0 correct out of 100 attempts is
    resolvable against everything, so leaving it in would manufacture twenty
    significant differences out of a routing failure.
    """
    return [a for a in results["arms"] if a.get("measured", a["accuracy"] is not None)]


def _unmeasured_note(results: dict) -> str:
    missing = [a for a in results["arms"]
               if not a.get("measured", a["accuracy"] is not None)]
    if not missing:
        return ""
    names = ", ".join(f"<code>{html.escape(a['arm'])}</code>" for a in missing)
    return (f"<p class='sub'>Excluded as unmeasured: {names}. These models returned no "
            f"usable answer on any call; see the failure table for why.</p>")




def _tickets() -> list[dict]:
    """The dataset the run used, or an empty list if it is not beside the code.

    Empty rather than raising: a missing dataset must show as an empty table the
    reader can see, not as a report that silently omits the section.
    """
    path = ROOT / "data" / "tickets.json"
    return json.loads(path.read_text()) if path.exists() else []


def category_table() -> str:
    """The five queues, rendered from the definitions the models were given.

    Read from `jevdemo.labels`, never retyped. A definition restated in the
    report is one that can quietly drift from the one under test.
    """
    from jevdemo.labels import LABELS
    head = "<tr><th>Queue</th><th>Definition given to every model</th></tr>"
    body = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td>"
        f"<td>{html.escape(definition)}</td></tr>"
        for name, definition in LABELS.items()
    )
    return f"<table class='results categories'><thead>{head}</thead><tbody>{body}</tbody></table>"


def examples_table(tickets: list[dict]) -> str:
    """One real ticket per category, quoted verbatim from the dataset.

    First occurrence in file order, not a chosen one: picking the clearest
    example of each category would describe the report's taste rather than the
    data the models actually saw.
    """
    if not tickets:
        return "<p class='sub'>No tickets were available to quote.</p>"
    seen: dict[str, dict] = {}
    for t in tickets:
        seen.setdefault(t["label"], t)
    head = "<tr><th>Hand label</th><th>The ticket, as sent</th></tr>"
    body = "".join(
        f"<tr><td><code>{html.escape(label)}</code></td>"
        f"<td>{html.escape(t['text'])}</td></tr>"
        for label, t in seen.items()
    )
    return f"<table class='results examples'><thead>{head}</thead><tbody>{body}</tbody></table>"


def headline_table(results: dict) -> str:
    """The table a reader meets first: accurate, what it costs, how fast.

    Priced as a multiple of the cheapest model as well as in dollars, because the
    spread here is three orders of magnitude and "132 times" is legible where
    "$2.44 against $0.0185" is not. The full ten-column table is in the appendix;
    this one is deliberately not it.
    """
    measured = sorted(_measured(results), key=lambda a: a["cost_per_1000_micro"])
    cheapest = measured[0]["cost_per_1000_micro"] if measured else 0
    head = ("<tr><th>Model</th><th class='n'>Accuracy</th>"
            "<th class='n'>Cost per 1,000 tickets</th><th class='n'>vs cheapest</th>"
            "<th class='n'>Typical speed</th></tr>")
    body = []
    for a in measured:
        acc = "\u2014" if a["accuracy"] is None else f"{a['accuracy'] * 100:.0f}%"
        p50 = "\u2014" if a["p50_latency_ms"] is None else f"{a['p50_latency_ms']:,.0f} ms"
        if not cheapest:
            mult = "\u2014"
        else:
            ratio = a["cost_per_1000_micro"] / cheapest
            mult = "1\u00d7" if ratio < 1.05 else f"{ratio:,.0f}\u00d7"
        body.append(
            f"<tr><td><code>{html.escape(a['arm'])}</code></td>"
            f"<td class='n'>{acc}</td>"
            f"<td class='n'>{usd(a['cost_per_1000_micro'], 4)}</td>"
            f"<td class='n'>{mult}</td><td class='n'>{p50}</td></tr>"
        )
    return (f"<table class='results headline'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


#: Chart geometry, in SVG user units. The viewBox is the document's content width
#: so the chart is never scaled up and its type never disagrees with the prose.
CHART_W, CHART_H = 760, 400
# WeasyPrint's CSS engine does not implement `fill`, so a chart painted only by
# stylesheet prints in undifferentiated black. Every SVG element therefore carries
# its own presentation attributes; the classes remain for the screen.
ACCENT, INK, INK_2, MUTED, MUTED_2 = "#1E6FFF", "#0A0A0A", "#1A1A1A", "#5C6066", "#8A8F96"
RULE, RULE_2 = "#E5E5E5", "#F0F0F0"
MONO = "JetBrains Mono, ui-monospace, monospace"
PAINT = {
    "grid": f"stroke='{RULE_2}' stroke-width='1'",
    "axis": f"stroke='{INK}' stroke-width='1'",
    "leader": f"stroke='{RULE}' stroke-width='1'",
    "ax": f"fill='{MUTED_2}' font-family='{MONO}' font-size='10'",
    "ax-title": (f"fill='{MUTED}' font-family='{MONO}' font-size='10' "
                 "letter-spacing='1.2'"),
    "pt": f"fill='{MUTED}'",
    "pt jev": f"fill='{ACCENT}'",
    "pt-label": f"fill='{INK_2}' font-family='{MONO}' font-size='10'",
    "pt-label jev": f"fill='{ACCENT}' font-family='{MONO}' font-size='10' font-weight='500'",
}
CHART_PAD = {"l": 52, "r": 150, "t": 20, "b": 48}
#: Minimum vertical gap between two point labels before one is nudged away.
LABEL_GAP = 11.0

NO_CHART = ("<p class='sub'>No model produced a measurable result, so there is "
            "nothing to plot.</p>")


def _declutter(points: list[dict]) -> None:
    """Nudge colliding labels apart, top to bottom.

    Nineteen models sit inside an eighteen-point accuracy band, so drawn at their
    true y the labels overlap into an unreadable stack. The dot stays at the
    measurement; only its label moves, which is why each one keeps a leader line.
    """
    # Sorted once. Re-sorting inside the loop reads a list that the previous
    # iteration has already moved, so a label could be compared against itself.
    ordered = sorted(points, key=lambda q: q["ly"])
    for prev, cur in zip(ordered, ordered[1:]):
        if cur["ly"] - prev["ly"] < LABEL_GAP:
            cur["ly"] = prev["ly"] + LABEL_GAP


def cost_effectiveness_chart(results: dict) -> str:
    """Accuracy against cost, one dot per model, cost on a logarithmic axis.

    Cost spans three orders of magnitude across this roster. On a linear axis
    eleven of nineteen models would land inside the first tenth of the width, so
    the axis is log10 and every decade is ruled. Drawn from results.json, so the
    chart cannot disagree with the table beside it.
    """
    measured = [a for a in _measured(results) if a["cost_per_1000_micro"]]
    if not measured:
        return NO_CHART

    x0, x1 = CHART_PAD["l"], CHART_W - CHART_PAD["r"]
    y0, y1 = CHART_H - CHART_PAD["b"], CHART_PAD["t"]

    costs = [a["cost_per_1000_micro"] / 1_000_000 for a in measured]
    lo, hi = math.log10(min(costs)), math.log10(max(costs))
    if hi - lo < 0.5:          # a single model, or a roster with no spread
        lo, hi = lo - 0.5, hi + 0.5

    accs = [a["accuracy"] for a in measured]
    a_lo = max(0.0, math.floor(min(accs) * 20 - 1) / 20)
    a_hi = min(1.0, math.ceil(max(accs) * 20 + 1) / 20)
    if a_hi - a_lo < 0.02:
        a_lo, a_hi = max(0.0, a_lo - 0.05), min(1.0, a_hi + 0.05)

    def px(cost_usd: float) -> float:
        return x0 + (math.log10(cost_usd) - lo) / (hi - lo) * (x1 - x0)

    def py(acc: float) -> float:
        return y0 - (acc - a_lo) / (a_hi - a_lo) * (y0 - y1)

    parts = [
        f"<svg viewBox='0 0 {CHART_W} {CHART_H}' class='chart' "
        f"xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='Accuracy against cost per 1,000 tickets'>"
    ]

    # Horizontal rules at each 5-point accuracy step: the reader compares heights,
    # so the gridlines run along the axis being compared and nowhere else.
    step = 0.05
    tick = math.ceil(a_lo / step) * step
    while tick <= a_hi + 1e-9:
        y = py(tick)
        parts.append(f"<line class='grid' {PAINT['grid']} x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}'/>")
        parts.append(f"<text class='ax' {PAINT['ax']} x='{x0 - 8}' y='{y + 3:.1f}' "
                     f"text-anchor='end'>{tick * 100:.0f}%</text>")
        tick += step

    decade = math.floor(lo)
    while decade <= math.ceil(hi):
        value = 10 ** decade
        if lo <= decade <= hi:
            x = px(value)
            parts.append(f"<line class='grid' {PAINT['grid']} x1='{x:.1f}' y1='{y0}' x2='{x:.1f}' y2='{y1}'/>")
            label = f"${value:,.2f}" if value >= 0.01 else f"${value:.3f}"
            parts.append(f"<text class='ax' {PAINT['ax']} x='{x:.1f}' y='{y0 + 18}' "
                         f"text-anchor='middle'>{label}</text>")
        decade += 1

    parts.append(f"<line class='axis' {PAINT['axis']} x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}'/>")
    parts.append(f"<text class='ax-title' {PAINT['ax-title']} x='{(x0 + x1) / 2:.0f}' y='{CHART_H - 8}' "
                 f"text-anchor='middle'>Cost per 1,000 tickets</text>")
    parts.append(f"<text class='ax-title' {PAINT['ax-title']} x='-{(y0 + y1) / 2:.0f}' y='14' "
                 f"text-anchor='middle' transform='rotate(-90)'>Accuracy</text>")

    points = [
        {"name": a["arm"],
         "x": px(a["cost_per_1000_micro"] / 1_000_000),
         "y": py(a["accuracy"]),
         "ly": py(a["accuracy"]),
         "jev": a["kind"] == "jev"}
        for a in measured
    ]
    _declutter(points)

    for p in points:
        cls = "pt jev" if p["jev"] else "pt"
        paint = PAINT[cls]
        parts.append(f"<line class='leader' {PAINT['leader']} x1='{p['x']:.1f}' y1='{p['y']:.1f}' "
                     f"x2='{x1 + 6}' y2='{p['ly']:.1f}'/>")
        parts.append(f"<circle class='{cls}' {paint} cx='{p['x']:.1f}' cy='{p['y']:.1f}' r='4'/>")
        lab_cls = "pt-label jev" if p["jev"] else "pt-label"
        lab_paint = PAINT[lab_cls]
        parts.append(f"<text class='{lab_cls}' {lab_paint} x='{x1 + 10}' y='{p['ly'] + 3:.1f}'>"
                     f"{html.escape(p['name'])}</text>")

    parts.append("</svg>")
    return "".join(parts)


def break_even_table_html(results: dict) -> str:
    """What one misroute must be worth before each arm's price is rational.

    Read it as a question put to the reader: if a wrong route costs your business
    more than the figure in this row, this arm is worth its price. If less, it is
    not. The benchmark cannot supply that figure; only the business can.
    """
    rows = break_even_table(_measured(results))
    head = ("<tr><th>Model</th><th class='n'>Cost / 1k</th><th class='n'>Accuracy</th>"
            "<th class='n'>Break-even cost of one misroute</th><th>Reading</th></tr>")
    by = {a["arm"]: a for a in _measured(results)}
    body = []
    for r in rows:
        a = by[r["arm"]]
        acc = "\u2014" if a["accuracy"] is None else f"{a['accuracy'] * 100:.0f}%"
        if r["is_baseline"]:
            threshold, note = "baseline", "the cheapest model; every other row is priced against it"
        elif r["break_even_nano"] is None:
            threshold, note = "\u2014", r["note"]
        else:
            threshold = f"${r['break_even_usd']:,.2f}"
            note = r["note"] or "worth its price above this figure"
        cls = "" if r["resolved"] or r["is_baseline"] else " class='unresolved'"
        body.append(
            f"<tr{cls}><td><code>{html.escape(r['arm'])}</code></td>"
            f"<td class='n'>{usd(a['cost_per_1000_micro'], 2)}</td>"
            f"<td class='n'>{acc}</td><td class='n'>{threshold}</td>"
            f"<td><span class='sub'>{html.escape(note)}</span></td></tr>"
        )
    return (f"<table class='results breakeven'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>{_unmeasured_note(results)}")


#: A tie is a statement about the sample, not about the models. The glyphs are
#: deliberately plain: the matrix is a lookup, and decoration in a lookup is noise.
VERDICT_GLYPH = {"better": "+", "worse": "\u2212", "tie": "=", "self": "\u00b7"}


def distinguishability_html(results: dict) -> str:
    """Every ordered pair of arms, and whether this run can tell them apart."""
    arms = sorted(_measured(results), key=lambda a: -(a["accuracy"] or 0))
    names = [a["arm"] for a in arms]
    matrix = distinguishability_matrix(_measured(results))
    # Columns are numbered, not named. Twenty arm names across the head is ~880px of
    # table in a 810px column, and a matrix that scrolls sideways is a matrix nobody
    # reads. The row label carries the name and the index ties the two together.
    index = {n: i + 1 for i, n in enumerate(names)}
    head = "<tr><th>Row beats column?</th>" + "".join(
        f"<th class='n rot' title='{html.escape(n)}'>{index[n]}</th>" for n in names) + "</tr>"
    body = []
    for a in arms:
        cells = []
        for n in names:
            cell = matrix[a["arm"]][n]
            lo, hi = cell["interval"]
            title = f"{cell['verdict']}: difference 95% CI [{lo:+.3f}, {hi:+.3f}]"
            cells.append(
                f"<td class='n v-{cell['verdict']}' title='{html.escape(title)}'>"
                f"{VERDICT_GLYPH[cell['verdict']]}</td>"
            )
        body.append(
            f"<tr><td><span class='idx'>{index[a['arm']]}</span> "
            f"<code>{html.escape(a['arm'])}</code></td>{''.join(cells)}</tr>")
    legend = ("<p class='sub'>Columns are numbered in the same order as the rows &#183; "
              "+ row is measurably better &#183; \u2212 row is measurably worse "
              "&#183; = this run cannot tell them apart. Hover any cell for the 95% interval "
              "on the difference (Newcombe method 10).</p>")
    return (f"<table class='results matrix'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>{legend}"
            f"{_unmeasured_note(results)}")


def failures_html(results: dict) -> str:
    """Every call that did not produce a usable label, by arm and by kind.

    Failures are counted into the denominator of accuracy, so this table is the
    audit trail for why an arm's accuracy is lower than its correct-answer rate.
    """
    kinds = sorted({k for a in results["arms"] for k in a["failures"]})
    totals = results.get("totals", {})
    if not kinds:
        return ("<p>No call in this run failed to produce a parseable label. "
                f"Rate-limit retries: {totals.get('rate_limit_retries', 0)}.</p>")
    head = ("<tr><th>Model</th>" + "".join(f"<th class='n'>{html.escape(k)}</th>" for k in kinds)
            + "<th class='n'>Total</th></tr>")
    body = []
    for a in results["arms"]:
        total = sum(a["failures"].values())
        if total == 0:
            continue
        cells = "".join(
            f"<td class='n'>{a['failures'].get(k) or chr(8212)}</td>" for k in kinds)
        body.append(f"<tr><td><code>{html.escape(a['arm'])}</code></td>{cells}"
                    f"<td class='n'>{total}</td></tr>")
    note = (f"<p class='sub'>rate_limit_retries across the whole run: "
            f"{totals.get('rate_limit_retries', 0)}. A retry is not a failure \u2014 it is a "
            f"call that succeeded on a later attempt, and its latency is excluded.</p>")
    return (f"<table class='results failures'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>{note}")


def judge_html(judge: dict | None) -> str:
    """The label-free evaluation experiment, or an honest note that it was not run.

    Rendering an empty table here would read as a null result. It is not one.
    """
    if judge is None:
        return ("<p><strong>This experiment was not run for this build.</strong> The judge pass "
                "is a separate, separately-paid pass over the stored predictions; re-run "
                "<code>scripts/judge_run.py</code> and rebuild to populate this section.</p>")
    # Both accuracy columns are computed over judged rows only: a call that failed
    # left no prediction to grade. A model that lost calls therefore reads higher
    # here than in the measurement table, so the denominator is a column of its own.
    head = ("<tr><th>Model</th><th class='n'>Judged rows</th>"
            "<th class='n'>Gold accuracy<br>over judged rows</th>"
            "<th class='n'>Judge accuracy</th>"
            "<th class='n'>Judge 95% CI</th><th class='n'>Row agreement</th>"
            "<th class='n'>Mean P(correct)</th><th>Note</th></tr>")
    body = []
    for a in judge["arms"]:
        lo, hi = a["judge_accuracy_ci95"]
        note = "self-judged \u2014 excluded from the overall agreement figure" if a["self_judged"] else ""
        body.append(
            f"<tr><td><code>{html.escape(a['arm'])}</code></td>"
            f"<td class='n'>{a['rows']:,}</td>"
            f"<td class='n'>{a['gold_accuracy'] * 100:.0f}%</td>"
            f"<td class='n'>{a['judge_accuracy'] * 100:.0f}%</td>"
            f"<td class='n'>[{lo * 100:.0f}, {hi * 100:.0f}]</td>"
            f"<td class='n'>{a['agreement_rate'] * 100:.1f}%</td>"
            f"<td class='n'>{a['mean_probability']:.2f}</td>"
            f"<td><span class='sub'>{html.escape(note)}</span></td></tr>"
        )
    rho = judge["rank_correlation_spearman"]
    rho_text = ("not computable \u2014 fewer than two models carry a judged score"
                if rho is None else f"{rho:.2f}")
    foot = (
        f"<p class='sub'>Judge: <code>{html.escape(judge['judge_model'])}</code> &#183; "
        f"{judge['rows_judged']:,} rows judged, "
        f"{judge['rows_skipped_arm_failure']:,} skipped because the model returned no "
        f"prediction to grade &#183; overall agreement with the gold labels "
        f"{judge['overall_agreement_matched']:,}/{judge['overall_agreement_scored']:,} = "
        f"{judge['overall_agreement_rate'] * 100:.1f}% &#183; Spearman rank correlation between "
        f"the judge's model ranking and the gold ranking: {rho_text} &#183; cost of judgement "
        f"{usd(judge['judge_cost_micro'], 2)}.</p>"
    )
    return (f"<table class='results judge'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>{foot}")


#: An uppercase token is a block: it renders a table and must sit alone on its own
#: line. A lowercase token is an inline value. `substitute` passes blocks through
#: untouched so they survive the markdown render and are swapped afterwards.
BLOCKS = {
    "CATEGORY_TABLE": lambda r, j: category_table(),
    "EXAMPLES_TABLE": lambda r, j: examples_table(_tickets()),
    "COST_CHART": lambda r, j: cost_effectiveness_chart(r),
    "HEADLINE_TABLE": lambda r, j: headline_table(r),
    "RESULTS_TABLE": lambda r, j: results_table(r),
    "BREAK_EVEN_TABLE": lambda r, j: break_even_table_html(r),
    "DISTINGUISHABILITY_MATRIX": lambda r, j: distinguishability_html(r),
    "FAILURES_TABLE": lambda r, j: failures_html(r),
    "JUDGE_TABLE": lambda r, j: judge_html(j),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results.json")
    ap.add_argument("--findings", default=ROOT / "report" / "findings.md")
    ap.add_argument("--judge", default=ROOT / "judge.json")
    ap.add_argument("--out", default=ROOT / "report.html")
    args = ap.parse_args(argv)

    results = json.loads(Path(args.results).read_text())
    # A missing judge file is not an error: the eval experiment is a separate,
    # separately-paid pass. Its section then says so rather than rendering empty.
    judge_path = Path(args.judge)
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else None
    src = substitute(Path(args.findings).read_text(), tokens(results, judge))
    # The marker survives markdown() as its own paragraph, then swaps for the table. Doing it
    # after the render keeps the table's markup out of the inline escaper.
    body = markdown(src)
    for name, render in BLOCKS.items():
        body = body.replace("<p>{{%s}}</p>" % name, render(results, judge))
    for name in BLOCKS:
        if "{{%s}}" % name in body:
            raise ValueError("{{%s}} must sit alone on its own line" % name)

    css = (ROOT / "assets" / "doctrine.css").read_text()
    extra = (ROOT / "assets" / "report.css").read_text()
    page = TEMPLATE.replace("{{CSS}}", css + "\n" + extra).replace("{{BODY}}", body)
    Path(args.out).write_text(page)
    print(f"wrote {args.out}")
    return 0


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Jev vs chat models on ticket routing</title>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>{{CSS}}</style>
</head>
<body>
<main>
{{BODY}}
</main>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
