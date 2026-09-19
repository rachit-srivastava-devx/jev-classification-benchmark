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
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def usd(micro: int | None, places: int = 4) -> str:
    return "—" if micro is None else f"${micro / 1_000_000:.{places}f}"


def tokens(results: dict) -> dict[str, str]:
    arms = results["arms"]
    by = {a["arm"]: a for a in arms}
    jev = by["jev"]
    chat = sorted((a for a in arms if a["kind"] == "chat"), key=lambda a: a["cost_per_1000_micro"])
    median = chat[len(chat) // 2]
    best = max(arms, key=lambda a: (a["accuracy"] or 0))
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
    for a in arms:
        slug = a["arm"].replace(".", "_").replace("-", "_")
        out[f"{slug}_acc"] = "—" if a["accuracy"] is None else f"{a['accuracy'] * 100:.0f}%"
        out[f"{slug}_cost_1k"] = usd(a["cost_per_1000_micro"], 2)
        out[f"{slug}_p50"] = "—" if a["p50_latency_ms"] is None else f"{a['p50_latency_ms']:.0f}"
        out[f"{slug}_think"] = f"{a['reasoning_tokens']:,}"
    return out


def substitute(text: str, table: dict[str, str]) -> str:
    def swap(match: re.Match) -> str:
        key = match.group(1)
        if key == "RESULTS_TABLE":
            return match.group(0)
        if key not in table:
            raise KeyError(f"findings.md references unknown token {{{{{key}}}}}")
        return table[key]

    return re.sub(r"\{\{(\w+)\}\}", swap, text)


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


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
        elif line.lstrip().startswith(("- ", "1. ")):
            tag = "ul" if line.lstrip().startswith("- ") else "ol"
            items = []
            while i < len(lines) and lines[i].lstrip().startswith(("- ", "1. ", "2. ", "3. ",
                                                                   "4. ", "5. ")):
                items.append(f"<li>{inline(lines[i].lstrip()[2:].lstrip())}</li>")
                i += 1
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(
                    ("#", "- ", "> ", "EYEBROW ")):
                para.append(lines[i])
                i += 1
            out.append(f'<p>{inline(" ".join(para))}</p>')
    return "\n".join(out)


def results_table(results: dict) -> str:
    head = ("<tr><th>Arm</th><th>Reasoning</th><th class='n'>Accuracy</th>"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results.json")
    ap.add_argument("--findings", default=ROOT / "report" / "findings.md")
    ap.add_argument("--out", default=ROOT / "report.html")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    src = substitute(Path(args.findings).read_text(), tokens(results))
    # The marker survives markdown() as its own paragraph, then swaps for the table. Doing it
    # after the render keeps the table's markup out of the inline escaper.
    body = markdown(src).replace("<p>{{RESULTS_TABLE}}</p>", results_table(results))
    if "{{RESULTS_TABLE}}" in body:
        raise ValueError("{{RESULTS_TABLE}} must sit alone on its own line")

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
