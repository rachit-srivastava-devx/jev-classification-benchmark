"""Re-score every bulk prediction with a label-free judge.

This is the Instant Evals experiment described in specs/06-report-structure.md.
It reads the predictions the benchmark already paid for, shows a judge the ticket
and the routed department but never the gold label, and asks whether the routing
was correct.

The question it answers: would a team with no labelled data have ranked these
arms the way the gold labels rank them?

    python3 scripts/judge_run.py --in results.json --out judge.json

Costs roughly $1.40 against claude-sonnet-5 at 2,000 rows. `--limit` caps the row
count for a cheap dry run; the full set is the default because a per-arm sample
smaller than the gold set would make the rank comparison noisier than the thing
it is measuring.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevdemo import judge  # noqa: E402
from jevdemo.dataset import load  # noqa: E402
from jevdemo.metrics import BULK_PASS  # noqa: E402
from jevdemo.runner import WORKERS, headers_for, load_key  # noqa: E402
from jevdemo.stats import spearman, wilson_interval  # noqa: E402
from jevdemo.transport import HttpTransport, Throttled  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.json"


def judge_one(row, ticket_text, model, transport, headers):
    url, body = judge.build_request(model, ticket_text, row["predicted"])
    status, raw, _ = transport(url, body, headers)
    return row, judge.parse(raw, status)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="results.json")
    ap.add_argument("--out", default="judge.json")
    ap.add_argument("--model", default=judge.DEFAULT_JUDGE)
    ap.add_argument("--limit", type=int, default=0, help="0 means every row")
    args = ap.parse_args(argv)

    key = load_key()
    if not key:
        print("no OPENROUTER_API_KEY", file=sys.stderr)
        return 2

    payload = json.loads(Path(args.src).read_text())
    if "records" not in payload:
        print(f"{args.src} has no per-record dump; re-run the benchmark", file=sys.stderr)
        return 2

    tickets = {t["id"]: t["text"] for t in load(DATA)}

    # Only bulk-pass rows the arm actually answered. A row with no predicted label
    # is an arm failure and there is nothing for a judge to grade.
    rows = [
        r for r in payload["records"]
        if r["pass"] == BULK_PASS and r["predicted"] is not None
    ]
    skipped_failures = sum(
        1 for r in payload["records"]
        if r["pass"] == BULK_PASS and r["predicted"] is None
    )
    if args.limit:
        rows = rows[: args.limit]

    print(f"judging {len(rows)} rows with {args.model} "
          f"({skipped_failures} arm failures have no prediction to grade)")

    transport = Throttled(HttpTransport())
    headers = headers_for(key)
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(judge_one, r, tickets[r["ticket_id"]], args.model, transport, headers)
            for r in rows
        ]
        for i, f in enumerate(futures, 1):
            results.append(f.result())
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}")

    # Per-arm tallies. `gold_correct` is the ground truth the judge never saw.
    per_arm = defaultdict(lambda: {"verdicts": [], "gold": [], "probs": []})
    cost_micro = 0
    judge_failures: dict[str, int] = defaultdict(int)
    for row, verdict in results:
        if verdict.reported_cost_micro:
            cost_micro += verdict.reported_cost_micro
        if verdict.failure:
            judge_failures[verdict.failure] += 1
        bucket = per_arm[row["arm"]]
        bucket["verdicts"].append(verdict.correct)
        bucket["gold"].append(row["predicted"] == row["gold"])
        if verdict.probability is not None:
            bucket["probs"].append(verdict.probability)

    arms = []
    for name, b in sorted(per_arm.items()):
        scored, agreed = judge.agreement(b["verdicts"], b["gold"])
        judged_correct = sum(1 for v in b["verdicts"] if v is True)
        gold_correct = sum(1 for g in b["gold"] if g)
        n = len(b["gold"])
        arms.append({
            "arm": name,
            "rows": n,
            "gold_accuracy": gold_correct / n if n else None,
            "judge_accuracy": judged_correct / n if n else None,
            "judge_accuracy_ci95": list(wilson_interval(judged_correct, n)),
            "agreement_scored": scored,
            "agreement_matched": agreed,
            "agreement_rate": agreed / scored if scored else None,
            "mean_probability": (sum(b["probs"]) / len(b["probs"])) if b["probs"] else None,
            "self_judged": name == "sonnet-5",
        })

    ranked = [a for a in arms if a["gold_accuracy"] is not None]
    rho = spearman(
        [a["judge_accuracy"] for a in ranked],
        [a["gold_accuracy"] for a in ranked],
    )
    total_scored = sum(a["agreement_scored"] for a in arms)
    total_matched = sum(a["agreement_matched"] for a in arms)

    out = {
        "judge_model": args.model,
        "rows_judged": len(rows),
        "rows_skipped_arm_failure": skipped_failures,
        "judge_failures": dict(judge_failures),
        "overall_agreement_scored": total_scored,
        "overall_agreement_matched": total_matched,
        "overall_agreement_rate": total_matched / total_scored if total_scored else None,
        "rank_correlation_spearman": rho,
        "judge_cost_micro": cost_micro,
        "arms": arms,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"agreement {total_matched}/{total_scored} "
          f"({100 * total_matched / total_scored:.1f}%)" if total_scored else "no rows scored")
    print(f"rank correlation (judge vs gold): {rho}")
    print(f"judge cost: ${cost_micro / 1_000_000:.4f}")
    if judge_failures:
        print(f"judge failures: {dict(judge_failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
