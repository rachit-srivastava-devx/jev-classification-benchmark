#!/usr/bin/env python3
"""Reprice an existing results.json from its stored token totals.

The first corrected run was written while `computed_cost_micro` was a sum of
per-call costs, each rounded half-up to a whole micro-dollar. A call can cost
less than one micro-dollar, so that rounding inflated every cheap arm and made
the reconciliation compare a sum-of-roundings against a rounding-of-sums.

This reprices in place using the same `compute_micro` and `per_thousand_micro`
the runner now uses, from token counts that were measured and are not in doubt.
It touches no other field, so nothing here is a second pricing path: re-running
the benchmark would produce the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jevdemo.arms import by_name  # noqa: E402
from jevdemo.pricing import compute_micro, per_thousand_micro, reconcile  # noqa: E402


def recost(results: dict) -> list[tuple[str, int, int]]:
    """Reprice every arm in place. Returns (arm, before, after) where it moved."""
    moved = []
    for row in results["arms"]:
        spec = by_name(row["arm"])
        before = row["computed_cost_micro"]
        after = compute_micro(spec, row["input_tokens"], row["output_tokens"])
        row["computed_cost_micro"] = after
        row["cost_per_1000_micro"] = (
            None if row["attempted"] == 0 else per_thousand_micro(after, row["attempted"])
        )
        reported = row["reported_cost_micro"]
        row["cost_reconciled"] = reconcile(
            spec, row["input_tokens"], row["output_tokens"],
            None if reported is None else reported / 1_000_000,
            calls=max(row["attempted"], 1),
        ).ok
        if before != after:
            moved.append((row["arm"], before, after))
    return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=ROOT / "results.json")
    args = ap.parse_args(argv)

    path = Path(args.results)
    results = json.loads(path.read_text())
    moved = recost(results)
    path.write_text(json.dumps(results, indent=2))

    for arm, before, after in moved:
        pct = (after - before) / before * 100 if before else 0.0
        print(f"{arm:24} {before:>8} -> {after:>8} micro-USD ({pct:+.2f}%)")
    print(f"\n{len(moved)} of {len(results['arms'])} arms repriced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
