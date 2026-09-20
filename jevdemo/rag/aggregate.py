"""Per-call rows into per-model numbers.

Four rules decide every number below, and each one exists because the obvious
alternative makes the comparison unfair:

1. A failed call is not a zero. An HTTP 500 is a reliability fact; folding it
   into a quality score means the report can no longer tell a bad ranker from a
   flaky one. Failures are counted separately and named.

2. Two scores, not one. `recall@k` is over every judged gold passage, including
   the ones BM25 never retrieved — that is what the whole pipeline delivers, and
   it is the number a system owner lives with. `reachable@k` is over only the
   gold that was actually in the candidate list, which is the reranker's own
   share of the work. Reporting only the first blames the reranker for the
   retriever; only the second hides the ceiling.

3. Comparisons are paired on query id, never on position. Rows come back from a
   thread pool in arrival order.

4. Speed and price are per *question*, and a question does not cost every arm the
   same number of HTTP calls. The per-passage encodings send one call per
   candidate — a hundred of them at depth 100 — so their per-question latency and
   spend are a hundred calls' worth, while a single-call arm's are one. Both are
   real, and neither is comparable to the other without saying which it is, so
   `calls_per_query` is carried alongside them and the report prints it.
"""

from __future__ import annotations

import random

from jevdemo.arms import ARMS

from jevdemo.rag.rank_stats import mean_interval

SCORES = ("recall@1", "recall@3", "recall@5", "recall@10",
          "reachable@5", "reachable@10", "ndcg@5", "ndcg@10", "mrr")


#: Arm name -> kind, so `summarise` can tell a one-call arm from a per-passage one
#: without re-deriving the roster. Missing name falls back to one call, which is
#: what every non-Jev arm does.
_KIND = {a.name: a.kind for a in ARMS}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarise(rows: list[dict], seed: int) -> dict[str, dict]:
    """One dict per arm. `seed` fixes the bootstrap so the report reproduces."""
    out: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm]
        ok = [r for r in mine if r["failure"] is None]
        rng = random.Random(seed)
        s: dict = {
            "arm": arm,
            # One HTTP call per candidate for the per-passage encodings, one for
            # everyone else. Derived from the arm registry and the depth actually
            # run, so it cannot drift from what the harness did.
            "calls_per_query": (mine[0]["depth"]
                                if _KIND.get(arm) == "jev-pair" else 1),
            "attempted": len(mine),
            "scored": len(ok),
            "failures": len(mine) - len(ok),
            "failure_kinds": sorted({r["failure"] for r in mine if r["failure"]}),
            "dropped_ids": sum(r["dropped_ids"] for r in mine),
            "queries_with_dropped_ids": sum(1 for r in mine if r["dropped_ids"]),
        }
        for key in SCORES:
            # `reachable@k` is None on a query whose gold was never retrieved,
            # and those Nones are dropped rather than counted as zeros.
            vals = [r[key] for r in ok if r.get(key) is not None]
            s[key] = sum(vals) / len(vals) if vals else None
            s[f"{key}_ci95"] = mean_interval(vals, rng) if vals else None
            if key == "reachable@5":
                s["reachable_n"] = len(vals)

        lat = [r["elapsed_ms"] for r in ok]
        s["p50_ms"] = _pct(lat, 0.50)
        s["p95_ms"] = _pct(lat, 0.95)
        s["input_tokens"] = sum(r["input_tokens"] for r in mine)
        s["output_tokens"] = sum(r["output_tokens"] for r in mine)
        s["reasoning_tokens"] = sum(r["reasoning_tokens"] for r in mine)
        # Every attempt is billed, so the numerator spans `mine` — but a call that
        # came back unusable bought nothing, so the denominator is `ok`. Dividing
        # by attempts instead would make an arm look cheaper exactly in proportion
        # to how often it failed. With no answers at all the price is undefined,
        # not zero: None, so nothing downstream can print it as cheap.
        s["cost_micro"] = sum(r["reported_cost_micro"] or 0 for r in mine)
        s["cost_per_1k_micro"] = (
            round(s["cost_micro"] * 1000 / len(ok)) if ok else None)
        out[arm] = s
    return out


def align(rows: list[dict], a: str, b: str, key: str) -> tuple[list[float], list[float]]:
    """Score vectors for two arms over the queries *both* answered successfully.

    Paired on query id and returned in a fixed query order, so the comparison is
    the same whichever order the thread pool finished in.
    """
    def by_query(arm):
        return {r["query_id"]: r[key] for r in rows
                if r["arm"] == arm and r["failure"] is None and r.get(key) is not None}

    left, right = by_query(a), by_query(b)
    shared = sorted(set(left) & set(right))
    return [left[q] for q in shared], [right[q] for q in shared]
