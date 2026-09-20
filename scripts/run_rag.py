"""Run the reranking benchmark and write the results file.

    python scripts/run_rag.py --depth 20 --cap-micro 2000000

Every model sees the same queries, the same candidate passages, in the same
retrieval order, and is asked for the same thing. The only variable is the model
and the protocol it answers in.

Two experiments, both driven from this one script:

  depth 20  — the whole roster, the short list a reranker usually sees.
  depth 100 — the cheap models only, the realistic context size. The premium
              models are left out of that one because it costs more than the
              budget allows, which is itself part of the finding.

A hard spend cap is enforced against the provider's own reported cost, checked
after every call. It stops the run rather than the arm: a partial roster is
readable, a silently truncated one is not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevdemo.rag.corpus import load_tasks
from jevdemo import negotiate                                   # noqa: E402
from jevdemo.arms import ARMS                                   # noqa: E402
from jevdemo.rag import chat_rank, jev_pair, jev_rank           # noqa: E402
from jevdemo.rag.rank_metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from jevdemo.runner import headers_for, load_key                # noqa: E402
from jevdemo.transport import HttpTransport, Throttled, TransportError  # noqa: E402

BY_NAME = {a.name: a for a in ARMS}

#: The full roster, at depth 20. `sonnet-5` is the premium anchor: one model
#: whose per-token price is two orders of magnitude above Jev's, to show what
#: that buys on this task.
#: Depth 100 is the headline: it is the context size a real RAG pipeline hands a
#: reranker, and on the hard corpus a 20-passage short list contains so little
#: gold (2-5%) that no reranker could find it. These five are the models whose
#: price makes 100 passages x 101 queries affordable at all.
#: `jev-noul` and `jev-score` are the same model as `jev`, asked in its other two
#: documented grammars — one call per passage instead of one call per query. They
#: run at both depths because the earlier report's headline number was a property
#: of the Choice encoding and was reported as a property of the product.
#: `sonnet-5` is here too, unlike in the first report: at the raised budget a
#: premium model is affordable at the realistic depth, so the comparison no
#: longer has to be made across two different depths.
ROSTER_100 = ["jev", "jev-noul", "jev-score", "glm-5.3-flash-low",
              "deepseek-v4.1-flash-low", "qwen3.8-flash-low", "gemini-3.8-flash-low",
              "sonnet-5"]

#: Depth 20 is the affordability contrast, and the only depth at which a premium
#: model fits the budget. Two cheap models come along so the depths are
#: comparable rather than being two unrelated experiments.
ROSTER_20 = ["jev", "jev-noul", "jev-score", "glm-5.3-flash-low", "sonnet-5"]

TOP_K = 10      #: how many ids a chat model is asked to return
WORKERS = 4
PAIR_OUTER_WORKERS = 2   #: x PAIR_WORKERS inside jev_pair = 12 in flight, the cookbook's number
SCORE_AT = (1, 3, 5, 10)


def module_for(arm):
    return jev_rank if arm.kind == "jev" else chat_rank


def one(arm, task, depth, transport, headers, reasoning):
    view = dict(task, candidates=task["candidates"][:depth])
    ids = [c["id"] for c in view["candidates"]]
    started = time.monotonic()
    if arm.kind == "jev-pair":
        # One HTTP call per candidate, fanned out inside `jev_pair.rank`. The
        # elapsed time is the whole query's wall clock, which is the honest
        # latency for this arm: a user waits for the last passage, not the first.
        res = jev_pair.rank(arm, arm.primitive, view, transport, headers)
        elapsed_ms = (time.monotonic() - started) * 1000
    else:
        mod = module_for(arm)
        url, body = mod.build_request(arm, view, TOP_K, reasoning)
        try:
            status, raw, elapsed_ms = transport(url, body, headers)
        except TransportError as exc:
            from jevdemo.rag.rank_result import HTTP_ERROR, failed
            res = failed(HTTP_ERROR, str(exc)[:200])
            elapsed_ms = (time.monotonic() - started) * 1000
        else:
            res = mod.parse(raw, status, ids)
    gold = set(task["gold_ids"])
    row = {
        "arm": arm.name, "query_id": task["query_id"], "source": task["source"],
        "depth": depth, "elapsed_ms": round(elapsed_ms, 1),
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "reasoning_tokens": res.reasoning_tokens,
        "reported_cost_micro": res.reported_cost_micro,
        "dropped_ids": res.dropped, "failure": res.failure, "detail": res.detail,
        "ranking": res.ranking[:max(SCORE_AT)],
        "gold_in_view": len(gold.intersection(ids)),
        "gold_total": len(gold),
        "subcalls": res.subcalls, "subcall_failures": res.subcall_failures,
        # Carried through from the task so every cut in the report — by source,
        # by difficulty — can be made from the results file alone, without
        # re-deriving the stratum and risking a different rule the second time.
        "stratum": task.get("stratum", "unknown"),
    }
    row.update(_score(res.ranking, gold, ids))
    return row


def _score(ranking, gold, ids):
    """Two denominators, both kept.

    `recall@k` is over all judged gold passages, including the ones the first
    stage never retrieved — that is the number a RAG system actually lives with.
    `reachable@k` is over only the gold passages present in the candidate list,
    which is the reranker's own share of the work. Reporting just the first
    blames the reranker for the retriever; just the second hides the ceiling.
    """
    out = {}
    reachable = gold.intersection(ids)
    for k in SCORE_AT:
        out[f"recall@{k}"] = recall_at_k(ranking, gold, k)
        out[f"reachable@{k}"] = recall_at_k(ranking, reachable, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, gold, k)
    out["mrr"] = mrr(ranking, gold)
    return out


def bm25_rows(tasks, depth):
    """The free baseline: the retriever's own order, reranked by nobody."""
    rows = []
    for t in tasks:
        ids = [c["id"] for c in t["candidates"][:depth]]
        row = {
            "arm": "bm25-baseline", "query_id": t["query_id"], "source": t["source"],
            "depth": depth, "elapsed_ms": 0.0, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "reported_cost_micro": 0, "dropped_ids": 0,
            "failure": None, "detail": "", "ranking": ids[:max(SCORE_AT)],
            "gold_in_view": len(set(t["gold_ids"]).intersection(ids)),
            "gold_total": len(t["gold_ids"]),
            "subcalls": 0, "subcall_failures": 0,
            "stratum": t.get("stratum", "unknown"),
        }
        row.update(_score(ids, set(t["gold_ids"]), ids))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks (0 = all)")
    ap.add_argument("--cap-micro", type=int, default=2_000_000,
                    help="stop the run once reported spend passes this, in micro-dollars")
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names, overriding the roster for this depth. "
                         "Used for throughput and cost probes before a full paid run.")
    args = ap.parse_args(argv)

    key = load_key()
    if not key:
        print("no OPENROUTER_API_KEY", file=sys.stderr)
        return 2
    headers = headers_for(key)
    transport = Throttled(HttpTransport(), time.sleep)

    payload = load_tasks(ROOT / "data/rag/tasks.json")
    tasks = payload["tasks"][: args.limit] if args.limit else payload["tasks"]
    roster = ROSTER_20 if args.depth <= 20 else ROSTER_100
    if args.arms:
        roster = [n.strip() for n in args.arms.split(",") if n.strip()]
        unknown = [n for n in roster if n not in BY_NAME]
        if unknown:
            print(f"unknown arms: {', '.join(unknown)}", file=sys.stderr)
            return 2

    print(f"depth {args.depth} · {len(tasks)} queries · {len(roster)} models "
          f"· cap ${args.cap_micro / 1e6:.2f}")
    rows = bm25_rows(tasks, args.depth)
    spent = 0
    stopped = None

    for i, name in enumerate(roster, 1):
        arm = BY_NAME[name]
        print(f"[{i}/{len(roster)}] {name}", flush=True)
        # "Reasoning off" is spelled differently by different providers and is a
        # hard 400 on some of them, so it is negotiated once per arm — not per
        # query, and not assumed. Costs one short call.
        reasoning = arm.reasoning
        if arm.kind != "jev" and arm.reasoning is not None:
            outcome = negotiate.resolve(arm, tasks[0]["query"][:400], transport, headers)
            reasoning = outcome.reasoning
            print(f"      reasoning={reasoning}"
                  + (f" ({outcome.error})" if outcome.error else ""), flush=True)
        # The per-pair arms already run their own pool over the candidates, so
        # the outer pool is narrowed for them: concurrency is the product of the
        # two, and 4 x 6 against a public endpoint is how a run turns into 429s.
        outer = PAIR_OUTER_WORKERS if arm.kind == "jev-pair" else WORKERS
        # Consumed as they land rather than with `pool.map`, so the spend cap is
        # checked against every query instead of only at the end of an arm. At
        # depth 100 a single premium arm is tens of dollars; a cap that can only
        # fire after it has already been paid for is a receipt, not a cap.
        got = []
        with ThreadPoolExecutor(max_workers=outer) as pool:
            futures = [pool.submit(one, arm, t, args.depth, transport, headers, reasoning)
                       for t in tasks]
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except CancelledError:
                    # Expected once the cap fires: `as_completed` keeps yielding
                    # the futures that were cancelled before they ever ran.
                    continue
                got.append(row)
                spent += row["reported_cost_micro"] or 0
                if spent > args.cap_micro:
                    # Cancels what has not started; the calls already in flight
                    # still land and are still counted below.
                    for f in futures:
                        f.cancel()
                    stopped = name
        rows.extend(got)
        ok = [r for r in got if r["failure"] is None]
        r5 = [r["recall@5"] for r in ok if r["recall@5"] is not None]
        print(f"      recall@5 {sum(r5) / len(r5) * 100:5.1f}%  " if r5 else "      no scores  ",
              f"{len(ok)}/{len(got)} ok  spend ${spent / 1e6:.3f}", flush=True)
        if stopped:
            print(f"      SPEND CAP HIT during {name}: "
                  f"${spent / 1e6:.3f} > ${args.cap_micro / 1e6:.2f}", file=sys.stderr)
            break

    out = pathlib.Path(args.out) if args.out else ROOT / f"data/rag/results-d{args.depth}.json"
    out.write_text(json.dumps({
        "depth": args.depth, "top_k": TOP_K, "tasks": len(tasks),
        "roster": roster, "stopped_after": stopped,
        "spend_micro": spent, "cap_micro": args.cap_micro,
        "score_at": list(SCORE_AT), "rows": rows,
    }, indent=1))
    print(f"\n{len(rows)} rows -> {out}  ·  spend ${spent / 1e6:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
