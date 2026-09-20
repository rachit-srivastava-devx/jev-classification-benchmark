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
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevdemo.rag.corpus import load_tasks
from jevdemo import negotiate                                   # noqa: E402
from jevdemo.arms import ARMS                                   # noqa: E402
from jevdemo.rag import chat_rank, jev_rank                     # noqa: E402
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
ROSTER_100 = ["jev", "glm-5.3-flash-low", "deepseek-v4.1-flash-low",
              "qwen3.8-flash-low", "gemini-3.8-flash-low"]

#: Depth 20 is the affordability contrast, and the only depth at which a premium
#: model fits the budget. Two cheap models come along so the depths are
#: comparable rather than being two unrelated experiments.
ROSTER_20 = ["jev", "glm-5.3-flash-low", "sonnet-5"]

TOP_K = 10      #: how many ids a chat model is asked to return
WORKERS = 4
SCORE_AT = (1, 3, 5, 10)


def module_for(arm):
    return jev_rank if arm.kind == "jev" else chat_rank


def one(arm, task, depth, transport, headers, reasoning):
    view = dict(task, candidates=task["candidates"][:depth])
    ids = [c["id"] for c in view["candidates"]]
    mod = module_for(arm)
    url, body = mod.build_request(arm, view, TOP_K, reasoning)
    started = time.monotonic()
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
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            got = list(pool.map(
                lambda t: one(arm, t, args.depth, transport, headers, reasoning), tasks))
        rows.extend(got)
        spent += sum(r["reported_cost_micro"] or 0 for r in got)
        ok = [r for r in got if r["failure"] is None]
        r5 = [r["recall@5"] for r in ok if r["recall@5"] is not None]
        print(f"      recall@5 {sum(r5) / len(r5) * 100:5.1f}%  " if r5 else "      no scores  ",
              f"{len(ok)}/{len(got)} ok  spend ${spent / 1e6:.3f}", flush=True)
        if spent > args.cap_micro:
            stopped = name
            print(f"      SPEND CAP HIT after {name}: "
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
