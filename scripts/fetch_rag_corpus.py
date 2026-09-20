"""Build the reranking task set from two public corpora. Run once; commit the output.

This is the only script in the project with a third-party dependency (pyarrow,
to read HuggingFace's parquet). It is a build-time tool, not part of the
measured path, and its output — `data/rag/tasks.json` — is committed so the
experiment can be re-run and audited without it.

    python scripts/fetch_rag_corpus.py

Two corpora, deliberately different in kind:

  BRIGHT (xlangai/BRIGHT) — reasoning-heavy retrieval. Real StackExchange posts
  as queries, several hundred words each, with several gold passages apiece.
  Built so that lexical overlap does not find the answer.

  FiQA-2018 (BeIR/fiqa) — ordinary retrieval. Short real finance questions over
  57,638 real answer passages.

Candidates come from BM25 over the real corpus, so they are the hard negatives a
first-stage retriever actually returns. No query is dropped for being hard: the
share of gold BM25 never surfaces is the ceiling every reranker is measured
against, and it is published rather than filtered away.
"""

from __future__ import annotations

import json
import pathlib
import sys

import fsspec
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from jevdemo.rag.bm25 import BM25  # noqa: E402

HF = "https://huggingface.co/datasets/"
BRIGHT = HF + "xlangai/BRIGHT/resolve/refs%2Fconvert%2Fparquet/"
FIQA = HF + "BeIR/fiqa/resolve/refs%2Fconvert%2Fparquet/"
FIQA_QRELS = HF + "BeIR/fiqa-qrels/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

#: Three BRIGHT domains, chosen for being readable by a non-specialist. The
#: maths and code domains (aops, theoremqa, leetcode) are excluded: they measure
#: symbolic reasoning, which is a different question from chunk selection.
BRIGHT_DOMAINS = ("economics", "biology", "psychology")

DEPTH = 100          #: candidates retrieved per query
PER_DOMAIN = 17      #: BRIGHT queries per domain
FIQA_N = 50          #: FiQA queries
PASSAGE_CHARS = 1200  #: hard cap per passage, so one long document cannot dominate


def _read(url: str) -> list[dict]:
    return pq.read_table(fsspec.open(url, "rb").open()).to_pylist()


def _task(source, qid, query, cands, gold, index, corpus):
    ranked = [d for d, _ in cands]
    return {
        "source": source,
        "query_id": f"{source}:{qid}",
        "query": query,
        "gold_ids": sorted(gold),
        "candidates": [{"id": d, "text": corpus[d][:PASSAGE_CHARS]} for d in ranked],
        "bm25_order": ranked,
    }


def bright_tasks() -> list[dict]:
    out = []
    for domain in BRIGHT_DOMAINS:
        docs = _read(f"{BRIGHT}documents/{domain}/0000.parquet")
        corpus = {d["id"]: d["content"] for d in docs}
        index = BM25(corpus)
        examples = _read(f"{BRIGHT}examples/{domain}/0000.parquet")
        print(f"  bright/{domain}: {len(corpus):,} passages, {len(examples)} queries")
        taken = 0
        for ex in examples:
            gold = {g for g in ex["gold_ids"] if g in corpus}
            if not gold:
                continue          # no judged passage in this corpus: nothing to score
            excluded = {e for e in ex["excluded_ids"] if e != "N/A"}
            hits = [(d, s) for d, s in index.search(ex["query"], k=DEPTH + len(excluded))
                    if d not in excluded][:DEPTH]
            if len(hits) < DEPTH:
                continue          # too few candidates to pose the same task to everyone
            out.append(_task(f"bright-{domain}", ex["id"], ex["query"], hits, gold, index, corpus))
            taken += 1
            if taken >= PER_DOMAIN:
                break
    return out


def fiqa_tasks() -> list[dict]:
    docs = _read(f"{FIQA}corpus/corpus/0000.parquet")
    corpus = {d["_id"]: (d["title"] + "\n" + d["text"]).strip() for d in docs}
    index = BM25(corpus)
    queries = {q["_id"]: q["text"] for q in _read(f"{FIQA}queries/queries/0000.parquet")}
    gold_by_q: dict[str, set[str]] = {}
    for r in _read(FIQA_QRELS):
        if int(r["score"]) > 0:
            gold_by_q.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    print(f"  fiqa: {len(corpus):,} passages, {len(gold_by_q)} judged queries")
    out = []
    for qid in sorted(gold_by_q, key=int):
        gold = {g for g in gold_by_q[qid] if g in corpus}
        if not gold or qid not in queries:
            continue
        hits = index.search(queries[qid], k=DEPTH)
        if len(hits) < DEPTH:
            continue
        out.append(_task("fiqa", qid, queries[qid], hits, gold, index, corpus))
        if len(out) >= FIQA_N:
            break
    return out


def main() -> None:
    print("building reranking tasks (BM25 over the real corpora)")
    tasks = bright_tasks() + fiqa_tasks()
    out = pathlib.Path(__file__).resolve().parents[1] / "data/rag/tasks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"depth": DEPTH, "passage_chars": PASSAGE_CHARS,
                               "tasks": tasks}, indent=1))
    kb = out.stat().st_size / 1024
    print(f"\n{len(tasks)} tasks -> {out} ({kb:,.0f} KB)")
    for src in sorted({t["source"] for t in tasks}):
        rows = [t for t in tasks if t["source"] == src]
        gold = sum(len(t["gold_ids"]) for t in rows) / len(rows)
        reach = sum(len(set(t["gold_ids"]) & set(t["bm25_order"])) / len(t["gold_ids"])
                    for t in rows) / len(rows)
        print(f"  {src:20} {len(rows):3} queries  {gold:4.1f} gold/query  "
              f"BM25 recall@{DEPTH} {reach * 100:5.1f}%")


if __name__ == "__main__":
    main()
