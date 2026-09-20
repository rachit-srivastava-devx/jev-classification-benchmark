"""Build the reranking task set from three public corpora. Run once; commit the output.

This is the only script in the project with a third-party dependency (pyarrow,
to read HuggingFace's parquet). It is a build-time tool, not part of the
measured path, and its output — `data/rag/tasks.json` — is committed so the
experiment can be re-run and audited without it.

    python scripts/fetch_rag_corpus.py

Three corpora, deliberately different in kind:

  BRIGHT (xlangai/BRIGHT) — reasoning-heavy retrieval. Real StackExchange posts
  as queries, several hundred words each, with several gold passages apiece.
  Built so that lexical overlap does not find the answer.

  FiQA-2018 (BeIR/fiqa) — ordinary retrieval. Short real finance questions over
  57,638 real answer passages.

  WANDS (wayfair/WANDS) — e-commerce product search. 480 real Wayfair queries
  over 42,994 real products, and the only one of the three that is *completely*
  judged: all 233,448 query-product pairs carry a label, so "a good passage
  nobody happened to judge" cannot penalise a model here. Its `Partial` label is
  treated as a negative on purpose — a partial match shares the query's words
  and is still the wrong product, which is the hardest kind of distractor a
  lexical retriever can hand over.

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

#: WANDS ships as three tab-separated files in the repository itself, not on a
#: dataset hub, so it is pinned to a commit rather than read from `main`: a
#: benchmark whose inputs can be edited under it is not reproducible. This is
#: the repository's current head, 3b74dcf of 2022-01-18, and it has not moved
#: since — the repository has two commits in total.
WANDS = ("https://raw.githubusercontent.com/wayfair/WANDS/"
         "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset/")

DEPTH = 100          #: candidates retrieved per query
PER_DOMAIN = 50      #: BRIGHT queries per domain
FIQA_N = 100         #: FiQA queries
WANDS_N = 100        #: WANDS queries
PASSAGE_CHARS = 1200  #: hard cap per passage, so one long document cannot dominate

#: A WANDS query with dozens of correct products is easy by construction: five
#: slots and twenty-eight right answers is not a discrimination test. Keeping
#: only the queries with few correct products is the "specially difficult" cut,
#: and it is declared here as a number rather than applied by eye.
WANDS_MAX_GOLD = 8

#: Corpus sizes, recorded as the corpora are read rather than typed into the
#: report by hand. The report used to carry one of these as a literal with a
#: `dict.get` default that could never fire, which is how it came to describe
#: FiQA's 57,638 passages with the 218,052 total of all four text corpora.
#: A number the reader is asked to trust has to come from the run that made it.
CORPUS_SIZES: dict[str, int] = {}
SIZES_PATH = "data/rag/corpus-sizes.json"


def _note_size(source: str, n: int) -> None:
    """Record one corpus's size, refusing a silent disagreement between runs."""
    if CORPUS_SIZES.setdefault(source, n) != n:
        raise ValueError(
            f"{source}: counted {n:,} passages now, {CORPUS_SIZES[source]:,} earlier")


def _read(url: str) -> list[dict]:
    return pq.read_table(fsspec.open(url, "rb").open()).to_pylist()


def _task(source, qid, query, cands, gold, index, corpus):
    ranked = [d for d, _ in cands]
    reachable = gold.intersection(ranked)
    return {
        "source": source,
        "query_id": f"{source}:{qid}",
        "query": query,
        "gold_ids": sorted(gold),
        "candidates": [{"id": d, "text": corpus[d][:PASSAGE_CHARS]} for d in ranked],
        "bm25_order": ranked,
        # The difficulty stratum, decided here by a rule rather than later by
        # eye. `hard` is the stratum where a reranker is the only thing that can
        # help: the retriever did put a correct passage somewhere in the 100, and
        # put none of them in the top 5. Queries the retriever already solved
        # cannot show a reranker doing anything, and queries where it retrieved
        # nothing correct cannot either — those are `easy` and `unreachable`.
        # Every query is kept and the stratum is published, so nothing is
        # selected away; splitting after the results are in is how a benchmark
        # gets accused of picking its own sample, and rightly.
        "stratum": ("unreachable" if not reachable
                    else "hard" if not gold.intersection(ranked[:5])
                    else "easy"),
        "gold_reachable": len(reachable),
    }


def bright_tasks() -> list[dict]:
    out = []
    for domain in BRIGHT_DOMAINS:
        docs = _read(f"{BRIGHT}documents/{domain}/0000.parquet")
        corpus = {d["id"]: d["content"] for d in docs}
        index = BM25(corpus)
        examples = _read(f"{BRIGHT}examples/{domain}/0000.parquet")
        _note_size(f"bright-{domain}", len(corpus))
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
    _note_size("fiqa", len(corpus))
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


def _wands_rows(name: str) -> list[dict]:
    """WANDS files are tab-separated with embedded commas, so csv with an
    explicit delimiter, not a naive split."""
    import csv
    import io
    with fsspec.open(WANDS + name, "rb") as fh:
        text = fh.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def wands_tasks() -> list[dict]:
    """E-commerce product search, fully judged.

    A product's text is assembled from the fields a shopper actually reads —
    name, class, category path, description, features — because ranking on the
    name alone would make this a string-match task rather than a retrieval one.

    Gold is `Exact` only. `Partial` is deliberately scored as wrong: WANDS
    defines it as a product of the right general kind that is not what was
    asked for, which is exactly the distractor BM25 ranks highly and exactly the
    distinction a reranker is bought to make. Counting Partial as correct would
    hand every model most of the score for free.
    """
    prods = _wands_rows("product.csv")
    corpus = {}
    for p in prods:
        parts = [p["product_name"], p["product_class"], p["category hierarchy"],
                 p["product_description"], p["product_features"]]
        text = "\n".join(x for x in parts if x and x != "NULL").strip()
        if text:
            corpus[p["product_id"]] = text
    index = BM25(corpus)

    queries = {q["query_id"]: q["query"] for q in _wands_rows("query.csv")}
    gold_by_q: dict[str, set[str]] = {}
    for r in _wands_rows("label.csv"):
        if r["label"] == "Exact" and r["product_id"] in corpus:
            gold_by_q.setdefault(r["query_id"], set()).add(r["product_id"])
    _note_size("wands", len(corpus))
    print(f"  wands: {len(corpus):,} products, {len(gold_by_q)} queries with an exact match")

    out = []
    for qid in sorted(gold_by_q, key=int):
        gold = gold_by_q[qid]
        if not 1 <= len(gold) <= WANDS_MAX_GOLD or qid not in queries:
            continue          # too many correct answers to be a discrimination test
        hits = index.search(queries[qid], k=DEPTH)
        if len(hits) < DEPTH:
            continue          # too few candidates to pose the same task to everyone
        out.append(_task("wands", qid, queries[qid], hits, gold, index, corpus))
        if len(out) >= WANDS_N:
            break
    return out


def _write_sizes(root: pathlib.Path) -> pathlib.Path:
    """Write the corpus sizes the report quotes, keyed by the source names in tasks.json."""
    out = root / SIZES_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"corpus_sizes": dict(sorted(CORPUS_SIZES.items())),
         "total": sum(CORPUS_SIZES.values())}, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    # `--sizes-only` re-reads the same pinned corpora to record their sizes
    # without touching tasks.json: the committed task set is what the published
    # results were measured against, so rebuilding it would invalidate them.
    sizes_only = "--sizes-only" in (argv if argv is not None else sys.argv[1:])
    root = pathlib.Path(__file__).resolve().parents[1]
    print("building reranking tasks (BM25 over the real corpora)")
    tasks = bright_tasks() + fiqa_tasks() + wands_tasks()
    sizes = _write_sizes(root)
    print(f"\ncorpus sizes -> {sizes}")
    for name, n in sorted(CORPUS_SIZES.items()):
        print(f"  {name:20} {n:>9,} passages")
    print(f"  {'TOTAL':20} {sum(CORPUS_SIZES.values()):>9,} passages")

    if sizes_only:
        # Free provenance check: the committed tasks must still be the tasks
        # these pinned sources produce. A mismatch means a source moved.
        committed = json.loads((root / "data/rag/tasks.json").read_text())["tasks"]
        now, was = [t["query_id"] for t in tasks], [t["query_id"] for t in committed]
        print(f"\ntasks.json NOT rewritten (--sizes-only). "
              f"query ids match committed set: {now == was} "
              f"({len(now)} rebuilt vs {len(was)} committed)")
        return

    out = root / "data/rag/tasks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"depth": DEPTH, "passage_chars": PASSAGE_CHARS,
                               "corpus_sizes": dict(sorted(CORPUS_SIZES.items())),
                               "tasks": tasks}, indent=1))
    kb = out.stat().st_size / 1024
    print(f"\n{len(tasks)} tasks -> {out} ({kb:,.0f} KB)")
    for src in sorted({t["source"] for t in tasks}):
        rows = [t for t in tasks if t["source"] == src]
        gold = sum(len(t["gold_ids"]) for t in rows) / len(rows)
        reach = sum(len(set(t["gold_ids"]) & set(t["bm25_order"])) / len(t["gold_ids"])
                    for t in rows) / len(rows)
        strata = {s: sum(1 for r in rows if r["stratum"] == s)
                  for s in ("hard", "easy", "unreachable")}
        print(f"  {src:20} {len(rows):3} queries  {gold:4.1f} gold/query  "
              f"BM25 recall@{DEPTH} {reach * 100:5.1f}%  "
              f"hard {strata['hard']:3} easy {strata['easy']:3} "
              f"unreachable {strata['unreachable']:3}")


if __name__ == "__main__":
    main()
