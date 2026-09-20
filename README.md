# jev-demo — two benchmarks against Jev 1.13

[TypeSafe's **Jev 1.13**](https://openrouter.ai/typesafe/jev-1.13) is a typed-decision model:
you hand it a state and a set of questions, and it answers with structured choices,
probabilities and confidence instead of prose. This repo measures it against ordinary chat
models on two jobs, and publishes both the numbers and the code that produced them.

Both benchmarks have been run end to end against live APIs. Both reports are built from the
run's own output — there is no figure in either PDF typed in by hand.

| Report | Job | Scale | Output |
|---|---|---|---|
| **1 · Classification** | route a support ticket to one of 5 queues | 100 tickets × 20 arms, 2,400 model calls | [`report.pdf`](report.pdf) |
| **2 · RAG reranking** | pick the best 5 of 100 retrieved chunks | 350 questions × 9 arms, 2,800 model calls | [`rag-report.pdf`](rag-report.pdf) |

The two reach opposite conclusions about Jev, and the reason is in report 2's own limits
section: report 1 gives Jev a 5-criteria question, which is its native shape; report 2 gives
it 100 candidate chunks, which exceeds its ~32,768-token input ceiling, so two of the three
Jev encodings were run as one call per chunk. Read the cost and latency figures for
`Jev · Score` and `Jev · Noul` as a property of that harness, not of the model.

## Reproducing it

Everything except the two `run_*` steps is offline, deterministic and free.

```bash
python3 -m venv .venv
.venv/bin/pip install aiohttp requests pyarrow pypdf
python3 -m pytest -q                       # 358 tests, offline, no key, no spend
```

The test suite has no third-party dependencies at all — the four packages above are
only needed by the fetch, run and report scripts. PDF rendering uses `weasyprint`,
installed separately (`brew install weasyprint`).

### Report 2 — RAG reranking

```bash
# 1. Build the task set from the pinned public corpora (~10 min, no API key).
#    Downloads BRIGHT, FiQA and WANDS, runs BM25 over each full corpus, and
#    writes the 350 frozen questions with their 100 candidates each.
.venv/bin/python scripts/fetch_rag_corpus.py

# 2. The paid run. Needs OPENROUTER_API_KEY in .env. --cap-micro is a hard
#    spend cap checked per query against the provider's own reported cost.
.venv/bin/python scripts/run_rag.py --depth 100 --cap-micro 60000000

# 3. Build the report from the run's output.
.venv/bin/python scripts/build_rag_report.py
weasyprint rag-report.html rag-report.pdf
```

`data/rag/tasks.json` and `data/rag/results-d100.json` are committed, so steps 3 works
without running 1 or 2 at all — which is the point. Anyone can rebuild the report from the
published raw rankings and check that the numbers fall out.

Step 1 is reproducible in the strict sense: re-running it on 2026-09-20 against the pinned
sources regenerated the identical 350 query ids. To re-check the corpus sizes the report
quotes without rebuilding the task set (which would invalidate the committed results):

```bash
.venv/bin/python scripts/fetch_rag_corpus.py --sizes-only
```

### Report 1 — classification

```bash
.venv/bin/python -m jevdemo.runner        # the paid run: 2,400 calls, $1.36
.venv/bin/python scripts/build_report.py
weasyprint report.html report.pdf
```

## What the measured path looks like

```
HF parquet + GitHub (WANDS pinned to commit 3b74dcf)
  ↓  BM25 over each FULL corpus — 261,046 passages in total
  ↓  top-100 per query, retrieval order preserved as the tie-break
  ↓  keep only queries with ≥1 gold passage, ≥100 hits, and (WANDS) ≤8 gold
  ↓  cap: BRIGHT 50 per domain · FiQA 100 · WANDS 100
data/rag/tasks.json          350 questions, frozen and hashed before any model call
  ↓  one identical candidate view handed to every arm
  ↓  Jev · Choice / Jev · Noul / Jev · Score / 5 chat models / BM25 floor
data/rag/results-d100.json   3,150 rows of raw rankings — no metrics stored here
  ↓  recall / nDCG / MRR recomputed from rankings + qrels at report time
rag-report.pdf
```

Metrics are deliberately not stored in the results file. The run records what each model
actually returned; scoring happens later, in code anyone can read, against gold labels that
ship with the datasets. Nobody on this project wrote a question or judged an answer.

## Data

| Source | What it is | Size |
|---|---|---|
| [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) | StackExchange posts as queries, built so lexical overlap does not find the answer | 160,414 passages across 3 domains |
| [FiQA-2018](https://huggingface.co/datasets/BeIR/fiqa) | real finance questions over answer passages | 57,638 passages |
| [WANDS](https://github.com/wayfair/WANDS) | Wayfair product search, fully human-judged | 42,994 products, 233,448 judgements |

Sizes are counted by the run that reads the corpora and written to
`data/rag/corpus-sizes.json`; the report builder raises rather than printing a number nobody
measured.

## How the code avoids flattering itself

The defects worth knowing about are the ones that make a benchmark look better than it is.
These are the guards, each with the mistake it exists to prevent:

- **Cost is divided by answers, not attempts** (`jevdemo/rag/aggregate.py`). Dividing by
  attempts makes an arm look cheaper exactly in proportion to how often it failed. An arm
  with no answers reports no price rather than `$0.00`.
- **An arm answering under 90% of questions has its score withheld**, not averaged over the
  subset that happened to survive.
- **Two denominators are published**: every arm over the questions it answered, and every arm
  over the 217 questions all of them answered. The rankings disagree, and that disagreement
  is reported rather than resolved by picking one.
- **Comparisons are paired on the query** and bootstrapped with a fixed seed, because every
  model answered the same questions.
- **Invented passage ids are dropped and counted**, so a model cannot score by naming chunks
  that were never retrieved.
- **A ceiling is published**: 46.5% of gold chunks were never put in front of the models by
  the retriever at all. No reranker can find a chunk it was never shown.

## Known limits

- The `Jev · Score` and `Jev · Noul` cost and latency figures in `rag-report.pdf` measure one
  call per chunk. The code now batches, and is unit-tested, but has not been re-run — the key
  hit its spend cap. Those two columns are an upper bound.
- `report/pipeline.architecture.json` still labels those arms "100 calls", which matches the
  published measurement and not the current code.
- Two of the three corpora are read from a mutable HuggingFace ref; only WANDS is pinned to a
  commit.
- Queries are the first N that qualify, not a random draw.

## Out of scope

Agent CLIs — Claude Code, Codex CLI, Kiro. They authenticate separately, expose no per-call
token or cost accounting, and would need subprocess driving, so their numbers would not be
comparable to the API arms.

## Specs

The design documents written before the first benchmark are in [`specs/`](specs/) —
[00-overview](specs/00-overview.md), [01-architecture](specs/01-architecture.md),
[02-dataset](specs/02-dataset.md), [03-arms](specs/03-arms.md),
[04-metrics](specs/04-metrics.md), [05-report](specs/05-report.md),
[06-tdd](specs/06-tdd.md), and the live API evidence they cite in
[PROBE-RESULTS](specs/PROBE-RESULTS.md).
