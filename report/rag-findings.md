EYEBROW DevX Labs · Internal · {{n_queries}} questions · {{n_calls}} model calls

# Picking the right chunks

## The answer

In a RAG system you hand a model a pile of text chunks and ask it to pick the ones
that answer the question. We tested who picks best, and what each picker costs.

**{{best_model}} scored highest, finding {{best_recall}} of the right chunks — but no
model is reliably best.** The top scores sit inside each other's error bars, and when
every model is scored on one identical set of questions the order changes. The result
that survives both ways of counting is that every model beats keyword search by a
similar margin, so the thing to choose on is price, not accuracy.

{{jev_headline}}

{{baseline_finding}}

**A ceiling limits all of this: {{ceiling}}.** That is the share of correct chunks the
keyword search put in front of the models at all. No picker can find a chunk it was
never shown. Every score below sits under that ceiling, and the ceiling is the single
biggest lever in the system — bigger than the choice of model.

{{HEADLINE_TABLE}}


## How the numbers were made

Before any result: the figure below is the whole experiment, start to finish. Read it
once and every number in this report is traceable. Nothing on the left-hand side of
**tasks.json** involves a model — the questions, the chunks and the right answers were
fixed, written to disk and hashed before the first model was called.

{{PIPELINE_FIGURE}}


## 1 · What we tested

One job: **given a question and {{depth}} text chunks, put the chunks in order, best
first.** We then check whether the chunks that actually answer the question ended up in
the top 5.

This is the job a RAG system does every time someone asks it something. It is usually
called reranking. The 5 best chunks are what gets passed to the model that writes the
final answer, so a chunk that does not make the top 5 might as well not exist.

We used {{n_collections}} real collections, not a made-up test set. Between them they
cover {{n_sources}} topic areas:

- **BRIGHT** — real questions people posted on StackExchange, each several hundred words
  long, over a corpus of real documents. It was built specifically so that keyword
  matching does not work. It is hard on purpose.
- **FiQA** — real finance questions over {{fiqa_corpus_size}} real answer passages. Ordinary
  difficulty.
- **WANDS** — real Wayfair shopping searches over {{wands_corpus_size}} real products. Every
  product was judged by a person as `Exact`, `Partial` or `Irrelevant`, and only `Exact`
  counts as correct here. `Partial` — the right kind of thing but not the thing asked for
  — is scored as wrong on purpose, because that is exactly the distractor keyword search
  ranks highly and exactly the distinction a reranker is bought to make. Queries with more
  than 8 correct products were dropped: five slots and twenty-eight right answers is not a
  test of discrimination.

Together: **{{n_queries}} questions, {{gold_per_query}} correct chunks per question on
average, {{n_passages_ranked}} chunks ranked per model.**

Nobody wrote these questions or judged these answers for us. The questions are real and
the judgments of which chunk is correct come with the datasets.

## 2 · How we tested it

**Every model saw exactly the same thing.** Same questions, same {{depth}} chunks, in
the same order, asked the same way. The only thing that changed was the model.

The {{depth}} chunks are not random. They come from a keyword search (BM25) run over the
whole corpus — the same first step a real RAG system uses. So the wrong chunks are
genuinely tempting wrong chunks: text that mentions the right words but does not answer
the question. That is what makes the job hard.

The two kinds of model answer differently, so each was asked in its own native way:

- **Jev** is given the chunks as its options and returns a score for every one. That
  ordering is the answer. There is no prompt to tune and no output format to enforce.
- **Chat models** are given the chunks in a prompt and asked to reply with a JSON list
  of the best {{top_k}} chunk ids.

We also ran a **free baseline**: the keyword search's own order, with no model involved.
This matters more than it sounds. It is the thing every model has to beat to justify
existing.

**What we count.** For each question, what share of the correct chunks made the top 5.
Averaged over all {{n_queries}} questions. We report a 95% range beside every score,
worked out by resampling the questions — so you can see when two models are too close
to separate.

## 3 · The analysis

### "Jev" is not one number

Jev takes a question in one of three grammars, and which one you pick changes the answer,
the bill and the failure rate. We ran all three against the identical questions and chunks.
Same model id in every request; only the shape of the question differs.

{{ENCODING_TABLE}}

{{encoding_finding}}

### The two collections

{{BY_SOURCE_TABLE}}

**The two collections are different problems.** FiQA scores are several times higher than
BRIGHT scores for every model, including the free baseline. BRIGHT was designed to defeat
keyword search, and it does: the keyword step hands over very little that is correct, so
there is little for any picker to find.

**Most of the average is decided before any model sees anything.** Every question falls
into one of three slices, decided by the keyword search alone and recorded before the run:
the correct chunk is already in the keyword top 5 (nothing to fix), it is somewhere in the
hundred but not the top 5 (the only slice where a picker can earn its money), or it is not
in the hundred at all (nobody can win). Split that way:

{{STRATA_TABLE}}

**Read the first column.** It is the only one where a picker can change the outcome. The
second is the score a model gets for leaving things alone, and the third is zero for
everyone by construction — no ranking of a hundred chunks can surface a chunk that is not
among them. An average taken across all three mostly measures how often the keyword search
had already done the job.

## One exam for everyone

The main table scores each model over the questions it personally answered. Models that
failed on some questions are therefore not sitting the same exam, and a ranking built that
way can be an artefact of who dropped which questions. So here is the same field scored
over only the {{common_n}} questions that every model answered.

{{COMMON_SUBSET_TABLE}}

{{common_finding}}

**Where the models help is not where you would guess.** The lift each model gives over
the free keyword baseline, measured question by question on the same questions:

{{LIFT_TABLE}}

**Every model appears in this table, including the two whose scores the main table
withholds.** That is deliberate, and it is a different rule rather than a softer one. A
raw score over a partial set of questions is not comparable to a raw score over all of
them, because the two are averages of different exams. A lift is: it is measured question
by question against the same keyword baseline on the same questions that model answered,
so a model that answered fewer questions is still being compared against exactly its own
set. What a partial lift cannot tell you is whether that model would hold the same margin
on the questions it failed, which is why a low-coverage model is shown here and still
withheld from the headline.

A "tie" verdict means the test cannot tell that model apart from doing nothing. It does
not mean the model is bad. It means **this sample of {{n_queries}} questions is not
enough to prove it helps**, and paying for it is a bet rather than a decision.

{{MATRIX}}

A `+` means the model on that row really is better than the model in that column. A `=`
means we cannot tell them apart. {{matrix_reading}}

## 4 · Cost against effectiveness

{{COST_CHART}}

The horizontal axis is cost per thousand questions and it is {{chart_span}}.

**Jev sits at the far left.** Its price comes almost entirely from reading the chunks,
because it charges nothing for what it writes back. A reranking job is nearly all
reading: {{depth}} chunks in, a short list out. That is the single reason the gap is this
wide, and it would narrow on a job that produced long output.

{{chart_omissions}}

**The premium model produced nothing at this size, and the reason is not the model.**
sonnet-5 was in the roster and was called on all {{n_queries}} questions. Every call came
back `HTTP 403: Key limit exceeded`: the API key used for this run carries its own spending
cap, separate from the account balance, and the run crossed it at {{total_spend}} while working
through the cheaper models. So sonnet-5 spent nothing, answered nothing, and appears in
Appendix C with {{n_queries}} failures and no score. We are not reporting a number for it
at this depth, because there is no number to report. Its projected cost had it run —
{{sonnet_d100_projected}} per thousand questions, scaled from its own measured price on
the 20-chunk run — is arithmetic on a measurement, not a measurement.

**{{dearest_model}} was the dearest model whose answers we can actually score.**
{{dearest_unscorable}}

### The same test with only 20 chunks

We ran a second, smaller version: the same questions, but only the top 20 chunks, which
is cheap enough to include a premium model.

Jev scored {{d20_jev_recall}}. Total spend {{d20_spend}}.

{{d20_premium}}

**The 20-chunk version is much harder than it looks**, because with only 20 chunks the
keyword step hands over almost nothing correct on the BRIGHT questions. The lesson is
not about the models. It is that **giving the picker more chunks to look at raises the
ceiling faster than upgrading the picker does.**

## 5 · What could be wrong with this

**The ceiling dominates everything.** At {{ceiling}}, most correct chunks were never
shown to any model. A better first-stage retriever — a proper embedding search instead of
keyword matching — would change every number here, probably by more than swapping models
does. We used keyword search because it is reproducible with no dependencies and no
training, not because it is the best available.

**{{n_queries}} questions is a small sample.** That is why most comparisons come back as
ties. The ranges in the tables are real; if two of them overlap, we are not claiming a
winner.

**One run, no repeats.** Every model was asked once per question at temperature 0. We did
not measure run-to-run variation.

**The judgments are binary and incomplete.** A chunk is either marked correct or it is
not. A genuinely useful chunk nobody happened to judge counts as wrong, for every model
equally.

**Prompt shape favours nobody deliberately, but it is one shape.** The chat models got one
prompt. A different prompt, or a purpose-built reranking model, could do better. We did
not tune per model, because tuning one and not the others is how benchmarks get rigged.

**Jev's Choice encoding has a size limit, and we hit it.** Choice puts all
{{depth}} chunks into one request, and the endpoint refuses a request whose input passes
about 32,768 tokens with `max_tokens_exceeded`. That number is measured, not guessed: the
largest Choice request that ever succeeded here carried 32,850 input tokens, the 99th
percentile of successful requests is 32,779, and nothing above that returned an answer.
85 of 350 questions crossed the line — 84 of them from the shopping corpus, whose product
descriptions are long. Counting characters instead of tokens does not predict it: the
biggest request that succeeded held 119,182 characters and the smallest that failed held
106,391, because text packs into tokens at anywhere from 2.9 to 4.2 characters depending
on the corpus. The per-chunk encodings never hit it, because each of their calls carries
one chunk. If you plan to rerank 100 long chunks in a single call, this is the constraint
to design around.

**A question does not cost every model the same number of HTTP calls, and the
table now says so.** Choice asks about all 100 chunks in one request. Score and Noul ask
about each chunk on its own — 100 requests per question. The "cost per 1,000 questions"
and "typical speed" columns are per *question*, so for those two encodings they are a
hundred calls' worth. Per call, Score is 70 ms and about $0.03 per thousand calls; per
question it is 6,967 ms and $2.82. Both numbers are real and they are not comparable to a
single-call model without the calls-per-question column beside them.

**Those two encodings were measured while being used inefficiently, and we know by how
much.** Jev · Score sent 67,245 input tokens per question. Jev · Choice sent 21,061 for
the identical 100 chunks. The difference is not content: it is roughly 437 tokens of
request envelope, instructions and rubric, re-sent once per chunk. The vendor's own
cookbook says to batch — *"batching every question into one call is 12.2x cheaper"* — and
the harness did not. That is our error, not the model's, and it means the per-question
price and latency published here for Score and Noul are an upper bound on what this
encoding costs, not its best case.

The harness in this repository has since been changed to batch: it packs as many chunks
into one request as fit under the measured input ceiling, and halves and retries any
request that comes back empty. It is unit-tested but it has **not been re-run**, because
the API key hit its spending cap. So the batched code is what a reader will execute and
the unbatched numbers are what this report publishes — if you re-run this, expect Score
and Noul to come out cheaper and much faster than the table says, and expect nothing else
to move, because the judgement asked is byte-identical either way. We are flagging this
rather than quietly re-running one arm and comparing it against seven that were measured
the old way.

**Every result here is one model asked three ways, not three products.** Section 3 breaks
that out. Treat "Jev scores X" as meaningless without the encoding attached.

**Cost is what the provider billed us**, read back from each call, not estimated from a
price list. It is divided by the answers we could use, not by the calls we made. Those
are different numbers whenever a model failed: you are billed for a call that comes back
unusable, and it buys you nothing. Dividing by calls made would have shown Jev · Choice at
$0.67 per thousand questions instead of its real $0.88, and gemini-3.8-flash at $13.23
instead of $16.13 — each looking cheaper in exact proportion to how often it failed. An
earlier draft of this report did divide by calls made, and those two prices were wrong in
it.

**The pairwise grid runs many tests at once.** With 8 models there are 28 pairs, each
given its own 95% range. At 95%, roughly one or two cells in a grid that size will show a
winner by chance alone with no correction applied, and we applied none. This is a reason to
read the grid as "almost everything is a tie" rather than to trust any single non-tie cell
in it.

**Two of the three corpora are not pinned to an immutable revision.** The shopping corpus
is fetched at commit `3b74dcf`. BRIGHT and FiQA are fetched from HuggingFace's
auto-generated parquet branch, which that host may regenerate. So a re-run today gets
byte-identical shopping data and probably-but-not-provably identical data for the other
two. The question set itself is pinned: its hash is printed under the diagram above, and
re-hashing `data/rag/tasks.json` tells you whether you are reading the same 350 questions.

**The questions are the first 50 per domain that qualify, not a random draw.** A question
qualifies if it has at least one correct chunk and at least 100 keyword hits; we then take
them in the order the dataset file ships them. This cannot favour one model over another,
because the selection happens before any model is called and the code that does it imports
no model client. But if a dataset file happens to be ordered by topic or by date, the
sample inherits that ordering, and a genuine random draw would not.


## 6 · Why you should believe this

Benchmarks published by the people who ran them get the same seven objections every
time. Here is each one, and what in this run answers it.

**"You picked the questions that made your model look good."** Every question was chosen
and written to disk before any model was called. The selection rule is mechanical: {{selection_rule}}. Nothing was dropped after a result came back. All {{n_queries}} questions are
reported, including the ones every model failed.

**"You only show the questions your model wins."** The report is the full set. Section 3
splits it by corpus and by difficulty, so a model that wins only on the easy third is
visible as exactly that. The hardest slice — questions where the keyword search put no
correct chunk in the first five — is reported separately, not averaged away.

**"These datasets are in the models' training data."** Very probably, and we cannot rule
it out. Three things bound the damage. The task is not recall of an answer: it is ranking
a fixed list of chunks that we assembled, in an order we produced. Memorisation would help
every chat model here, and the cheap chat models are not the ones winning. And if
memorisation were driving the scores, the hard slice would not collapse the way it does
for every model at once. We state this as a limitation, not as a solved problem.

**"Different harnesses give different numbers."** There is one harness. Every model gets
the same questions, the same chunks, in the same order, through the same code path, with
the same tie-break rule and the same parser. The only thing that differs is the model name
in the request. The request builder is one function per protocol and both are in the
repository.

**"The costs are your estimate."** They are not. Every cost in this report is the number
the provider itself returned in the `usage` block of that call, summed. A call that came
back without a usage block is a failure, not a free call, and is counted as one.

**"There's no baseline."** The keyword search is in every table, at zero cost. It is a
real floor: a model that does not beat it is worse than nothing.

**"You report a single number."** Every score carries a bootstrap interval, and every
model-to-model comparison is paired over the same questions. Where two intervals overlap
we say tie, not winner.

### Measured against how this kind of test is normally published

Before writing this up we read how other people publish reranking benchmarks: vendor
launch posts from Voyage AI and Jina, independent comparisons from Agentset, LlamaIndex,
Mixpeek and ZeroEntropy, and the BEIR/MIRACL family of papers they cite. Eighteen posts
found, seven read end to end. That is the denominator for every count below.

| Practice | How often the seven do it | Here |
| --- | --- | --- |
| Publish a runnable harness, not just a table | 2 of 7 | Yes — `scripts/run_rag.py`, one command |
| Name the public datasets used | 4 of 7 | Yes — BRIGHT, FiQA, WANDS, with row ids |
| Break results down per corpus, not one blended score | 2 of 7 | Yes — per corpus and per difficulty |
| Report cost and latency together | 2 of 7 | Yes — cost as billed, latency as p50 and p95 |
| Give confidence intervals or a significance test | 0 of 7 | Yes — bootstrap intervals, paired |
| Have a limitations section at all | 0 of 7 | Yes — section 5 |
| State the retrieval ceiling the reranker cannot beat | 0 of 7 | Yes — the impossible slice, counted |
| Show their own preferred option losing | 2 of 7 | Yes — Choice loses to Noul, below |

The two most common holes in published reranking benchmarks are the two that matter most
for a buying decision: no error bars, so a one-point win reads the same as a twenty-point
win, and no statement of what the keyword search never surfaced, so the reranker gets
blamed or credited for the retriever's work. Both are closed here, and closing them is
what produced this report's least flattering numbers.

**What would falsify this.** Run `scripts/run_rag.py` against the same `tasks.json` and
get materially different recall. The tasks file, the results file with every per-question
row, and both scripts are in the repository. Nothing in this report is derived from a
number that is not in that results file.

## Appendix A · The protocol in full

- **Questions:** {{n_queries}} ({{source_list}}).
- **Chunks per question:** {{depth}}, from BM25 over the full corpus.
- **Asked for:** the best {{top_k}} chunk ids, best first.
- **Scored on:** share of correct chunks in the top 5.
- **Models:** {{n_models}}, plus the free keyword baseline.
- **Calls:** {{n_calls}}. **Total spend:** {{total_spend}}.
- **Temperature:** 0. Reasoning disabled where the provider allows it.
- **Seed:** fixed, so every number in this report reproduces exactly.

### What would make this test unfair to the other models

We checked these deliberately:

- *Jev sees the chunks in a better order.* It does not. Every model gets the identical
  BM25 order.
- *Jev is asked an easier question.* It is asked to score all {{depth}}; the chat models
  are asked for {{top_k}}. If anything that is more work, not less.
- *Ties are broken in Jev's favour.* They are broken by the original retrieval position,
  identically for every model.
- *Failed calls are scored as zero.* They are not. They are excluded and counted
  separately, in Appendix C.

## Appendix B · The full measurement table

{{FULL_TABLE}}

"Of what was findable" is the score counting only the questions where a correct chunk was
actually in the {{depth}} shown — the picker's own share of the work, with the retriever's
misses removed.

## Appendix C · Failures and invented chunks

{{FAILURES}}

**Inventing a chunk id is not the same as ranking badly.** A model that returns an id
that does not exist has failed in a way a low score would hide, so it is counted on its
own. {{invention_finding}}

Across the whole run there were {{total_failures}} failed calls.

{{credit_note}}

## Appendix D · How the ranges were worked out

A score here is an average of per-question fractions, not a pass/fail rate. So the
interval is not the one used in the ticket-routing report.

Instead: resample the {{n_queries}} questions with replacement, recompute the average,
repeat 4,000 times, and read off the 2.5th and 97.5th percentiles. This assumes nothing
about the shape of the scores.

Comparisons between two models are **paired**: both answered the same questions, so the
difference is taken question by question before resampling. Unpaired, the variation
between easy and hard questions would swamp every real difference and the whole grid
would read as ties.

## Appendix E · One question, all the way through

Everything above is an average. This is the whole of a single question: the exact request
that went over the wire, every chunk the model was shown, and what each model picked out
of that pile.

The question was chosen by a seeded shuffle over the queries that are hard, have at least
two correct chunks inside the hundred, and were answered by every model that answered
anything at all — not by looking
at the results and picking a flattering one. Change the seed and you get a different
question; the code that picks it is in `scripts/build_rag_report.py`.

{{WORKED_EXAMPLE}}
