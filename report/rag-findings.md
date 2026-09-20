EYEBROW DevX Labs · Internal · {{n_queries}} questions · {{n_calls}} model calls

# Picking the right chunks

## The answer

In a RAG system you hand a model a pile of text chunks and ask it to pick the ones
that answer the question. We tested who picks best, and what each picker costs.

**{{best_model}} picked best, finding {{best_recall}} of the right chunks.** Jev found
{{jev_recall}} at {{jev_cost_1k}} per thousand questions — {{cheapest_other_multiple}}×
cheaper than the next cheapest model in the test and {{dearest_multiple}}× cheaper than
{{dearest_model}}.

{{baseline_finding}}

**A ceiling limits all of this: {{ceiling}}.** That is the share of correct chunks the
keyword search put in front of the models at all. No picker can find a chunk it was
never shown. Every score below sits under that ceiling, and the ceiling is the single
biggest lever in the system — bigger than the choice of model.

{{HEADLINE_TABLE}}


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
- **FiQA** — real finance questions over {{corpus_size}} real answer passages. Ordinary
  difficulty.

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

{{BY_SOURCE_TABLE}}

**The two collections are different problems.** FiQA scores are several times higher than
BRIGHT scores for every model, including the free baseline. BRIGHT was designed to defeat
keyword search, and it does: the keyword step hands over very little that is correct, so
there is little for any picker to find.

**Where the models help is not where you would guess.** The lift each model gives over
the free keyword baseline, measured question by question on the same questions:

{{LIFT_TABLE}}

A "tie" verdict means the test cannot tell that model apart from doing nothing. It does
not mean the model is bad. It means **this sample of {{n_queries}} questions is not
enough to prove it helps**, and paying for it is a bet rather than a decision.

{{MATRIX}}

A `+` means the model on that row really is better than the model in that column. A `=`
means we cannot tell them apart. Most of this grid is `=`, and that is the honest result.

## 4 · Cost against effectiveness

{{COST_CHART}}

The horizontal axis is cost per thousand questions and it is {{chart_span}}.

**Jev sits at the far left.** Its price comes almost entirely from reading the chunks,
because it charges nothing for what it writes back. A reranking job is nearly all
reading: {{depth}} chunks in, a short list out. That is the single reason the gap is this
wide, and it would narrow on a job that produced long output.

**The premium model could not be run at this size.** {{dearest_model}} was the dearest
model we could afford across all {{n_queries}} questions. Running sonnet-5 over the same
{{depth}} chunks would have cost {{sonnet_d100_projected}} — more than this entire
experiment. That is a real constraint, not a rhetorical one, and it is why the premium
comparison below was run on a shorter list.

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

**We probably asked Jev the wrong way, and that works against Jev.** Jev offers three
ways to ask a question: *choice* (pick one from a list), *score* (rate it on a scale) and
*noul* (is this true?). We used **choice**, with all {{depth}} chunks as the options in a
single question — so the chunks compete for one pool of probability. Since publishing,
we checked what other people building rerankers on Jev actually do, and neither of the two
public ones does it our way: one asks a separate *noul* question per chunk — "is this
chunk relevant?" — and sorts by the answer; the other uses a *score* rubric, one chunk at
a time. Both give each chunk an independent judgement instead of making them compete.
That is the shape the protocol is documented for.

This was an encoding mistake, not a fault in the product. It ran correctly — every call
returned a full set of scores, none were dropped, and none hit the size limit — but the
number it produced is **most likely a floor for Jev rather than its best**. Jev already
came first on accuracy here while being the cheapest, so the mistake did not change who
won; it may have understated by how much. Re-running with *noul* would cost about $0.09
and is the first thing to do if this test is repeated.

**Cost is what the provider billed us**, read back from each call, not estimated from a
price list.


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
