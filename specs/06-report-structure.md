# 06 · Report structure

The audience is a CEO, and the shape is a research paper. Those two facts pull in
opposite directions — one wants the answer in ten seconds, the other wants the
method before the claim — so the document resolves them by order, not by
compromise: the conclusion is stated first with its uncertainty attached, and
everything that earns it follows in full.

## Audience contract

A CEO reading only the abstract and the decision rule must be able to act. A
researcher reading the whole document must be able to attack it: every number
traceable to `results.json`, every interval computed rather than asserted, every
arm's failures counted separately from its errors.

Nothing is softened for the executive reader. The honest finding at n=100 is that
most accuracy gaps in this roster will not be resolvable, and the report says so
in the abstract rather than burying it in limitations.

## Section order

Adapted from the conventions of HELM, MLPerf Inference, and the ML
Reproducibility Checklist, reordered so the decision comes first.

1. **Abstract** — the finding, the magnitude, and the uncertainty, in one
   paragraph. Includes the number of arms, tickets, and calls.
2. **The decision rule** — the break-even theorem applied to this roster. The
   single table a CEO acts on: for each arm, what a misroute must cost before
   that arm is worth its price.
3. **Method**
   - Task and label set
   - Dataset construction, including the boundary cases and the balance check
   - Arm selection, and why a `-low` arm needs a default twin
   - Protocol: two passes, why latency and accuracy are measured separately
   - Decoding parameters per arm kind
   - Pricing source and date, token counting, cost reconciliation
   - What was run once and what was run repeatedly, stated plainly
4. **Mathematical analysis** — the three propositions, each with a proof and a
   check against the measured data. See below.
5. **Results**
   - Accuracy with 95% Wilson intervals
   - Cost per call, per 1,000 calls, and per million input/output tokens
   - Latency P50 and P95 from the sequential pass only
   - Token usage, input and output and reasoning counted separately
6. **Which differences are real** — the pairwise distinguishability matrix. The
   section that prevents the results table from being over-read.
7. **Label-free evaluation** — the Instant Evals experiment. See below.
8. **Failure modes** — counted by kind per arm: HTTP errors, malformed output,
   invalid labels, missing usage. Rate-limit retries reported separately, because
   a rate limit is our throttle and not a property of the model.
9. **Limitations** — n=100, one run per item, one prompt, one machine, one day,
   no few-shot tuning, Jev's parallel-question advantage unmeasured.
10. **Hypotheses, scored** — the four predictions recorded in `00-overview.md`
    before the run, marked right or wrong.
11. **Appendix** — full prompts, per-arm reasoning negotiation outcomes, the cost
    reconciliation table, and a pointer to the per-record dump.

## Mathematical analysis: the three propositions

Each is stated, proved, and then checked against the run. A proposition that the
data contradicts is reported as contradicted.

**Proposition 1 — Jev's price is affine in input tokens alone.**
Claim: `cost = p_in · T_in`, with output tokens billed at zero. Proved from the
pricing model, then checked against every Jev call in the run by comparing the
provider-reported cost to the computed one. The check is already mechanised as
the reconciliation in `jevdemo/pricing.py`; this section reports its denominator.

**Proposition 2 — the resolution bound.**
Claim: at n attempts, an accuracy estimate carries a Wilson interval of a stated
width, and two arms are distinguishable only when the Newcombe interval on their
difference excludes zero. Derived in `jevdemo/stats.py`, with the derivation
reproduced in the report. Consequence, stated up front: at n=100 near 90%
accuracy the interval is roughly ±6 points, so gaps under about 8 points are not
resolvable by this experiment.

**Proposition 3 — the break-even misroute cost.**
The one that answers the buying question. With per-call cost `c` and accuracy
`α`, expected cost per ticket is `E = c + K(1 − α)` where `K` is what one misroute
costs the business. Arm *d* beats arm *c* exactly when `K > (c_d − c_c)/(α_d − α_c)`.
Two lines of algebra, and it converts the entire results table into one number per
pair. Reported against a stated range of plausible `K`, not a single guessed value.

The proofs are written out because the conclusion — that a cheaper arm can be the
correct choice even when it is genuinely less accurate — is counter-intuitive
enough that a reader is entitled to check it rather than trust it.

## Label-free evaluation: the Instant Evals experiment

LangWatch sells Instant Evals as evaluation without a labelled dataset: an
operator writes the question in plain words and a proprietary judge scores every
stored row, returning a calibrated probability. Their published claim is 97%
agreement with human labels against 91% for Claude Opus 5, at roughly $0.32 per
10,000 conversations (langwatch.ai/instant-evals, read 2026-09-20).

That is a different product from this benchmark, and the interesting question is
not which is better but whether the cheaper one would have reached the same
conclusion. This report answers it directly.

**Design.** Every bulk-pass prediction in the run is re-scored by a judge that
sees the ticket and the predicted department, the full department taxonomy, and
nothing else. The gold label never enters the prompt — `test_judge.py` asserts
this, because the experiment is void otherwise. The judge is a strong model
(`anthropic/claude-sonnet-5`) so that a negative result indicts the paradigm
rather than a weak judge.

**Three numbers reported.**

1. **Row agreement** — how often the judge's verdict matches the gold verdict,
   over the rows it actually scored, with that denominator printed.
2. **Rank agreement** — the Spearman correlation between the arm ranking the
   judge produces and the ranking the gold labels produce. This is the one that
   decides whether a team could skip labelling: a judge that is systematically
   generous but orders the arms correctly is still a usable buying instrument.
3. **Cost of the judgement** — what the label-free pass cost, against the cost of
   labelling 100 tickets by hand.

**Confound, named in advance.** `sonnet-5` is both an arm and the judge, so its
rows are self-graded. Self-judged rows are reported separately from the rest, and
if the gap between them is material it is a finding rather than a footnote.

**What this cannot show.** LangWatch's own judge is not available through this
API key, so the experiment tests the paradigm, not their model. Their 97% figure
is theirs and is reported as a claim, not reproduced. The report says so where the
number appears.

## Traceability

Every figure in the report is substituted from `results.json` by
`scripts/build_report.py`, which raises on an unknown token. No number is typed
into the prose by hand. A claim that cannot be rendered from the results file is
a claim this run did not earn.
