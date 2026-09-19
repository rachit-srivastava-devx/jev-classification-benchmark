EYEBROW DevX Labs · Research note · 2026-09-20

# What a typed decision model costs, and when it is the right purchase

## Abstract

{{ABSTRACT}}

## 1 · The decision rule

A benchmark that reports accuracy and price without joining them leaves the
reader to do the arithmetic that decides the purchase. This report does that
arithmetic first, because the answer inverts the usual instinct.

Let an arm have per-call price `c` and accuracy `α`, and let `K` be what one
misrouted ticket costs the business — the rework, the delay, the escalation.
Expected cost per ticket is then `E = c + K(1 − α)`. A dearer arm *d* beats a
cheaper arm *c* precisely when

    K > (c_d − c_c) / (α_d − α_c)

That right-hand side is a price: the value one misroute must have before the
dearer arm earns its cost. It is denoted `K*` and it is the only number in this
report a reader has to act on. The benchmark supplies `c` and `α`. The business
already knows `K`. The table below joins them.

{{BREAK_EVEN_TABLE}}

Read a row as a conditional, not a verdict. If one misroute costs your business
more than the figure in the row, that arm is worth its price. If less, it is not.
Rows shown in grey are rows where this run cannot resolve the accuracy gap; there
the threshold is a lower bound on what you would be paying for, and section 6
says which gaps those are.

## 2 · Method

### Task and label set

A free-text support ticket is routed to exactly one of five departments:
`billing`, `technical`, `account`, `sales`, `other`. Fixed label set, short
input, no prose required in the answer. This is the canonical shape where a
typed decision model should beat a chat model, and therefore the fairest place
to test the claim rather than the most flattering one.

`other` exists because a model cannot pick a value it was never offered.
Dropping it would add the same errors to every arm and separate nothing.

`jevdemo/labels.py` is the single source of truth for the five definitions. The
Jev arm renders them to the endpoint's `criteria` field; the chat arms render
them into the system prompt. Neither restates a definition in its own words, and
a test asserts both renderings carry the same five keys and the same definition
strings. Without that test the comparison would be between two different tasks.

### Dataset construction

{{dataset_n}} synthetic tickets, 20 per label, hand-written and hand-labelled
for this study. Every ticket is 15 to 60 words, and no ticket contains its own
label as a word — "I have a billing question" would test string matching rather
than classification.

25 tickets sit deliberately on a class boundary, because that is the only place
the arms disagree and therefore the only place the comparison carries
information. The planted ambiguities are of this kind:

- A failed payment caused by an expired card is `billing`, not `technical`.
- An account locked after five bad passwords is `account`, not `technical`.
- An existing customer asking what the next tier costs is `sales`, not `billing`.
- A refund request that is really a cancellation threat is `billing`; the churn
  signal is not a label in this task.

The remaining 75 tickets are not there to be hard. They are there to make
accuracy readable: a roster that scores near-perfectly on the easy three
quarters and splits on the boundary quarter is a roster whose differences are
legible. A validator enforces the counts, the per-label balance, the word-length
bounds and the requirement that every boundary case carries a written
justification; it is tested against deliberately malformed datasets, because a
validator that never fails is theatre.

### Arm selection

{{arm_count}} arms, one provider, one API key, one run. Models that expose a
reasoning control on this endpoint get two arms, default and lowest-reasoning,
because the gap between those two settings is a configuration choice teams make
without ever seeing its cost.

A `-low` arm with no default twin measures nothing, because there is nothing to
subtract it from. The roster was corrected during implementation for exactly
this reason: four models had been listed as `-low` only, and their default arms
were added, taking the roster to {{arm_count}} arms.

There is no single switch for reasoning. `{"enabled": false}` returns HTTP 400
on at least one model in the roster — verbatim: reasoning is mandatory on that
endpoint and cannot be disabled — so each low arm negotiates, trying the
disable first and falling back to minimum effort. Which setting actually
answered is recorded per arm and printed in the appendix, because the two are
not the same setting and reporting them as one would be a quiet lie.

### Protocol

Two passes, deliberately separated.

1. **Latency pass** — 20 tickets per arm, strictly sequential, one request in
   flight at a time. This is the only source of the reported P50 and P95. A
   timing taken under concurrency measures the worker pool, not the model.
2. **Bulk pass** — {{dataset_n}} tickets per arm, four workers. This is the only
   source of accuracy, tokens and cost.

{{record_count}} calls in total across the two passes, {{total_cost}} of spend.
Every ticket is sent once per arm. Nothing is retried for a better answer; the
only retry is on HTTP 429, which is our throughput limit and not a property of
the model, and its latency is excluded.

### Decoding parameters

Chat arms: `temperature: 0`, `max_tokens: 512`, JSON response format, and
`usage: {"include": true}` so that cost and reasoning-token counts come back.
The 512 matters — an earlier probe at 64 truncated a reasoning model mid-answer
and scored it as a parse failure, which would have been a measurement of the cap
rather than of the model. The Jev arm takes no decoding parameters; the endpoint
returns a typed choice, a probability distribution over the criteria, and a
confidence.

### Pricing, token counting, and reconciliation

Prices are read from the provider's model listing on 2026-09-20 and stored as
integer micro-dollars per million tokens. All money in this report is integer
minor units end to end; no cost is ever held in a float.

Cost is computed from the arm's tariff and the returned token counts, then
reconciled against the provider's own reported cost for every call, at a
tolerance of {{recon_tolerance}} micro-dollars. Reconciliation status is reported
per arm in the appendix rather than assumed: {{recon_arms}} arms reconcile.

### What was run once, and what was run repeatedly

Stated plainly, because it bounds every claim that follows. Each ticket was sent
**once** per arm. There is **one** run, on **one** machine, on **one** day, with
**one** prompt per arm kind. Accuracy is therefore a single-sample estimate per
ticket, and its uncertainty is the binomial uncertainty of {{dataset_n}} draws —
quantified in the next section, not hand-waved. Latency is 20 samples per arm and
is reported as P50 and P95 only; a mean would be dominated by the tail and a
single slow call would move it.

## 4 · Mathematical analysis

Three propositions carry the report. Each is stated, proved, and then checked
against the run. The algebra is written out rather than cited because the
conclusion it reaches — that a measurably less accurate model can be the correct
purchase — is counter-intuitive enough that a reader is entitled to verify it.

### Proposition 1 — Jev's price is affine in input tokens alone

**Claim.** For the decisions endpoint, cost is `p_in · T_in / 10^6`, with output
tokens billed at zero.

**Proof.** The general two-part tariff is

    C = (p_in · T_in + p_out · T_out) / 10^6

with `p_in`, `p_out` the prices per million tokens. The published tariff for
`typesafe/jev-1.13` sets `p_out = 0`, which annihilates the second term for every
`T_out`, leaving `C = p_in · T_in / 10^6`. Since `p_in` is constant, `C` is linear
in `T_in` and independent of the work done producing the answer. ∎

**Consequence.** Jev's cost is bounded by the length of the ticket alone. A harder
question does not cost more, which is the structural difference between a
decisions endpoint and a chat model: on a chat model, `T_out` is chosen by the
model at inference time, so the buyer does not control the bill.

**Check.** Verified against every Jev call in the run by reconciling the
provider-reported cost with `p_in · T_in`, at a tolerance of {{recon_tolerance}}
micro-dollars. Result: {{jev_recon}}.

### Proposition 2 — the resolution bound

**Claim.** An accuracy measured over `n` attempts is an interval, not a point, and
two arms are distinguishable only when the interval on their *difference* excludes
zero.

**Proof.** Let `X ~ Binomial(n, α)` be the correct count and `p = X/n`. The Wilson
score interval inverts the score test `|p − α| / sqrt(α(1−α)/n) ≤ z`, which is
quadratic in `α`; solving gives

    centre = (p + z²/2n) / (1 + z²/n)
    half   = z/(1 + z²/n) · sqrt( p(1−p)/n + z²/4n² )

The Wald interval `p ± z·sqrt(p(1−p)/n)` is not used: at `p = 1` it has zero
width, asserting certainty from a finite sample, and near the boundary it extends
outside `[0,1]`. Both failures occur exactly where this roster lives.

For two independent arms, Newcombe's method 10 combines each arm's Wilson bounds
into an interval on the difference `δ = α_a − α_b`:

    δ_lo = (p_a − p_b) − sqrt( (p_a − l_a)² + (u_b − p_b)² )
    δ_hi = (p_a − p_b) + sqrt( (u_a − p_a)² + (p_b − l_b)² )

Arm *a* is more accurate than arm *b* at 95% confidence precisely when `δ_lo > 0`. ∎

**Consequence, stated before the results.** At `n = {{dataset_n}}` and accuracy near
90%, the half-width is about {{ci_halfwidth_pp}} percentage points. Differences
smaller than roughly {{resolution_pp}} points are therefore not resolvable by this
experiment, no matter how the table is sorted. The results section reports which
pairs clear that bar and which do not; a higher percentage that does not clear it
is reported as a tie.

This is a limitation of the sample size, not of the models. It was known before
the run and is the reason the report leads with cost and latency, where the gaps
span orders of magnitude and the measurement is not sample-limited.

### Proposition 3 — the break-even misroute cost

**Claim.** Given a cheap arm `c` and a dearer, more accurate arm `d`, there is a
single threshold `K*` — the cost of one misroute — above which `d` is the rational
choice and below which `c` is.

**Proof.** Route one ticket with arm `a`. The expected cost is the call plus the
expected cost of being wrong:

    E_a = c_a + K · (1 − α_a)

where `K` is what one misrouted ticket costs the business: the re-route, the
second touch, the delay. Arm `d` is preferable exactly when `E_d < E_c`:

    c_d + K(1 − α_d)  <  c_c + K(1 − α_c)
    c_d − c_c         <  K(α_d − α_c)

Since `α_d > α_c` by assumption, dividing preserves the inequality:

    K  >  (c_d − c_c) / (α_d − α_c)  =  K*                                    ∎

**Three corollaries the table uses.**

1. If `α_d ≤ α_c` the arm is *dominated* — dearer and no better — and no value of
   `K` rescues it. Reported as a dash, not a number.
2. If `c_d ≤ c_c` while `α_d > α_c`, then `K* = 0`: the arm wins even when
   misroutes are free. Strict dominance in the other direction.
3. `K*` scales with the cost gap and inversely with the accuracy gap. Two arms
   separated by a large price and a small accuracy difference produce an enormous
   `K*` — which is the quantitative form of "you are paying for nothing."

**Why this is the decision rule.** `K` is the only quantity in the model the
benchmark cannot measure, and it is the one the business already knows. The
report therefore does not guess a value; it reports `K*` per arm and lets the
reader compare it against their own cost of a misroute. Every other number in the
document feeds this one.

**Honest caveat.** Proposition 3 treats `α` as known. By Proposition 2 it is not —
it is an interval. Where the accuracy gap between two arms is not resolvable,
`K*` inherits that uncertainty and is reported with the gap's interval rather than
as a point estimate. An unresolvable accuracy gap makes `K*` unbounded in the
limit, which is itself the finding: if you cannot tell the arms apart on accuracy,
buy the cheaper one.


## 5 · Results

{{RESULTS_TABLE}}

{{RESULTS_PROSE}}

## 6 · Which differences are real

The table above is sorted, and a sorted table invites a reader to treat rank as
fact. At {{dataset_n}} tickets it is not. This section is what prevents the
results from being over-read.

Every ordered pair of arms is tested with a 95% interval on the difference of two
proportions (Newcombe method 10). A pair is called only when that interval
excludes zero.

{{DISTINGUISHABILITY_MATRIX}}

{{MATRIX_PROSE}}

## 7 · Label-free evaluation

LangWatch sells Instant Evals as evaluation without a labelled dataset: an
operator writes the question in plain words and a judge scores every stored row,
returning a calibrated probability. Their published claim is 97% agreement with
human labels against 91% for a strong general model, at roughly $0.32 per 10,000
conversations (langwatch.ai/instant-evals, read 2026-09-20). That figure is
theirs and is reported here as a claim, not reproduced.

The interesting question is not which product is better. It is whether the
label-free method would have reached the same conclusion as the labelled one —
because if it would, the {{dataset_n}} hand-labelled tickets that made this
report possible were avoidable work.

**Design.** Every bulk-pass prediction in the run is re-scored by a judge that
sees the ticket, the predicted department, and the full department taxonomy, and
nothing else. The gold label never enters the prompt; a test asserts this,
because the experiment is void otherwise. The judge is a strong model so that a
negative result indicts the paradigm rather than a weak judge.

**The confound, named in advance.** One arm in the roster is also the judge, so
its rows are self-graded. Those rows are marked and excluded from the agreement
figure. If the gap between self-graded and peer-graded rows is material, that is
a finding and not a footnote.

{{JUDGE_TABLE}}

{{JUDGE_PROSE}}

## 8 · Failure modes

Failures count into the denominator of accuracy, so an arm that returns nothing
usable is not rewarded for silence. This table is the audit trail for any arm
whose accuracy is below its correct-answer rate.

{{FAILURES_TABLE}}

## 9 · Limitations

Stated at full strength, because a limitation discovered by the reader costs more
than one disclosed by the author.

- **{{dataset_n}} tickets bounds everything.** The widest 95% interval at this
  sample size has a half-width of about {{ci_halfwidth_pp}} percentage points,
  and gaps below roughly {{resolution_pp}} points near the top of the roster are
  not resolvable at all. Of {{pair_count}} arm pairs, {{resolved_pairs}} are
  resolved and {{unresolved_pairs}} are not.
- **One run, one day, one machine.** Latency carries this machine's network floor,
  and provider-side load on 2026-09-20 is not controlled for. The cost and token
  figures do not have this problem; they are properties of the tariff and the
  request.
- **One prompt per arm kind.** No few-shot examples, no prompt search, no
  per-model tuning. A tuned prompt would move accuracy and the report does not
  claim otherwise.
- **The dataset is synthetic.** It is balanced and deliberately boundary-loaded,
  which real ticket queues are not. Accuracy here is not a forecast of accuracy
  on a production queue.
- **Jev's parallel-question advantage is unmeasured.** The endpoint prices several
  questions at roughly the cost of one. Exercising it would have made the arms
  structurally different tasks, so it was left out. It is unmeasured upside, and
  it is named rather than claimed.

## 10 · Hypotheses, scored

Four predictions were written into the specification before the run, so that the
report could record which were wrong. A benchmark that only ever confirms is not
being run honestly.

{{HYPOTHESES}}

## 11 · Appendix

{{APPENDIX}}
