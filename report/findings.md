EYEBROW DevX Labs · 2026-09-20

# Routing support tickets: what each model costs, and what it gets right

## The answer

We put {{arm_count}} models on the same {{dataset_n}} support tickets and measured
three things: did it pick the right queue, what did it cost, how long did it take.

**Jev was the cheapest and the fastest, and not one of the other models was
measurably more accurate than it.** It routed 92% of tickets correctly at
{{jev_acc_cost}} per thousand and a typical {{jev_p50}} ms response. The most
expensive model in the test, `grok-4.6`, cost {{dearest_ratio}} times as much and
scored 93% — a one-point difference this test cannot tell apart from chance.

Two caveats up front, both in section 5. A hundred tickets can only resolve
accuracy differences of about {{resolution_pp}} points, so most of the ranking
below is a cost ranking, not an accuracy ranking. And for {{unrecon_count}} of
the models, what the provider billed differed from its published price by 10% to
53%, which we could not explain from this run's data.

## 1 · What we tested

One job: a support ticket arrives as free text, and something has to decide which
queue it belongs in. Get it wrong and a person has to re-route it, so the ticket
waits and someone is paid twice for it.

Five queues. These are the exact definitions every model was given, word for word:

{{CATEGORY_TABLE}}

The question we set out to answer is not "which model is best." It is: **for a job
this ordinary, how much does accuracy actually cost?**

## 2 · How we tested it

{{dataset_n}} tickets, 20 in each of the five categories, written and labelled by
hand before any model saw them. Every model got the identical {{dataset_n}}
tickets, the identical definitions, and the identical instruction to reply with
one label and nothing else. {{record_count}} calls in total.

Here is one real ticket from each category, exactly as sent:

{{EXAMPLES_TABLE}}

A model's answer is correct only if it matches the hand label exactly. A reply we
could not read as one of the five labels counts as wrong, not as a retry — a
router that returns something unusable has failed to route, whatever the reason.

Each model was run twice over the set: once measuring accuracy and cost, once
measuring response time on a smaller sample so that timing was never taken while
the connection was busy. Full protocol in Appendix A.

## 3 · The analysis

{{HEADLINE_TABLE}}

**Read the table as a cost ranking with an accuracy floor, not as a leaderboard.**
Every model in it lands between 79% and 97%, and with {{dataset_n}} tickets the
test can only resolve gaps of roughly {{resolution_pp}} points. Seventeen of the
nineteen sit inside that band of each other. What separates them by a factor of
171 is price.

Three things the table says that are not close calls:

1. **Jev is the cheapest and the fastest.** {{jev_acc_cost}} per thousand tickets
   and {{jev_p50}} ms typical, at 92% correct. The next cheapest,
   `glm-5.3-flash-low`, costs 13% more and answers in 922 ms.
2. **Nothing in the test is measurably more accurate than Jev.** `deepseek-v4.1-flash-low`
   scored the highest raw accuracy at 97% against Jev's 92%, but the interval on
   that five-point difference still contains zero. Appendix B has every pair.
3. **The expensive models did not buy accuracy.** `grok-4.6` at
   {{dearest_ratio}}× Jev's price scored 93%. `opus-5` at 132× scored 94%.
   `sonnet-5`, the best of the premium models at 96%, costs 29× and is five times
   slower.

The one real accuracy signal is at the bottom: `mistral-small-low` at 79% is the
only model this sample can call worse than the rest, and it also lost 16 of its
100 calls to HTTP errors.

## 4 · Cost against effectiveness

{{COST_CHART}}

The x-axis is logarithmic because the price range spans more than two orders of
magnitude; the y-axis is plain accuracy. **A good result sits high and to the
left. Jev is the leftmost point on the chart, and nothing above it is far enough
above to be real.**

The shape to notice is that the cloud is flat. Moving right by a factor of a
hundred does not move a point up. If cost bought accuracy on this task, the
points would trend upward to the right; they do not trend at all.

The four points on the far right — `kimi-k3`, `grok-4.6-low`, `opus-5`,
`grok-4.6` — are between 96× and 171× the price of the leftmost point, for
accuracies of 92% to 94% against its 92%.

## 5 · What could be wrong with this

Five things that could change the conclusion, in the order they would matter:

1. **{{dataset_n}} tickets is a small sample.** It resolves accuracy differences
   of about {{resolution_pp}} points and no finer. Read the ranking as "these are
   all in the same band" rather than as an order. Ten times the data would be
   needed to separate 92% from 96%.
2. **{{unrecon_count}} of the models did not reconcile on cost.** What the
   provider billed differed from its published per-token price by between 10% and
   53%. We checked both obvious explanations — a missing cost field and a stale
   price table — against the live API, and neither holds: the field is always
   present, and all 20 prices match the provider's current list exactly. We could
   not establish the mechanism from this run's stored data, so it is named here
   rather than explained. Cost figures for those models should be treated as
   list-price estimates. Jev's own cost does reconcile.
3. **One run, one day, one prompt.** No model was run twice, so none of this
   separates model behaviour from run-to-run variation. Latency in particular was
   measured from one machine over one network at one time of day.
4. **`llama-4-scout` produced no result at all** — all 100 calls returned an HTTP
   error, so it is absent from every table and from the chart. Four other models
   lost calls too: `mistral-small-low` 16, `mistral-small` 4, and four models lost
   2–3 calls to unreadable output. Those losses count as wrong answers, which is
   the conservative choice but does penalise those models twice.
5. **The tickets were written for this test.** They are realistic but not drawn
   from a production queue, and they are cleanly separable by construction — real
   tickets are messier and the accuracy numbers here are likely optimistic across
   the board.

## Appendix A · Protocol in full

### Task and label set

A free-text support ticket is routed to exactly one of five departments:
`billing`, `technical`, `account`, `sales`, `other`. Fixed label set, short
input, no prose required in the answer. This is the canonical shape where a
typed decision model should beat a chat model, and therefore the fairest place
to test the claim rather than the most flattering one.

`other` exists because a model cannot pick a value it was never offered.
Dropping it would add the same errors to every model and separate nothing.

`jevdemo/labels.py` is the single source of truth for the five definitions. The
Jev model renders them to the endpoint's `criteria` field; the chat models render
them into the system prompt. Neither restates a definition in its own words, and
a test asserts both renderings carry the same five keys and the same definition
strings. Without that test the comparison would be between two different tasks.

### Dataset construction

{{dataset_n}} synthetic tickets, 20 per label, hand-written and hand-labelled
for this study. Every ticket is 15 to 60 words, and no ticket contains its own
label as a word — "I have a billing question" would test string matching rather
than classification.

25 tickets sit deliberately on a class boundary, because that is the only place
the models disagree and therefore the only place the comparison carries
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

### Model selection

{{arm_count}} models, one provider, one API key, one run. Models that expose a
reasoning control on this endpoint get two models, default and lowest-reasoning,
because the gap between those two settings is a configuration choice teams make
without ever seeing its cost.

A `-low` model with no default twin measures nothing, because there is nothing to
subtract it from. The roster was corrected during implementation for exactly
this reason: four models had been listed as `-low` only, and their default models
were added, taking the roster to {{arm_count}} models.

There is no single switch for reasoning. `{"enabled": false}` returns HTTP 400
on at least one model in the roster — verbatim: reasoning is mandatory on that
endpoint and cannot be disabled — so each low model negotiates, trying the
disable first and falling back to minimum effort. Which setting actually
answered is recorded per model and printed in the appendix, because the two are
not the same setting and reporting them as one would be a quiet lie.

### Protocol

Two passes, deliberately separated.

1. **Latency pass** — 20 tickets per model, strictly sequential, one request in
   flight at a time. This is the only source of the reported P50 and P95. A
   timing taken under concurrency measures the worker pool, not the model.
2. **Bulk pass** — {{dataset_n}} tickets per model, four workers. This is the only
   source of accuracy, tokens and cost.

{{record_count}} calls in total across the two passes, {{total_cost}} of spend.
Every ticket is sent once per model. Nothing is retried for a better answer; the
only retry is on HTTP 429, which is our throughput limit and not a property of
the model, and its latency is excluded.

### Decoding parameters

Chat models: `temperature: 0`, `max_tokens: 512`, JSON response format, and
`usage: {"include": true}` so that cost and reasoning-token counts come back.
The 512 matters — an earlier probe at 64 truncated a reasoning model mid-answer
and scored it as a parse failure, which would have been a measurement of the cap
rather than of the model. The Jev model takes no decoding parameters; the endpoint
returns a typed choice, a probability distribution over the criteria, and a
confidence.

### Pricing, token counting, and reconciliation

Prices are read from the provider's model listing on 2026-09-20 and stored as
integer micro-dollars per million tokens. All money in this report is integer
minor units end to end; no cost is ever held in a float.

Cost is computed from the model's tariff and the returned token counts, then
reconciled against the provider's own reported cost for every call, at a
tolerance of {{recon_tolerance}} micro-dollars. Reconciliation status is reported
per model in the appendix rather than assumed: {{recon_arms}} models reconcile.

### What was run once, and what was run repeatedly

Stated plainly, because it bounds every claim that follows. Each ticket was sent
**once** per model. There is **one** run, on **one** machine, on **one** day, with
**one** prompt per model kind. Accuracy is therefore a single-sample estimate per
ticket, and its uncertainty is the binomial uncertainty of {{dataset_n}} draws —
quantified in the next section, not hand-waved. Latency is 20 samples per model and
is reported as P50 and P95 only; a mean would be dominated by the tail and a
single slow call would move it.

## Appendix B · Which differences are real

Accuracy differences are only meaningful if they survive the sample size. This
matrix applies Newcombe's method 10 to the difference between every pair of
models: **better** means the 95% interval on the difference lies entirely above
zero, **worse** entirely below, and **tie** means it contains zero and the two
cannot be told apart at {{dataset_n}} tickets.

Of {{pair_count}} pairs, {{resolved_pairs}} resolve and {{unresolved_pairs}} do
not. Jev's row is all ties: no model in the test is distinguishable from it on
accuracy in either direction.

{{DISTINGUISHABILITY_MATRIX}}

## Appendix C · What one wrong route has to cost

**A dearer model is only worth it if a wrong route costs enough.** Write `c` for
the cost of a call and `α` for its accuracy. Routing a ticket costs `c` plus, with
probability `1 − α`, the cost `K` of putting a misrouted ticket right. A dearer
model `d` beats the cheapest one `c` exactly when `K` exceeds
`(c_d − c_c) / (α_d − α_c)`. That threshold is the table's last column.

The thresholds are the answer to "what would have to be true for the expensive
option to be correct." To justify `sonnet-5` over Jev, a single misroute would
have to cost about $0.013. To justify `grok-4.6`, about $0.31. Those are small
numbers — a human touch on a misrouted ticket plausibly costs more — which is why
the accuracy caveat matters more than the arithmetic: **every one of these
thresholds is computed from an accuracy gap the test cannot confirm is real.**
Rows marked dominated cost more without being more accurate even on the raw
numbers, so no misroute cost justifies them.

{{BREAK_EVEN_TABLE}}

## Appendix D · The full measurement table

{{RESULTS_TABLE}}

## Appendix E · Failures

{{FAILURES_TABLE}}

## Appendix F · Grading without an answer key

Every number above depends on a hand-labelled answer key, which does not exist for
a live production queue. So we ran the same predictions past a label-free grader —
the approach LangWatch sells as Instant Evals — where a separate model is shown the
ticket and the proposed queue and asked whether the routing is right, with no
answer key at all.

The question is whether that grader's verdict tracks the hand labels well enough to
stand in for them. The table reports, per model, what the grader scored against what
the answer key scored, and the correlation between the two across models.

{{JUDGE_TABLE}}

## Appendix G · The mathematics

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
two models are distinguishable only when the interval on their *difference* excludes
zero.

**Proof.** Let `X ~ Binomial(n, α)` be the correct count and `p = X/n`. The Wilson
score interval inverts the score test `|p − α| / sqrt(α(1−α)/n) ≤ z`, which is
quadratic in `α`; solving gives

    centre = (p + z²/2n) / (1 + z²/n)
    half   = z/(1 + z²/n) · sqrt( p(1−p)/n + z²/4n² )

The Wald interval `p ± z·sqrt(p(1−p)/n)` is not used: at `p = 1` it has zero
width, asserting certainty from a finite sample, and near the boundary it extends
outside `[0,1]`. Both failures occur exactly where this roster lives.

For two independent models, Newcombe's method 10 combines each model's Wilson bounds
into an interval on the difference `δ = α_a − α_b`:

    δ_lo = (p_a − p_b) − sqrt( (p_a − l_a)² + (u_b − p_b)² )
    δ_hi = (p_a − p_b) + sqrt( (u_a − p_a)² + (p_b − l_b)² )

Model *a* is more accurate than model *b* at 95% confidence precisely when `δ_lo > 0`. ∎

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

**Claim.** Given a cheap model `c` and a dearer, more accurate model `d`, there is a
single threshold `K*` — the cost of one misroute — above which `d` is the rational
choice and below which `c` is.

**Proof.** Route one ticket with model `a`. The expected cost is the call plus the
expected cost of being wrong:

    E_a = c_a + K · (1 − α_a)

where `K` is what one misrouted ticket costs the business: the re-route, the
second touch, the delay. Model `d` is preferable exactly when `E_d < E_c`:

    c_d + K(1 − α_d)  <  c_c + K(1 − α_c)
    c_d − c_c         <  K(α_d − α_c)

Since `α_d > α_c` by assumption, dividing preserves the inequality:

    K  >  (c_d − c_c) / (α_d − α_c)  =  K*                                    ∎

**Three corollaries the table uses.**

1. If `α_d ≤ α_c` the model is *dominated* — dearer and no better — and no value of
   `K` rescues it. Reported as a dash, not a number.
2. If `c_d ≤ c_c` while `α_d > α_c`, then `K* = 0`: the model wins even when
   misroutes are free. Strict dominance in the other direction.
3. `K*` scales with the cost gap and inversely with the accuracy gap. Two models
   separated by a large price and a small accuracy difference produce an enormous
   `K*` — which is the quantitative form of "you are paying for nothing."

**Why this is the decision rule.** `K` is the only quantity in the model the
benchmark cannot measure, and it is the one the business already knows. The
report therefore does not guess a value; it reports `K*` per model and lets the
reader compare it against their own cost of a misroute. Every other number in the
document feeds this one.

**Honest caveat.** Proposition 3 treats `α` as known. By Proposition 2 it is not —
it is an interval. Where the accuracy gap between two models is not resolvable,
`K*` inherits that uncertainty and is reported with the gap's interval rather than
as a point estimate. An unresolvable accuracy gap makes `K*` unbounded in the
limit, which is itself the finding: if you cannot tell the models apart on accuracy,
buy the cheaper one.
