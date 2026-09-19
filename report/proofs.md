## Mathematical analysis

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
