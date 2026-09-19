from jevdemo import metrics
from jevdemo.arms import ARMS, Arm, by_name
from jevdemo.metrics import BULK_PASS, LATENCY_PASS, ArmMetrics, Record, aggregate
from jevdemo.result import Prediction, failed

ARM_MAP = {a.name: a for a in ARMS}


def ok(label="billing", *, tin=100, tout=10, reasoning=0, cost=5):
    return Prediction(label, tin, tout, reasoning, cost, None, None, "")


def rec(arm="sonnet-5", ticket="t001", pass_name=BULK_PASS, gold="billing", pred=None, elapsed=1.0):
    return Record(arm, ticket, pass_name, gold, pred or ok(), elapsed)


def test_an_empty_record_set_produces_no_arms_rather_than_a_zero_row():
    assert aggregate([], ARM_MAP) == {}


def test_accuracy_counts_failures_against_the_arm():
    records = [rec(), rec(pred=ok("technical")), rec(pred=failed("malformed_output"))]
    m = aggregate(records, ARM_MAP)["sonnet-5"]
    assert m.attempted == 3
    assert m.scored == 2
    assert m.correct == 1
    assert m.accuracy == 1 / 3
    assert m.failures == {"malformed_output": 1}


def test_latency_reads_only_the_sequential_pass():
    records = [
        rec(pass_name=LATENCY_PASS, elapsed=2.0),
        rec(pass_name=LATENCY_PASS, elapsed=4.0),
        rec(pass_name=BULK_PASS, elapsed=99.0),
    ]
    m = aggregate(records, ARM_MAP)["sonnet-5"]
    assert m.latencies == [2.0, 4.0]
    assert m.p50_latency_ms == 2.0


def test_the_latency_pass_contributes_no_tokens_and_no_cost():
    m = aggregate([rec(pass_name=LATENCY_PASS)], ARM_MAP)["sonnet-5"]
    assert m.input_tokens == 0
    assert m.computed_cost_micro == 0
    assert m.attempted == 0
    assert m.accuracy is None


def test_a_failed_latency_call_is_not_timed():
    m = aggregate(
        [rec(pass_name=LATENCY_PASS, pred=failed("http_error"), elapsed=30.0)], ARM_MAP
    )["sonnet-5"]
    assert m.latencies == []
    assert m.p50_latency_ms is None


def test_cost_is_computed_from_the_price_card_not_from_the_provider_figure():
    # sonnet-5: 2_000_000 micro per Mtok in, 10_000_000 out.
    m = aggregate([rec(pred=ok(tin=1_000_000, tout=1_000_000, cost=1))], ARM_MAP)["sonnet-5"]
    assert m.computed_cost_micro == 12_000_000
    assert m.reported_cost_micro == 1
    assert m.reconciliation().ok is False


def test_one_missing_provider_cost_makes_the_whole_arm_unreconciled():
    m = aggregate([rec(), rec(pred=ok(cost=None))], ARM_MAP)["sonnet-5"]
    assert m.reported_cost_complete is False
    assert m.reconciliation().ok is False


def test_reasoning_tokens_accumulate_separately_from_output_tokens():
    m = aggregate([rec(pred=ok(tout=200, reasoning=180))], ARM_MAP)["sonnet-5"]
    assert m.output_tokens == 200
    assert m.reasoning_tokens == 180


def test_cost_per_thousand_scales_from_the_attempted_denominator():
    records = [rec(pred=ok(tin=1000, tout=0, cost=2)) for _ in range(10)]
    m = aggregate(records, ARM_MAP)["sonnet-5"]
    assert m.computed_cost_micro == 20_000  # 10 x 1000 tok x 2_000_000/Mtok
    assert m.cost_per_1000_micro == 2_000_000


def test_arms_are_kept_apart():
    out = aggregate([rec(arm="jev"), rec(arm="opus-5")], ARM_MAP)
    assert sorted(out) == ["jev", "opus-5"]
    assert out["jev"].kind == "jev"


# --- an arm that never answered -------------------------------------------
#
# llama-4-scout returned HTTP 404 "No endpoints found" on all 100 calls of the
# first full run. Dividing 0 correct by 100 attempts yields 0.0, and a table
# printing "0%" makes a claim about the model that this run did not measure.
# Accuracy over zero usable observations is undefined, not zero.


def _arm_metrics_with(scored: int, correct: int, http_errors: int):
    from jevdemo.metrics import ArmMetrics
    m = ArmMetrics(spec=by_name("sonnet-5"))
    m.scored, m.correct = scored, correct
    if http_errors:
        m.failures["http_error"] = http_errors
    return m


def test_an_arm_that_never_answered_has_no_accuracy():
    m = _arm_metrics_with(scored=0, correct=0, http_errors=100)
    assert m.measured is False
    assert m.accuracy is None


def test_failures_still_count_against_an_arm_that_did_answer():
    """A partial failure is not free — it stays in the denominator."""
    m = _arm_metrics_with(scored=83, correct=73, http_errors=17)
    assert m.measured is True
    assert m.attempted == 100
    assert m.accuracy == 0.73


# --- money is priced once, never per call -----------------------------------
# Rounding each call to a whole micro-dollar and summing inflates any arm whose
# per-call cost sits near a half. At ~4.5 micro-USD a call, half-up rounding
# overstated the published cost per 1,000 by about ten percent, and it made the
# reconciliation compare two different quantities.


def _arm_with_price(in_price: int, out_price: int):
    return Arm(name="t", model="m", kind="chat", reasoning=None,
               price_in_micro_per_mtok=in_price, price_out_micro_per_mtok=out_price)


def test_cost_is_priced_from_totals_not_summed_per_call():
    # 4.5 micro-USD a call: per-call half-up would give 5 each, 10 in total.
    arm = _arm_with_price(in_price=4_500_000, out_price=0)
    m = ArmMetrics(spec=arm)
    for _ in range(2):
        m.scored += 1
        m.correct += 1
        m.input_tokens += 1
    assert m.computed_cost_micro == 9


def test_priced_once_matches_the_reconciliation_computation():
    arm = _arm_with_price(in_price=4_500_000, out_price=1_000_000)
    m = ArmMetrics(spec=arm)
    m.scored, m.correct = 3, 3
    m.input_tokens, m.output_tokens = 7, 5
    m.reported_cost_micro = m.computed_cost_micro
    assert m.reconciliation().ok
