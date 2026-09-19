"""A4 — money is integers, and the provider's cost is checked, not trusted."""
import pytest

from jevdemo import arms, pricing
from jevdemo.arms import by_name
from jevdemo.pricing import compute_micro, reconcile

JEV = arms.by_name("jev")
OPUS = arms.by_name("opus-5")


def test_the_jev_case_reconciles_exactly():
    """PROBE-RESULTS P1: 395 input tokens, output free, reported $1.659e-05."""
    computed = pricing.compute_micro(JEV, input_tokens=395, output_tokens=52)
    assert computed == 17  # 395 * 42_000 / 1e6 = 16.59 micro-dollars
    check = pricing.reconcile(JEV, input_tokens=395, output_tokens=52, reported_cost=1.659e-05)
    assert check.ok
    assert check.delta_micro == 0


def test_output_tokens_are_free_on_jev_at_any_volume():
    assert pricing.compute_micro(JEV, 0, 1_000_000) == 0


def test_cost_is_an_int_never_a_float():
    value = pricing.compute_micro(OPUS, 1234, 567)
    assert type(value) is int


def test_opus_arithmetic_is_exact_at_a_million_tokens():
    """A round number makes a rounding bug visible instead of plausible."""
    assert pricing.compute_micro(OPUS, 1_000_000, 0) == 5_000_000
    assert pricing.compute_micro(OPUS, 0, 1_000_000) == 25_000_000


def test_a_discrepancy_is_reported_not_swallowed():
    check = pricing.reconcile(JEV, input_tokens=395, output_tokens=52, reported_cost=2.0e-05)
    assert not check.ok
    assert check.delta_micro == 3  # 20 reported vs 17 computed
    assert "jev" in check.message


def test_a_discrepancy_within_tolerance_still_passes():
    check = pricing.reconcile(JEV, input_tokens=395, output_tokens=52, reported_cost=1.85e-05)
    assert check.ok  # 19 vs 17 == 2, at the tolerance boundary


def test_a_missing_reported_cost_is_a_warning_not_a_crash():
    """Spec 06: `usage` absent must not take the run down."""
    check = pricing.reconcile(JEV, input_tokens=395, output_tokens=52, reported_cost=None)
    assert not check.ok
    assert "not reported" in check.message
    assert check.computed_micro == 17


def test_negative_token_counts_raise():
    """A negative count means the parser is wrong; billing it would hide that."""
    with pytest.raises(ValueError):
        pricing.compute_micro(JEV, -1, 0)


def test_per_thousand_extrapolation_uses_integers_throughout():
    assert pricing.per_thousand_micro(total_micro=1700, n=100) == 17_000
    with pytest.raises(ValueError):
        pricing.per_thousand_micro(total_micro=1700, n=0)


def test_micro_to_usd_string_is_stable_at_four_decimals():
    assert pricing.usd(17_000) == "0.0170"
    assert pricing.usd(0) == "0.0000"
    assert pricing.usd(441_200) == "0.4412"


# --- tolerance scales with the number of calls ------------------------------
# A provider rounds each call it bills. Over a 120-call arm that drift compounds,
# so an absolute two-micro-dollar tolerance flagged four arms whose totals agreed
# to within four parts in ten thousand. The tolerance is per call, not per total.


def test_tolerance_absorbs_per_call_rounding_across_many_calls():
    arm = by_name("sonnet-5")
    computed = compute_micro(arm, 20_000, 1_300)
    # 15 micro-USD adrift over 120 calls is an eighth of a micro-dollar per call.
    r = reconcile(arm, 20_000, 1_300, (computed + 15) / 1_000_000, calls=120)
    assert r.ok


def test_tolerance_still_catches_a_real_billing_difference():
    arm = by_name("sonnet-5")
    computed = compute_micro(arm, 20_000, 1_300)
    r = reconcile(arm, 20_000, 1_300, computed * 1.2 / 1_000_000, calls=120)
    assert not r.ok


def test_a_single_call_keeps_the_tight_tolerance():
    arm = by_name("sonnet-5")
    computed = compute_micro(arm, 200, 13)
    assert reconcile(arm, 200, 13, (computed + 2) / 1_000_000, calls=1).ok
    assert not reconcile(arm, 200, 13, (computed + 40) / 1_000_000, calls=1).ok


def test_calls_must_be_positive():
    arm = by_name("sonnet-5")
    with pytest.raises(ValueError):
        reconcile(arm, 200, 13, 0.0002, calls=0)
