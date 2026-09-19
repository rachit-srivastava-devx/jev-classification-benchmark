"""Cost in integer micro-dollars, reconciled against what the provider billed.

Floats are converted once, at the boundary, and never afterwards. Summing 2,400
float costs and printing a total is exactly the drift this module exists to prevent.

A micro-dollar is one millionth of a US dollar. Registry prices are micro-dollars
per million tokens, so `tokens * price // 1_000_000` is micro-dollars.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arms import Arm

#: Rounding alone can put the provider's figure one micro-dollar either side of
#: ours; two absorbs that without hiding a real billing difference.
#:
#: It is a tolerance PER CALL. An arm's total is the sum of calls the provider
#: rounded individually, so over 120 calls that drift compounds; an absolute two
#: micro-dollars flagged four arms whose totals agreed to four parts in ten
#: thousand. Multiplying by the call count keeps the bound tight on one call and
#: honest on a run: a genuine billing difference is tens of percent, not tens of
#: micro-dollars.
TOLERANCE_MICRO = 2


@dataclass(frozen=True)
class Reconciliation:
    """Whether the provider's cost matches one computed from the registry."""

    arm: str
    computed_micro: int
    reported_micro: int | None
    delta_micro: int
    ok: bool
    message: str


def compute_micro(arm: Arm, input_tokens: int, output_tokens: int) -> int:
    """Cost of one call, in micro-dollars, from the registry price.

    Args:
        arm: The arm that answered.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed, reasoning included.

    Returns:
        Micro-dollars, rounded half-up to the nearest whole unit.

    Raises:
        ValueError: A token count is negative, which means the parser is wrong.
            Billing it would hide the defect behind a plausible number.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"negative token count for {arm.name}: in={input_tokens} out={output_tokens}"
        )
    total = (
        input_tokens * arm.price_in_micro_per_mtok
        + output_tokens * arm.price_out_micro_per_mtok
    )
    # Integer half-up: no float ever enters the arithmetic.
    return (total + 500_000) // 1_000_000


def reconcile(
    arm: Arm,
    input_tokens: int,
    output_tokens: int,
    reported_cost: float | None,
    calls: int = 1,
) -> Reconciliation:
    """Check the provider's cost against one computed from the registry.

    A mismatch means the roster price is stale, or the provider billed something not
    modelled — a cache read, a surcharge, a different served build. Either way the
    run says so; an unreconciled total is not published.

    Args:
        arm: The arm that answered.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        reported_cost: The provider's own figure in US dollars, or None when the
            response carried no `usage` block.
        calls: How many separately-billed calls the figures cover. The tolerance
            scales with it, because the provider rounded each one.

    Returns:
        The comparison. `ok` is False for a missing figure as well as a mismatched
        one — an absent cost is unverified, not free.

    Raises:
        ValueError: `calls` is not positive. A tolerance over zero calls would
            make every comparison pass, which is a gate that measures nothing.
    """
    if calls <= 0:
        raise ValueError(f"cannot reconcile over calls={calls}")
    tolerance = TOLERANCE_MICRO * calls
    computed = compute_micro(arm, input_tokens, output_tokens)

    if reported_cost is None:
        return Reconciliation(
            arm=arm.name,
            computed_micro=computed,
            reported_micro=None,
            delta_micro=0,
            ok=False,
            message=f"{arm.name}: provider cost not reported; computed {computed} micro-USD",
        )

    reported = round(reported_cost * 1_000_000)
    delta = abs(reported - computed)
    ok = delta <= tolerance
    return Reconciliation(
        arm=arm.name,
        computed_micro=computed,
        reported_micro=reported,
        delta_micro=delta,
        ok=ok,
        message=(
            f"{arm.name}: reconciled ({computed} micro-USD)"
            if ok
            else f"{arm.name}: provider billed {reported} micro-USD, registry computes "
            f"{computed}, delta {delta} exceeds tolerance {tolerance}"
        ),
    )


def per_thousand_micro(total_micro: int, n: int) -> int:
    """Extrapolate a run's cost to 1,000 classifications.

    Raises:
        ValueError: `n` is zero or negative. A rate computed over nothing is the
            "gate green on empty input" failure in numeric form.
    """
    if n <= 0:
        raise ValueError(f"cannot extrapolate a cost over n={n}")
    return (total_micro * 1000 + n // 2) // n


def usd(micro: int) -> str:
    """Format micro-dollars as a fixed four-decimal dollar string."""
    return f"{micro / 1_000_000:.4f}"
