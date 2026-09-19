"""The statistics the report calls proofs.

Three questions this benchmark cannot answer with a bare percentage:

1. How wide is the uncertainty on an accuracy measured over n attempts?
2. Given two arms, is the gap between them real or is it sampling noise?
3. Given a cheap arm and a dearer, more accurate one, how expensive must a
   misroute be before the dearer arm is the rational choice?

Pure functions, no I/O, no clock. Money is integer micro-dollars throughout,
matching jevdemo.pricing.
"""

from __future__ import annotations

import math

# Two-sided standard-normal quantile at 95%: Phi^-1(0.975).
Z_95 = 1.959963984540054


def _check_counts(successes: int, n: int) -> None:
    if n < 0:
        raise ValueError(f"n must not be negative, got {n}")
    if successes < 0:
        raise ValueError(f"successes must not be negative, got {successes}")
    if successes > n:
        raise ValueError(f"successes {successes} exceeds n {n}")


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal (Wald) approximation because Wald produces
    intervals that leave [0, 1] exactly where this benchmark lives: near-perfect
    accuracy on a sample of 100. At 100/100, Wald reports a width of zero, which
    would assert certainty the data does not contain.

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/4n^2 )

    An empty sample returns the whole unit interval: no data is no constraint,
    not a proportion of zero.
    """
    _check_counts(successes, n)
    if n == 0:
        return (0.0, 1.0)

    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))

    # Snap the boundaries. At successes == 0 the lower bound is exactly zero, but
    # centre - half lands on floating dust (3.5e-18), which a report would print.
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return (lower, upper)


def difference_interval(
    successes_a: int, n_a: int, successes_b: int, n_b: int, z: float = Z_95
) -> tuple[float, float]:
    """Newcombe's method 10 for the difference of two independent proportions.

    Builds each arm's Wilson interval, then combines the distances from each
    point estimate to the relevant bound:

        lower = (pa - pb) - sqrt( (pa - la)^2 + (ub - pb)^2 )
        upper = (pa - pb) + sqrt( (ua - pa)^2 + (pb - lb)^2 )

    Chosen over a two-proportion z-test because the report needs an interval —
    "the gap is somewhere between -4 and +11 points" — not a p-value. A p-value
    answers "is there an effect"; a CEO deciding which model to buy is asking
    "how big is it, and could it be zero", which is the interval's question.
    """
    _check_counts(successes_a, n_a)
    _check_counts(successes_b, n_b)
    if n_a == 0 or n_b == 0:
        return (-1.0, 1.0)

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    lo_a, hi_a = wilson_interval(successes_a, n_a, z)
    lo_b, hi_b = wilson_interval(successes_b, n_b, z)

    delta = p_a - p_b
    lower = delta - math.sqrt((p_a - lo_a) ** 2 + (hi_b - p_b) ** 2)
    upper = delta + math.sqrt((hi_a - p_a) ** 2 + (p_b - lo_b) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


def distinguishable(
    successes_a: int, n_a: int, successes_b: int, n_b: int, z: float = Z_95
) -> bool:
    """True only when the difference interval excludes zero.

    This is the gate every accuracy comparison in the report passes through. An
    arm is not 'more accurate' because its percentage is higher; it is more
    accurate when the interval on the gap does not contain zero.
    """
    lower, upper = difference_interval(successes_a, n_a, successes_b, n_b, z)
    return lower > 0.0 or upper < 0.0


def break_even_micro(
    cost_cheap_micro: int,
    accuracy_cheap: float,
    cost_dear_micro: int,
    accuracy_dear: float,
) -> int | None:
    """The misroute cost at which the dearer arm starts paying for itself.

    Model: routing one ticket with arm a costs the call plus the expected cost of
    being wrong, where K is what one misroute costs the business (the re-route,
    the second touch, the delay):

        E_a = c_a + K * (1 - alpha_a)

    The dearer arm is preferable exactly when E_dear < E_cheap:

        c_d + K(1 - alpha_d) < c_c + K(1 - alpha_c)
        K(alpha_d - alpha_c) > c_d - c_c
        K > (c_d - c_c) / (alpha_d - alpha_c)          [alpha_d > alpha_c]

    Returns that threshold in micro-dollars, rounded up: the smallest integer
    misroute cost at which the dearer arm wins.

    None means the dearer arm is dominated — it costs more and is no more
    accurate, so no misroute cost rescues it. Zero means the arm named 'dear'
    is in fact both cheaper and better, and wins even when misroutes are free.
    """
    for value in (cost_cheap_micro, cost_dear_micro):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"money must be integer micro-dollars, got {value!r}")
    for value in (accuracy_cheap, accuracy_dear):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"accuracy must lie in [0, 1], got {value}")

    accuracy_gap = accuracy_dear - accuracy_cheap
    if accuracy_gap <= 0.0:
        return None

    cost_gap = cost_dear_micro - cost_cheap_micro
    if cost_gap <= 0:
        return 0

    # 0.95 - 0.90 is 0.050000000000000044, which turns an exact 5000 into
    # 5000.000000000006 and then, under ceil, into 5001. Round away the
    # subtraction's error before taking the ceiling. Micro-dollars are already
    # the smallest unit in play, so 6 places is far below anything reportable.
    return math.ceil(round(cost_gap / accuracy_gap, 6))


def _midranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged. [10, 20, 20, 30] -> [1, 2.5, 2.5, 4]."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, ties handled by midranks.

    Used to ask whether a label-free judge orders the arms the way the gold
    labels do. Rank correlation rather than Pearson because a judge that is
    systematically over-generous is still useful if its *ordering* is right —
    only the ranking drives a buying decision.

    None when the correlation is undefined: fewer than two points, or one of
    the inputs is constant (no ranking to correlate).
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        return None

    rx = _midranks(list(xs))
    ry = _midranks(list(ys))
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)
