"""The two tables the decision rests on.

`distinguishability_matrix` answers "is this gap real?" for every pair of arms.
`break_even_table` answers "what would this arm's extra accuracy have to be worth
to justify its price?" against the cheapest arm in the roster.

Both consume the `arms` rows written to results.json, so neither can drift from
the measured data. Neither has any I/O.
"""

from __future__ import annotations

from jevdemo.stats import break_even_exact, difference_interval

#: `cost_per_1000_micro` is micro-dollars per 1,000 calls. Numerically that is
#: exactly the per-call cost in NANO-dollars: (micro/1000 calls) x (1000 nano/micro)
#: = nano per call. The identity is used rather than dividing by 1000, because
#: dividing would leave money on a float and lose the integer-minor-unit rule.
#: Every threshold below is therefore in nano-dollars per misroute.
NANO_PER_USD = 1_000_000_000

#: A gain that the sample cannot resolve still gets a threshold, but carries this
#: warning. Suppressing the number entirely would hide how large the price gap is;
#: printing it unqualified would dress sampling noise as a business case.
UNRESOLVED_NOTE = (
    "accuracy gain is not resolvable at this sample size — treat the threshold as "
    "a lower bound on what you would be paying for"
)
DOMINATED_NOTE = "dominated: costs more and is no more accurate"
NO_COST_NOTE = "no measured cost for this arm"


def distinguishability_matrix(arms: list[dict]) -> dict[str, dict[str, dict]]:
    """For every ordered pair, whether the row arm is measurably better.

    'tie' does not mean the arms are equal. It means this experiment cannot tell
    them apart, which is a statement about the sample and not about the models.
    """
    out: dict[str, dict[str, dict]] = {}
    for a in arms:
        row: dict[str, dict] = {}
        for b in arms:
            if a["arm"] == b["arm"]:
                row[b["arm"]] = {"verdict": "self", "resolved": False,
                                 "interval": [0.0, 0.0]}
                continue
            lo, hi = difference_interval(
                a["correct"], a["attempted"], b["correct"], b["attempted"]
            )
            if lo > 0:
                verdict, resolved = "better", True
            elif hi < 0:
                verdict, resolved = "worse", True
            else:
                verdict, resolved = "tie", False
            row[b["arm"]] = {"verdict": verdict, "resolved": resolved,
                             "interval": [lo, hi]}
        out[a["arm"]] = row
    return out


def break_even_table(arms: list[dict]) -> list[dict]:
    """What one misroute must cost before each arm beats the cheapest one.

    The cheapest arm is the baseline because it is the floor any spend is measured
    against: choosing anything else is a decision to pay more, and this table
    prices that decision.
    """
    priced = [a for a in arms if a.get("cost_per_1000_micro") is not None]
    if not priced:
        return [
            {"arm": a["arm"], "is_baseline": False, "break_even_nano": None,
             "break_even_usd": None, "resolved": False, "note": NO_COST_NOTE}
            for a in arms
        ]

    baseline = min(priced, key=lambda a: a["cost_per_1000_micro"])
    base_cost = baseline["cost_per_1000_micro"]  # per-call nano-dollars; see above

    rows = []
    for a in arms:
        name = a["arm"]
        if a.get("cost_per_1000_micro") is None:
            rows.append({"arm": name, "is_baseline": False, "break_even_nano": None,
                         "break_even_usd": None, "resolved": False, "note": NO_COST_NOTE})
            continue

        if name == baseline["arm"]:
            rows.append({"arm": name, "is_baseline": True, "break_even_nano": 0,
                         "break_even_usd": 0.0, "resolved": True,
                         "note": "baseline: the cheapest arm measured"})
            continue

        threshold = break_even_exact(
            base_cost, baseline["correct"], baseline["attempted"],
            a["cost_per_1000_micro"], a["correct"], a["attempted"],
        )
        lo, hi = difference_interval(
            a["correct"], a["attempted"], baseline["correct"], baseline["attempted"]
        )
        resolved = lo > 0

        if threshold is None:
            note = DOMINATED_NOTE
        elif not resolved:
            note = UNRESOLVED_NOTE
        else:
            note = ""

        rows.append({
            "arm": name,
            "is_baseline": False,
            "break_even_nano": threshold,
            "break_even_usd": None if threshold is None else threshold / NANO_PER_USD,
            "resolved": resolved,
            "note": note,
        })
    return rows
