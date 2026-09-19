"""The one shape every arm returns, so metrics never branches on protocol."""

from __future__ import annotations

from dataclasses import dataclass

# The four ways a call can fail to produce a usable label. Spec 03.
HTTP_ERROR = "http_error"
MALFORMED = "malformed_output"
INVALID_LABEL = "invalid_label"
MISSING_USAGE = "missing_usage"


@dataclass(frozen=True)
class Prediction:
    label: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    reported_cost_micro: int | None
    confidence: float | None
    failure: str | None
    detail: str
    reasoning_rejected: bool = False


def failed(kind: str, detail: str = "", *, reasoning_rejected: bool = False) -> Prediction:
    return Prediction(
        label=None,
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        reported_cost_micro=None,
        confidence=None,
        failure=kind,
        detail=detail,
        reasoning_rejected=reasoning_rejected,
    )


def cost_to_micro(cost: object) -> int | None:
    """USD float from the provider to integer micro-dollars, half-up.

    Returns None when the provider did not report a cost. An absent cost is
    unverified, not free.
    """
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return None
    if cost < 0:
        return None
    return int(cost * 1_000_000 + 0.5)
