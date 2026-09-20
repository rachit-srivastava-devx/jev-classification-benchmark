"""One reranking call's outcome.

Mirrors `jevdemo.result.Prediction`, but carries an ordered list of passage ids
instead of a single label, and counts the ids the model named that do not exist.
That count is kept separate from the ranking quality on purpose: a model that
invents passage ids is failing in a way a low recall score would hide.
"""

from __future__ import annotations

from dataclasses import dataclass

from jevdemo.result import cost_to_micro

HTTP_ERROR = "http_error"      #: the provider did not return 200
MALFORMED = "malformed"        #: a 200 whose body has no ranking in it
BAD_IDS = "bad_ids"            #: a ranking was returned, none of it real
MISSING_USAGE = "missing_usage"  #: no token counts, so the call cannot be costed


@dataclass(frozen=True)
class RankResult:
    ranking: list[str]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    reported_cost_micro: int | None
    dropped: int
    failure: str | None
    detail: str
    #: How many HTTP requests this one result consumed, and how many of them
    #: failed. One for every arm that answers a query in a single call; 100 for
    #: the per-passage Jev primitives. Defaulted so the single-call constructors
    #: elsewhere keep working, and carried because "$0.84 per thousand queries"
    #: and "84 cents for a hundred thousand requests" are different sentences and
    #: the report has to be able to say which one it means.
    subcalls: int = 1
    subcall_failures: int = 0


def failed(kind: str, detail: str = "") -> RankResult:
    return RankResult([], 0, 0, 0, None, 0, kind, detail)


#: Re-exported rather than reimplemented. The conversion from the provider's USD
#: float to integer micro-dollars already exists for the ticket-routing report
#: and must stay identical, or the two reports would round money differently.
__all__ = ["RankResult", "cost_to_micro", "failed",
           "HTTP_ERROR", "MALFORMED", "BAD_IDS", "MISSING_USAGE"]
