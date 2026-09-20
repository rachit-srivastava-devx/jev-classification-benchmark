"""Reranking through Jev's per-pair primitives: one question per passage.

`jev_rank` asks a single Choice whose criteria are all 100 passages, so the
passages compete for one pool of probability. This module asks instead about one
(query, passage) pair at a time — the shape TypeSafe's own reranking cookbook
uses — and gets an independent judgement per passage. Two primitives answer that
shape:

    noul   "Is this passage an answer to the query?" -> a probability, 0 to 1.
    score  the same pair placed on a four-level rubric  -> a position, 0 to 3.

Both are documented as the right tool for their question shape: the docs say to
use Noul "when the answer is a yes or no" and Score when it is "a position on a
spectrum you can describe in steps". Choice is documented for "a fixed set of
options" — which a candidate list technically is, but only one of its 100 options
can hold most of the probability, and a query with six correct passages has six.
Measuring all three is the point of this module: the earlier report measured
Choice and reported it as "Jev", which is a claim about an encoding, not a
product.

Cost note, because it is the whole trade-off: one Choice call reads the query
once and all 100 passages once. A hundred Noul calls read one passage each but
re-read the query a hundred times, so a corpus with long queries (BRIGHT posts
run to a thousand tokens) pays for that query a hundred times over. The price of
an independent judgement per passage is paid in input tokens, and the report
gives the measured number rather than this argument.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from jevdemo.arms import Arm
from jevdemo.rag.rank_result import (
    HTTP_ERROR,
    MALFORMED,
    MISSING_USAGE,
    RankResult,
    cost_to_micro,
    failed,
)

QUESTION = "passage_answers_query"

#: Deliberately the same judgement in three grammars. The three primitives are
#: being compared as encodings of one question, so any difference in what they
#: are asked to decide would confound the comparison with a prompt difference.
NOUL_INSTRUCTIONS = (
    "Does this passage contain information that answers the query? "
    "The query is in state.query and the passage is in state.passage."
)
NOUL_CRITERIA = {
    "true": "The passage states something a correct answer to the query would use.",
    "false": "The passage is off-topic, or shares the query's vocabulary while "
             "answering a different question.",
}

#: Four levels, ordered low to high. The docs allow 2 to 10; four is the usual
#: graded-relevance scale in retrieval evaluation (TREC uses 0-3), so the rubric
#: is borrowed rather than invented.
SCORE_INSTRUCTIONS = (
    "How useful is this passage for answering the query? "
    "The query is in state.query and the passage is in state.passage."
)
SCORE_CRITERIA = [
    "Irrelevant: the passage is about something else.",
    "Related but not useful: same topic, answers a different question.",
    "Partly useful: contains something a correct answer would use, but not enough on its own.",
    "Directly answers the query.",
]

#: A hundred calls per query means a single flaky response should not discard the
#: query, and a broadly failing arm should not be quietly scored on whatever came
#: back. Below this share the failures are ranked last and counted; at or above
#: it the whole query is a failure. 10% of 100 passages is 10 — enough slack for
#: transient 5xx, far too much to be noise.
MAX_SUBCALL_FAILURE_RATE = 0.10

#: Per query. The cookbook uses 12 against the direct endpoint; this goes through
#: OpenRouter and the runner already has its own outer pool over queries, so the
#: inner pool is kept small enough that the product of the two stays civil.
PAIR_WORKERS = 6


def build_pair_request(arm: Arm, primitive: str, query: str, passage: str) -> tuple[str, dict]:
    """One request about one (query, passage) pair.

    Raises ValueError on an unknown primitive rather than silently building a
    Choice, because a typo that costs money should stop before the money.
    """
    if primitive == "noul":
        question = {"type": "noul", "instructions": NOUL_INSTRUCTIONS,
                    "criteria": NOUL_CRITERIA}
    elif primitive == "score":
        question = {"type": "score", "instructions": SCORE_INSTRUCTIONS,
                    "criteria": SCORE_CRITERIA}
    else:
        raise ValueError(f"unknown Jev primitive: {primitive!r}")
    body = {
        "model": arm.model,
        "state": {"query": query, "passage": passage},
        "questions": {QUESTION: question},
    }
    return arm.url, body


def parse_pair(raw: bytes, status: int, primitive: str) -> tuple[float | None, int, int, int | None, str]:
    """One pair's answer: (value, input_tokens, output_tokens, cost_micro, failure).

    `value` is None exactly when `failure` is non-empty, so a caller cannot
    accidentally rank on a value that was never returned.
    """
    if status != 200:
        return None, 0, 0, None, f"{HTTP_ERROR}: HTTP {status}"
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return None, 0, 0, None, f"{MALFORMED}: body is not JSON: {exc}"
    if not isinstance(body, dict):
        return None, 0, 0, None, f"{MALFORMED}: body is not an object"

    answer = (body.get("answers") or {}).get(QUESTION)
    if not isinstance(answer, dict):
        return None, 0, 0, None, f"{MALFORMED}: no answers.{QUESTION}"
    value = answer.get(primitive)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        # bool is excluded explicitly: `True` is an int in Python and would rank
        # as 1.0 without ever having been a probability.
        return None, 0, 0, None, f"{MALFORMED}: answers.{QUESTION}.{primitive} is {value!r}"

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None, 0, 0, None, f"{MISSING_USAGE}: no usage block"
    try:
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError):
        return None, 0, 0, None, f"{MISSING_USAGE}: incomplete usage: {sorted(usage)}"

    return float(value), input_tokens, output_tokens, cost_to_micro(usage.get("cost")), ""


def rank(arm: Arm, primitive: str, task: dict, transport, headers: dict) -> RankResult:
    """Rank one query's candidates by asking about each passage on its own.

    Returns a single RankResult covering all the sub-calls, so everything
    downstream — metrics, bootstrap, tables — treats this arm exactly like a
    one-call arm. The sub-call count and failure count ride along so the report
    can state how many HTTP requests a number actually cost.
    """
    cands = task["candidates"]
    ids = [c["id"] for c in cands]

    def ask(c):
        url, body = build_pair_request(arm, primitive, task["query"], c["text"])
        try:
            status, raw, _ = transport(url, body, headers)
        except Exception as exc:                      # noqa: BLE001 - see below
            # The transport raises TransportError for network faults; anything
            # else here is a bug, but one passage's bug must not abort the other
            # ninety-nine and lose their already-paid-for answers.
            return None, 0, 0, None, f"{HTTP_ERROR}: {str(exc)[:120]}"
        return parse_pair(raw, status, primitive)

    with ThreadPoolExecutor(max_workers=PAIR_WORKERS) as pool:
        got = list(pool.map(ask, cands))

    in_tok = sum(r[1] for r in got)
    out_tok = sum(r[2] for r in got)
    costs = [r[3] for r in got if r[3] is not None]
    cost = sum(costs) if costs else None
    failures = [r[4] for r in got if r[4]]

    if len(failures) >= max(1, round(MAX_SUBCALL_FAILURE_RATE * len(cands))):
        res = failed(HTTP_ERROR,
                     f"{len(failures)}/{len(cands)} pair calls failed; "
                     f"first: {failures[0][:120]}")
        return _with_subcalls(res, len(cands), len(failures), in_tok, out_tok, cost)

    # Retrieval position is the tie-break, identically to the Choice arm: noul
    # values repeat far more often than a softmax does, and without it the order
    # among equals would come from thread completion order — a different ranking
    # on every run of the same data.
    scored = [(got[i][0], i) for i in range(len(cands)) if got[i][0] is not None]
    ranking = [ids[i] for _, i in sorted(scored, key=lambda t: (-t[0], t[1]))]
    # A passage nobody could score is not evidence that it is bad, but it cannot
    # be ranked either. It goes last, in retrieval order, and is counted.
    ranking += [ids[i] for i in range(len(cands)) if got[i][0] is None]

    res = RankResult(ranking=ranking, input_tokens=in_tok, output_tokens=out_tok,
                     reasoning_tokens=0, reported_cost_micro=cost, dropped=0,
                     failure=None, detail="")
    return _with_subcalls(res, len(cands), len(failures), in_tok, out_tok, cost)


def _with_subcalls(res: RankResult, n: int, failures: int,
                   in_tok: int, out_tok: int, cost: int | None) -> RankResult:
    """Attach the call accounting, keeping usage even on a failed query.

    A query that failed still cost money for the sub-calls that succeeded, and a
    spend cap that cannot see that money is not a cap.
    """
    from dataclasses import replace
    return replace(res, subcalls=n, subcall_failures=failures,
                   input_tokens=in_tok, output_tokens=out_tok,
                   reported_cost_micro=cost)
