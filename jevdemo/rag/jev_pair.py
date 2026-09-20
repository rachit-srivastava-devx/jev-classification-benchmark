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

#: How many times a batch that returned nothing is halved and retried. One split
#: turns a 25-passage loss into at worst two 13-passage attempts, and in practice
#: rescues both transient errors and requests refused for size. More than one
#: round would start costing real money on a genuinely broken endpoint.
BATCH_RETRY_SPLITS = 1


def _add_cost(*vals: int | None) -> int | None:
    """Sum costs, treating 'not reported' as absent rather than as zero."""
    known = [v for v in vals if v is not None]
    return sum(known) if known else None

#: Per query. The cookbook uses 12 against the direct endpoint; this goes through
#: OpenRouter and the runner already has its own outer pool over queries, so the
#: inner pool is kept small enough that the product of the two stays civil.
PAIR_WORKERS = 6


#: Chars per token at the densest packing seen across these three corpora (2.89;
#: the loosest was 4.24). Dividing by the *densest* over-estimates the token count
#: for every other corpus, which is the safe direction: a batch that is smaller
#: than it needed to be costs one extra call, a batch that is larger than the
#: endpoint accepts costs the whole query.
CHARS_PER_TOKEN = 2.89

#: The Choice arm measured the endpoint's real input ceiling at ~32,768 tokens:
#: the largest request that ever succeeded carried 32,850 and nothing above that
#: returned an answer. Batches are packed to well under it, because a batch that
#: is refused is a query lost, and the only thing a smaller batch costs is one
#: more round trip. This is the whole reason batching beats Choice rather than
#: merely matching it: Choice must fit all 100 chunks in one request or fail,
#: while a batch that would not fit simply becomes two batches.
BATCH_TOKEN_BUDGET = 24_000

#: Never send a batch of one when the whole point is amortising the envelope, and
#: never let one enormous passage silently become a 1-passage batch without the
#: caller being able to see it happened.
MIN_BATCH = 1


def _est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def plan_batches(query: str, cands: list[dict],
                 budget: int = BATCH_TOKEN_BUDGET) -> list[list[int]]:
    """Group candidate indices into as few requests as fit under `budget`.

    Greedy first-fit in retrieval order, so the grouping is deterministic and a
    re-run of the same data sends the same requests. The query and the per-question
    boilerplate are charged to every batch, because every batch carries them.

    A single passage larger than the whole budget still goes out on its own rather
    than being dropped — one over-sized request that may fail is strictly more
    informative than silently not asking about that passage at all.
    """
    if not cands:
        return []
    fixed = _est_tokens(query) + _ENVELOPE_TOKENS
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_tok = fixed
    for i, c in enumerate(cands):
        cost = _est_tokens(c["text"]) + _PER_QUESTION_TOKENS
        if cur and cur_tok + cost > budget:
            batches.append(cur)
            cur, cur_tok = [], fixed
        cur.append(i)
        cur_tok += cost
    if cur:
        batches.append(cur)
    return batches


#: Rough, deliberately generous allowances so `plan_batches` errs small. Measured
#: against the unbatched run: ~437 input tokens of fixed cost per request, of
#: which the rubric and instructions are the part that repeats per question.
_ENVELOPE_TOKENS = 350
_PER_QUESTION_TOKENS = 120


def build_batch_request(arm: Arm, primitive: str, query: str,
                        cands: list[dict]) -> tuple[str, dict]:
    """One request asking about several passages against one shared query.

    The judgement is the same one `build_pair_request` asks — same rubric, same
    levels, same wording — so batching changes only how many HTTP requests carry
    it, never what the model is deciding. That matters because this arm exists to
    be compared with the other two encodings; a reworded prompt would confound
    the encoding comparison with a prompt change.
    """
    if primitive not in ("noul", "score"):
        raise ValueError(f"unknown Jev primitive: {primitive!r}")
    if not cands:
        raise ValueError("cannot build a batch request with no candidates")
    passages = {c["id"]: c["text"] for c in cands}
    questions = {}
    for c in cands:
        where = (f"The query is in state.query and the passage is in "
                 f"state.passages[{c['id']!r}].")
        if primitive == "noul":
            questions[c["id"]] = {
                "type": "noul",
                "instructions": NOUL_INSTRUCTIONS.split(" The query is in")[0] + " " + where,
                "criteria": NOUL_CRITERIA}
        else:
            questions[c["id"]] = {
                "type": "score",
                "instructions": SCORE_INSTRUCTIONS.split(" The query is in")[0] + " " + where,
                "criteria": SCORE_CRITERIA}
    body = {
        "model": arm.model,
        "state": {"query": query, "passages": passages},
        "questions": questions,
    }
    return arm.url, body


def parse_batch(raw: bytes, status: int, primitive: str,
                ids: list[str]) -> tuple[dict[str, float], int, int, int | None, str]:
    """One batch's answers: (id -> value, in_tok, out_tok, cost_micro, failure).

    A missing or malformed individual answer is dropped from the returned mapping
    rather than defaulted, so a passage the endpoint declined to score is ranked
    last and counted — exactly as in the unbatched path — instead of silently
    scoring zero and looking like a confident rejection.
    """
    if status != 200:
        return {}, 0, 0, None, f"{HTTP_ERROR}: HTTP {status}"
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return {}, 0, 0, None, f"{MALFORMED}: body is not JSON: {exc}"
    if not isinstance(body, dict):
        return {}, 0, 0, None, f"{MALFORMED}: body is not an object"
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}, 0, 0, None, f"{MISSING_USAGE}: no usage block"
    # Same key names the single-pair path reads. Guessing at `prompt_tokens`
    # here — as a first draft of this function did — reports zero tokens and
    # zero spend for every batch, which does not fail loudly: it just blinds
    # the spend cap while the run keeps buying calls.
    try:
        in_tok = int(usage["input_tokens"])
        out_tok = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError):
        return {}, 0, 0, None, f"{MISSING_USAGE}: incomplete usage: {sorted(usage)}"
    cost = cost_to_micro(usage.get("cost"))
    answers = body.get("answers")
    if not isinstance(answers, dict):
        return {}, in_tok, out_tok, cost, f"{MALFORMED}: no answers object"
    got: dict[str, float] = {}
    for pid in ids:
        a = answers.get(pid)
        if not isinstance(a, dict):
            continue
        v = a.get(primitive)
        # bool is excluded explicitly: `True` is an int in Python and would rank
        # as 1.0 without ever having been a probability.
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            got[pid] = float(v)
    return got, in_tok, out_tok, cost, ""


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
    """Rank one query's candidates, asking about as many at a time as will fit.

    The endpoint evaluates every question in a request in parallel and against the
    same state, so asking about one passage per request re-sends the query and the
    whole request envelope once per passage. Measured on the unbatched run that
    was ~437 input tokens of fixed cost repeated a hundred times per query — more
    than three times the tokens of the Choice arm for the identical information.
    Batching sends each of those once per batch instead.

    Returns a single RankResult covering all the sub-calls, so everything
    downstream — metrics, bootstrap, tables — treats this arm exactly like a
    one-call arm. The sub-call count and failure count ride along so the report
    can state how many HTTP requests a number actually cost.
    """
    cands = task["candidates"]
    ids = [c["id"] for c in cands]
    batches = plan_batches(task["query"], cands)

    def ask(idx: list[int]):
        group = [cands[i] for i in idx]
        gids = [c["id"] for c in group]
        url, body = build_batch_request(arm, primitive, task["query"], group)
        try:
            status, raw, _ = transport(url, body, headers)
        except Exception as exc:                      # noqa: BLE001 - see below
            # The transport raises TransportError for network faults; anything
            # else here is a bug, but one batch's bug must not abort the others
            # and lose their already-paid-for answers.
            return {}, 0, 0, None, f"{HTTP_ERROR}: {str(exc)[:120]}", len(group)
        got, i_t, o_t, cost, fail = parse_batch(raw, status, primitive, gids)
        return got, i_t, o_t, cost, fail, len(group)

    with ThreadPoolExecutor(max_workers=PAIR_WORKERS) as pool:
        results = list(pool.map(ask, batches))

        # Batching trades blast radius for efficiency: unbatched, one dead call
        # cost one passage out of a hundred and the query survived; batched, one
        # dead call can cost a quarter of them and trip the failure threshold.
        # So a batch that came back with nothing is halved and retried once. That
        # covers both reasons a batch fails — a transient error, and a request the
        # endpoint refused for size — and halving is what makes this arm strictly
        # better than Choice, which has no smaller request to fall back to.
        for attempt in range(BATCH_RETRY_SPLITS):
            dead = [n for n, r in enumerate(results) if not r[0] and len(batches[n]) > 1]
            if not dead:
                break
            halves = []
            for n in dead:
                idx = batches[n]
                mid = len(idx) // 2
                halves.append((n, idx[:mid], idx[mid:]))
            retried = list(pool.map(ask, [h for _, a, b in halves for h in (a, b)]))
            for k, (n, a, b) in enumerate(halves):
                left, right = retried[2 * k], retried[2 * k + 1]
                # Keep the usage from the failed attempt: it may still have been
                # billed, and a cap that cannot see that money is not a cap.
                prev = results[n]
                merged = dict(left[0]); merged.update(right[0])
                results[n] = (merged,
                              prev[1] + left[1] + right[1],
                              prev[2] + left[2] + right[2],
                              _add_cost(prev[3], left[3], right[3]),
                              left[4] or right[4],
                              prev[5])
                batches[n] = a + b

    in_tok = sum(r[1] for r in results)
    out_tok = sum(r[2] for r in results)
    costs = [r[3] for r in results if r[3] is not None]
    cost = sum(costs) if costs else None

    by_id: dict[str, float] = {}
    for got, _, _, _, _, _ in results:
        by_id.update(got)
    # A passage counts as failed when no usable value came back for it, whichever
    # way that happened: a batch that errored, or a batch that returned but left
    # this one out. Counting whole failed batches instead would under-report the
    # second case, which is the one that fails silently.
    failed_n = sum(1 for i in ids if i not in by_id)
    first_fail = next((r[4] for r in results if r[4]), "")

    if failed_n >= max(1, round(MAX_SUBCALL_FAILURE_RATE * len(cands))):
        detail = (f"{failed_n}/{len(cands)} passages unscored across "
                  f"{len(batches)} batch call(s)")
        if first_fail:
            detail += f"; first: {first_fail[:120]}"
        res = failed(HTTP_ERROR, detail)
        return _with_subcalls(res, len(batches), failed_n, in_tok, out_tok, cost)

    # Retrieval position is the tie-break, identically to the Choice arm: noul
    # values repeat far more often than a softmax does, and without it the order
    # among equals would come from thread completion order — a different ranking
    # on every run of the same data.
    scored = [(by_id[i], n) for n, i in enumerate(ids) if i in by_id]
    ranking = [ids[n] for _, n in sorted(scored, key=lambda t: (-t[0], t[1]))]
    # A passage nobody could score is not evidence that it is bad, but it cannot
    # be ranked either. It goes last, in retrieval order, and is counted.
    ranking += [i for i in ids if i not in by_id]

    res = RankResult(ranking=ranking, input_tokens=in_tok, output_tokens=out_tok,
                     reasoning_tokens=0, reported_cost_micro=cost, dropped=0,
                     failure=None, detail="")
    return _with_subcalls(res, len(batches), failed_n, in_tok, out_tok, cost)


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
