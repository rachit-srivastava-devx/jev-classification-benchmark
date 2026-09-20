"""The per-pair Jev primitives: request shape, parsing, ranking, failure policy.

`jev_pair` spends one HTTP call per candidate passage, so a hundred of these run
behind a single reported number. Everything that could silently corrupt that
number is pinned here: a value that was never returned being ranked on, a
tie-break that depends on thread completion order, and a query that fails
without its already-paid-for tokens being counted against the spend cap.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jevdemo.arms import JEV_URL, by_name
from jevdemo.rag import jev_pair

NOUL = by_name("jev-noul")
SCORE = by_name("jev-score")


def body_of(**over):
    """A well-formed response, with fields overridable per test."""
    out = {
        "model": "typesafe/jev-1.13",
        "answers": {jev_pair.QUESTION: {"noul": 0.8}},
        "usage": {"input_tokens": 1200, "output_tokens": 0, "cost": 0.00005},
    }
    out.update(over)
    return json.dumps(out).encode()


def task(n=4, query="why do cells divide?"):
    return {"query": query,
            "candidates": [{"id": f"p{i}", "text": f"passage {i}"} for i in range(n)]}


# --- request shape -------------------------------------------------------

def test_noul_request_is_a_noul_question_about_one_pair():
    url, body = jev_pair.build_pair_request(NOUL, "noul", "the query", "the passage")
    assert url == JEV_URL
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == {"query": "the query", "passage": "the passage"}
    q = body["questions"][jev_pair.QUESTION]
    assert q["type"] == "noul"
    assert set(q["criteria"]) == {"true", "false"}


def test_score_request_carries_an_ordered_criteria_list_within_the_documented_range():
    _, body = jev_pair.build_pair_request(SCORE, "score", "q", "p")
    q = body["questions"][jev_pair.QUESTION]
    assert q["type"] == "score"
    # The docs allow 2 to 10 levels and the list is ordered low to high; a dict
    # here would be a Choice wearing Score's name.
    assert isinstance(q["criteria"], list)
    assert 2 <= len(q["criteria"]) <= 10


def test_an_unknown_primitive_raises_instead_of_billing_for_the_wrong_question():
    with pytest.raises(ValueError, match="unknown Jev primitive"):
        jev_pair.build_pair_request(NOUL, "choice", "q", "p")


# --- parsing -------------------------------------------------------------

def test_a_good_response_yields_the_value_and_its_usage():
    value, in_tok, out_tok, cost, failure = jev_pair.parse_pair(body_of(), 200, "noul")
    assert failure == ""
    assert (value, in_tok, out_tok) == (0.8, 1200, 0)
    assert cost == 50  # $0.00005 -> micro-dollars


def test_a_non_200_is_a_failure_with_no_value():
    value, *_, failure = jev_pair.parse_pair(b"nope", 429, "noul")
    assert value is None and failure.startswith("http_error")


def test_a_body_that_is_not_json_is_malformed():
    value, *_, failure = jev_pair.parse_pair(b"<html>gateway</html>", 200, "noul")
    assert value is None and failure.startswith("malformed")


def test_a_boolean_is_rejected_rather_than_ranked_as_one():
    # True is an int in Python; ranking on it would look like a confident 1.0.
    raw = body_of(answers={jev_pair.QUESTION: {"noul": True}})
    value, *_, failure = jev_pair.parse_pair(raw, 200, "noul")
    assert value is None and failure.startswith("malformed")


def test_the_wrong_primitive_key_is_malformed_not_silently_zero():
    raw = body_of(answers={jev_pair.QUESTION: {"score": 2.0}})
    value, *_, failure = jev_pair.parse_pair(raw, 200, "noul")
    assert value is None and failure.startswith("malformed")


def test_a_missing_usage_block_fails_because_the_call_cannot_be_costed():
    raw = json.dumps({"answers": {jev_pair.QUESTION: {"noul": 0.5}}}).encode()
    value, *_, failure = jev_pair.parse_pair(raw, 200, "noul")
    assert value is None and failure.startswith("missing_usage")


# --- ranking -------------------------------------------------------------

def transport_for(values, unscored=()):
    """A fake batched transport: one response covering every passage in the batch.

    `unscored` names passage indices the endpoint declines to answer. It omits
    them from the answers object rather than returning a status code, because
    that is how a partial failure actually reaches us now that many passages
    share one request — and it is the case that fails silently if the ranking
    code defaults a missing answer instead of dropping it.
    """
    calls = []

    def send(url, body, headers):
        ids = list(body["questions"])
        calls.append(ids)
        answers = {pid: {"noul": values[int(pid[1:])]}
                   for pid in ids if int(pid[1:]) not in unscored}
        n = len(ids)
        return 200, body_of(answers=answers,
                            usage={"input_tokens": 1200 * n, "output_tokens": 0,
                                   "cost": 0.00005 * n}), 0.0

    send.calls = calls
    return send


def test_passages_are_ordered_by_descending_value():
    send = transport_for([0.1, 0.9, 0.4, 0.7])
    res = jev_pair.rank(NOUL, "noul", task(4), send, {})
    assert res.failure is None
    assert res.ranking == ["p1", "p3", "p2", "p0"]
    # Four short passages fit in one request, so that is one HTTP call, not four.
    assert send.calls == [["p0", "p1", "p2", "p3"]]
    assert res.subcalls == 1 and res.subcall_failures == 0


def test_ties_are_broken_by_retrieval_position_not_by_which_thread_finished_first():
    # Noul returns repeated values far more often than a softmax does. Without an
    # explicit tie-break the order among equals would differ run to run.
    res = jev_pair.rank(NOUL, "noul", task(4),
                        transport_for([0.5, 0.5, 0.5, 0.5]), {})
    assert res.ranking == ["p0", "p1", "p2", "p3"]


def test_usage_is_summed_across_every_subcall():
    res = jev_pair.rank(NOUL, "noul", task(4),
                        transport_for([0.1, 0.2, 0.3, 0.4]), {})
    assert res.input_tokens == 4 * 1200
    assert res.reported_cost_micro == 4 * 50


def test_one_unscorable_passage_is_ranked_last_rather_than_dropped():
    # 1 failure out of 20 is under the 10% threshold, so the query still scores.
    res = jev_pair.rank(NOUL, "noul", task(20),
                        transport_for([0.5] * 20, unscored=(0,)), {})
    assert res.failure is None
    assert res.ranking[-1] == "p0"
    assert len(res.ranking) == 20        # nothing lost
    assert res.subcall_failures == 1


def test_a_query_whose_subcalls_broadly_failed_is_a_failure_not_a_partial_score():
    res = jev_pair.rank(NOUL, "noul", task(20),
                        transport_for([0.5] * 20, unscored=(0, 1)), {})
    assert res.failure == "http_error"
    assert res.ranking == []
    assert res.subcall_failures == 2


def test_a_failed_query_still_reports_the_tokens_its_successful_subcalls_burned():
    # A spend cap that cannot see this money is not a cap.
    # The batch came back and was billed in full; two of its answers were missing.
    # The money is real either way and the cap has to see it.
    res = jev_pair.rank(NOUL, "noul", task(20),
                        transport_for([0.5] * 20, unscored=(0, 1)), {})
    assert res.input_tokens == 20 * 1200
    assert res.reported_cost_micro == 20 * 50


def _exploding(on_id="p0"):
    """A transport that kills any batch containing `on_id`, and records the batches."""
    seen = []

    def send(url, body, headers):
        ids = list(body["questions"])
        seen.append(ids)
        if on_id in ids:
            raise RuntimeError("connection reset")
        return 200, body_of(
            answers={pid: {"noul": 0.1 * int(pid[1:])} for pid in ids},
            usage={"input_tokens": 1200, "output_tokens": 0, "cost": 0.00005}), 0.0

    send.seen = seen
    return send


def _big_task(n=20, words=3000):
    """Passages long enough that they cannot all share one request."""
    return {"query": "why do cells divide?",
            "candidates": [{"id": f"p{i}", "text": "word " * words} for i in range(n)]}


def test_a_dead_batch_does_not_discard_the_batches_that_were_paid_for():
    # 60 passages, so one dead batch stays under the 10% threshold and the query
    # survives — which is the only case where "the rest still ranked" is testable.
    send = _exploding()
    res = jev_pair.rank(NOUL, "noul", _big_task(n=60), send, {})
    assert len(send.seen) > 1, "test is meaningless with a single batch"
    # Everything outside the exploding batch still ranked, and its tokens and
    # money are still counted. A cap that cannot see that money is not a cap.
    assert res.failure is None
    assert res.input_tokens > 0 and res.reported_cost_micro > 0
    assert len(res.ranking) == 60, "no passage may be silently dropped"


def test_a_failed_batch_is_split_and_retried_so_one_error_costs_fewer_passages():
    """Batching trades blast radius for efficiency; the split buys some of it back.

    Unbatched, one dead call cost one passage. Batched, it costs the whole batch —
    enough to trip the failure threshold and lose a query that would have survived
    before. Halving the dead batch and retrying recovers the half that was fine.
    """
    send = _exploding()
    res = jev_pair.rank(NOUL, "noul", _big_task(), send, {})
    first_batch = len(send.seen[0])
    assert res.subcall_failures < first_batch, (
        "without the retry, every passage in the dead batch is lost")
    # The retry re-sent the halves, so more requests went out than there were batches.
    assert len(send.seen) > len([b for b in send.seen if "p0" not in b])


def test_a_batch_refused_for_size_is_halved_rather_than_losing_the_query():
    """The reason this beats Choice: Choice has no smaller request to fall back to."""
    calls = []

    def send(url, body, headers):
        ids = list(body["questions"])
        calls.append(len(ids))
        if len(ids) > 5:                       # stands in for the input ceiling
            return 200, body_of(answers={}), 0.0
        return 200, body_of(
            answers={pid: {"noul": 0.1 * int(pid[1:])} for pid in ids},
            usage={"input_tokens": 1200, "output_tokens": 0, "cost": 0.00005}), 0.0

    res = jev_pair.rank(NOUL, "noul", _big_task(n=10, words=1500), send, {})
    assert max(calls) > 5, "the oversized request has to happen for this to mean anything"
    assert min(calls) <= 5, "it must then be retried smaller"
    assert res.failure is None
    assert len(res.ranking) == 10


def test_the_score_arm_ranks_on_the_score_key():
    def send(url, body, headers):
        ids = list(body["questions"])
        raw = body_of(answers={pid: {"score": float(pid[1:])} for pid in ids})
        return 200, raw, 0.0

    res = jev_pair.rank(SCORE, "score", task(4), send, {})
    assert res.ranking == ["p3", "p2", "p1", "p0"]


# --- batch packing -------------------------------------------------------

def test_every_candidate_lands_in_exactly_one_batch():
    cands = [{"id": f"p{i}", "text": "word " * 400} for i in range(100)]
    batches = jev_pair.plan_batches("a query", cands)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(100)), "no passage dropped or duplicated"
    assert flat == sorted(flat), "retrieval order is preserved, so packing is deterministic"


def test_a_hundred_short_passages_go_out_as_far_fewer_than_a_hundred_requests():
    """The whole point: the envelope and the query are paid for once per batch."""
    cands = [{"id": f"p{i}", "text": "short passage"} for i in range(100)]
    assert len(jev_pair.plan_batches("q", cands)) == 1


def test_no_batch_is_planned_over_the_measured_input_ceiling():
    cands = [{"id": f"p{i}", "text": "word " * 800} for i in range(100)]
    for b in jev_pair.plan_batches("q" * 2000, cands):
        est = (jev_pair._est_tokens("q" * 2000) + jev_pair._ENVELOPE_TOKENS
               + sum(jev_pair._est_tokens(cands[i]["text"])
                     + jev_pair._PER_QUESTION_TOKENS for i in b))
        assert len(b) == 1 or est <= jev_pair.BATCH_TOKEN_BUDGET


def test_one_passage_bigger_than_the_whole_budget_is_still_asked_about():
    """Dropping it would silently remove a candidate from the ranking."""
    cands = [{"id": "p0", "text": "word " * 100_000}]
    assert jev_pair.plan_batches("q", cands) == [[0]]


def test_no_candidates_means_no_requests_rather_than_one_empty_one():
    assert jev_pair.plan_batches("q", []) == []


def test_the_batched_request_asks_the_same_judgement_as_the_single_one():
    """Batching must change packaging only; a reworded rubric would confound the
    encoding comparison this arm exists to support."""
    _, single = jev_pair.build_pair_request(SCORE, "score", "q", "p")
    _, batch = jev_pair.build_batch_request(
        SCORE, "score", "q", [{"id": "p0", "text": "p"}])
    assert (batch["questions"]["p0"]["criteria"]
            == single["questions"][jev_pair.QUESTION]["criteria"])
    assert batch["questions"]["p0"]["type"] == "score"
    assert batch["state"]["query"] == "q"
    assert batch["state"]["passages"] == {"p0": "p"}


def test_an_unknown_primitive_stops_before_the_money():
    with pytest.raises(ValueError):
        jev_pair.build_batch_request(SCORE, "sentiment", "q", [{"id": "p0", "text": "p"}])


# --- The batched planner, exercised against the real committed task set. -------
# These are structural properties, not a cost measurement. They prove the batched
# path asks about every passage exactly once and stays inside the token budget on
# the same 350 questions the report was built from. They say nothing about what
# the API charges or how long it takes; only a live run establishes that.

def _real_tasks():
    p = pathlib.Path(__file__).resolve().parents[1] / "data/rag/tasks.json"
    if not p.exists():
        pytest.skip("committed task set not present")
    t = json.loads(p.read_text())
    return t["tasks"] if isinstance(t, dict) and "tasks" in t else t


def _batch_tokens(query: str, cands: list[dict], grp: list[int]) -> int:
    return (jev_pair._est_tokens(query) + jev_pair._ENVELOPE_TOKENS
            + sum(jev_pair._est_tokens(cands[i]["text"])
                  + jev_pair._PER_QUESTION_TOKENS for i in grp))


def test_batching_asks_about_every_passage_exactly_once_on_real_data():
    """A batch plan that silently dropped a candidate would raise the arm's
    apparent precision by shrinking the set it can get wrong."""
    tasks = _real_tasks()
    assert tasks, "task set is empty — this gate would otherwise pass on nothing"
    for q in tasks:
        cands = q["candidates"]
        flat = [i for grp in jev_pair.plan_batches(q["query"], cands) for i in grp]
        assert flat == list(range(len(cands))), q["query_id"]


def test_no_multi_passage_batch_exceeds_the_token_budget_on_real_data():
    """A batch over the ceiling is refused by the API, which would show up as
    the model failing rather than as our planner mis-sizing a request."""
    tasks = _real_tasks()
    checked = 0
    for q in tasks:
        cands = q["candidates"]
        for grp in jev_pair.plan_batches(q["query"], cands):
            if len(grp) > 1:
                tok = _batch_tokens(q["query"], cands, grp)
                assert tok <= jev_pair.BATCH_TOKEN_BUDGET, f"{q['query_id']}: {tok}"
                checked += 1
    assert checked > 0, "no multi-passage batches checked — the gate measured nothing"
