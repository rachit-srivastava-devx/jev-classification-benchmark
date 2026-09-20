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

def transport_for(values, statuses=None):
    """A fake transport that answers each passage by its text's trailing index."""
    calls = []

    def send(url, body, headers):
        i = int(body["state"]["passage"].split()[-1])
        calls.append(i)
        status = 200 if statuses is None else statuses[i]
        return status, body_of(answers={jev_pair.QUESTION: {"noul": values[i]}}), 0.0

    send.calls = calls
    return send


def test_passages_are_ordered_by_descending_value():
    res = jev_pair.rank(NOUL, "noul", task(4),
                        transport_for([0.1, 0.9, 0.4, 0.7]), {})
    assert res.failure is None
    assert res.ranking == ["p1", "p3", "p2", "p0"]
    assert res.subcalls == 4 and res.subcall_failures == 0


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
                        transport_for([0.5] * 20, statuses=[500] + [200] * 19), {})
    assert res.failure is None
    assert res.ranking[-1] == "p0"
    assert len(res.ranking) == 20        # nothing lost
    assert res.subcall_failures == 1


def test_a_query_whose_subcalls_broadly_failed_is_a_failure_not_a_partial_score():
    res = jev_pair.rank(NOUL, "noul", task(20),
                        transport_for([0.5] * 20, statuses=[500] * 2 + [200] * 18), {})
    assert res.failure == "http_error"
    assert res.ranking == []
    assert res.subcall_failures == 2


def test_a_failed_query_still_reports_the_tokens_its_successful_subcalls_burned():
    # A spend cap that cannot see this money is not a cap.
    res = jev_pair.rank(NOUL, "noul", task(20),
                        transport_for([0.5] * 20, statuses=[500] * 2 + [200] * 18), {})
    assert res.input_tokens == 18 * 1200
    assert res.reported_cost_micro == 18 * 50


def test_a_transport_exception_on_one_passage_does_not_abort_the_others():
    def send(url, body, headers):
        i = int(body["state"]["passage"].split()[-1])
        if i == 0:
            raise RuntimeError("connection reset")
        return 200, body_of(answers={jev_pair.QUESTION: {"noul": 0.1 * i}}), 0.0

    res = jev_pair.rank(NOUL, "noul", task(20), send, {})
    assert res.failure is None
    assert res.subcall_failures == 1
    assert len(res.ranking) == 20


def test_the_score_arm_ranks_on_the_score_key():
    def send(url, body, headers):
        i = int(body["state"]["passage"].split()[-1])
        raw = body_of(answers={jev_pair.QUESTION: {"score": float(i)}})
        return 200, raw, 0.0

    res = jev_pair.rank(SCORE, "score", task(4), send, {})
    assert res.ranking == ["p3", "p2", "p1", "p0"]
