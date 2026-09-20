"""What a ranker must do with a reply it cannot trust.

Both protocols are asked the same thing — put these passages in order — but they
answer in different shapes, and both can answer badly. A chat model can invent a
passage id, repeat one, or return prose instead of JSON. Jev can score only some
of the criteria. Every one of those is tested here, because the alternative is a
ranking that silently contains passages nobody ever retrieved.
"""

from __future__ import annotations

import json

import pytest

from jevdemo.arms import ARMS
from jevdemo.rag import chat_rank, jev_rank
from jevdemo.rag.rank_result import BAD_IDS, HTTP_ERROR, MALFORMED, MISSING_USAGE

BY_NAME = {a.name: a for a in ARMS}
TASK = {
    "query": "why did the central bank raise rates",
    "candidates": [
        {"id": "p0", "text": "Rates rose to cool inflation."},
        {"id": "p1", "text": "A recipe for sourdough bread."},
        {"id": "p2", "text": "Inflation targeting explained."},
    ],
}
IDS = ["p0", "p1", "p2"]


def _jev_body(probs, usage=None):
    return json.dumps({
        "answers": {jev_rank.QUESTION: {"choice": max(probs, key=probs.get),
                                        "probabilities": probs, "confidence": 1}},
        "usage": usage if usage is not None
        else {"input_tokens": 900, "output_tokens": 12, "cost": 0.00004},
    }).encode()


def _chat_body(content, usage=None):
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": usage if usage is not None
        else {"prompt_tokens": 900, "completion_tokens": 20, "cost": 0.0002},
    }).encode()


# --- the request ------------------------------------------------------------

def test_jev_puts_every_candidate_in_criteria_keyed_by_its_real_id():
    """The returned probability map is the ranking, so its keys must be the ids
    we will score against — not positions we would have to map back."""
    _, body = jev_rank.build_request(BY_NAME["jev"], TASK)
    criteria = body["questions"][jev_rank.QUESTION]["criteria"]
    assert set(criteria) == set(IDS)
    assert "sourdough" in criteria["p1"]
    assert body["state"]["query"] == TASK["query"]


def test_chat_request_carries_every_passage_and_names_the_ids_it_wants_back():
    _, body = chat_rank.build_request(BY_NAME["sonnet-5"], TASK, top_k=2)
    prompt = body["messages"][-1]["content"]
    for pid in IDS:
        assert pid in prompt
    assert "sourdough" in prompt
    assert "2" in prompt  # the number of ids it is asked for


# --- Jev replies ------------------------------------------------------------

def test_jev_ranking_is_probability_order():
    r = jev_rank.parse(_jev_body({"p0": 0.7, "p1": 0.05, "p2": 0.25}), 200, IDS)
    assert r.ranking == ["p0", "p2", "p1"]
    assert r.failure is None
    assert (r.input_tokens, r.output_tokens) == (900, 12)


def test_jev_ties_break_by_retrieval_order_not_by_dict_order():
    """Two passages scored identically must not be ordered by whatever order the
    JSON happened to arrive in, or the same reply could score differently twice."""
    r = jev_rank.parse(_jev_body({"p2": 0.4, "p0": 0.4, "p1": 0.2}), 200, IDS)
    assert r.ranking == ["p0", "p2", "p1"]


def test_jev_scoring_only_some_criteria_ranks_only_those():
    """Padding the rest back in would hand the model credit for passages it never
    ranked. A short ranking is an honest short ranking."""
    r = jev_rank.parse(_jev_body({"p1": 0.9}), 200, IDS)
    assert r.ranking == ["p1"]
    assert r.failure is None


def test_jev_probabilities_for_unknown_ids_are_dropped():
    r = jev_rank.parse(_jev_body({"p0": 0.5, "ghost": 0.5}), 200, IDS)
    assert r.ranking == ["p0"]
    assert r.dropped == 1


def test_jev_with_no_probability_map_is_malformed():
    body = json.dumps({"answers": {jev_rank.QUESTION: {"choice": "p0"}},
                       "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
    assert jev_rank.parse(body, 200, IDS).failure == MALFORMED


def test_jev_missing_usage_is_its_own_failure():
    assert jev_rank.parse(_jev_body({"p0": 1.0}, usage={}), 200, IDS).failure == MISSING_USAGE


# --- chat replies -----------------------------------------------------------

def test_chat_ranking_is_read_in_the_order_given():
    r = chat_rank.parse(_chat_body('{"ranking": ["p2", "p0"]}'), 200, IDS)
    assert r.ranking == ["p2", "p0"]
    assert r.failure is None
    assert (r.input_tokens, r.output_tokens) == (900, 20)


def test_chat_json_wrapped_in_prose_is_still_read():
    """Models add 'Here is the ranking:' whatever the instructions say. Throwing
    the reply away for that would measure instruction-following, not ranking."""
    r = chat_rank.parse(_chat_body('Sure!\n```json\n{"ranking": ["p0"]}\n```'), 200, IDS)
    assert r.ranking == ["p0"]


def test_chat_invented_ids_are_dropped_and_counted():
    """A hallucinated id in the ranking would otherwise be scored as a miss,
    which quietly reports a hallucination as an ordinary ranking error."""
    r = chat_rank.parse(_chat_body('{"ranking": ["p0", "p99", "p2"]}'), 200, IDS)
    assert r.ranking == ["p0", "p2"]
    assert r.dropped == 1


def test_chat_repeating_an_id_does_not_get_two_slots():
    r = chat_rank.parse(_chat_body('{"ranking": ["p0", "p0", "p1"]}'), 200, IDS)
    assert r.ranking == ["p0", "p1"]


def test_chat_reply_with_no_usable_id_is_a_failure_not_an_empty_ranking():
    """An empty ranking scores zero and looks like a bad reranker. It is not the
    same event as a model that never answered, and the report must tell them apart."""
    assert chat_rank.parse(_chat_body('{"ranking": ["nope"]}'), 200, IDS).failure == BAD_IDS


def test_chat_prose_only_reply_is_malformed():
    assert chat_rank.parse(_chat_body("I cannot rank these."), 200, IDS).failure == MALFORMED


@pytest.mark.parametrize("status", [400, 429, 500])
def test_any_non_200_is_an_http_failure_for_both_protocols(status):
    assert jev_rank.parse(b"nope", status, IDS).failure == HTTP_ERROR
    assert chat_rank.parse(b"nope", status, IDS).failure == HTTP_ERROR


def test_chat_request_uses_the_negotiated_reasoning_setting_over_the_arms_own():
    """Some providers 400 on {"enabled": false} and must be sent {"effort":
    "minimal"} instead. The runner negotiates that once; the builder must honour
    it, or every call after the negotiation repeats the error it just solved."""
    arm = BY_NAME["glm-5.3-flash-low"]
    _, body = chat_rank.build_request(arm, TASK, 2, {"effort": "minimal"})
    assert body["reasoning"] == {"effort": "minimal"}


def test_chat_request_with_negotiated_none_sends_no_reasoning_field_at_all():
    """None is a real outcome — 'this provider takes no reasoning field' — and
    must not fall back to the arm's default, which is what 400ed."""
    _, body = chat_rank.build_request(BY_NAME["glm-5.3-flash-low"], TASK, 2, None)
    assert "reasoning" not in body


def test_a_ranking_request_caps_its_reply_length():
    """Without a cap the provider reserves credit against the model's whole
    context window, which failed expensive models with HTTP 402 on a healthy
    balance. Regression guard: the cap must be sent, on every chat arm."""
    from jevdemo.rag.chat_rank import RANK_MAX_TOKENS
    for name in ("sonnet-5", "glm-5.3-flash-low"):
        arm = BY_NAME[name]
        _, body = chat_rank.build_request(arm, TASK, top_k=10)
        assert body["max_tokens"] == RANK_MAX_TOKENS


def test_the_reply_cap_leaves_room_for_a_full_ranking():
    """A reply cut off mid-array is unparseable. Ten ids plus JSON is a few
    hundred tokens, so the cap needs real headroom above that."""
    from jevdemo.rag.chat_rank import RANK_MAX_TOKENS
    assert RANK_MAX_TOKENS >= 1000
