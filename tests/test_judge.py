"""The label-free judge, tested offline against captured and constructed bodies.

The judge exists to answer one question the gold labels cannot: if a team had no
labelled data, would an LLM judge — the LangWatch Instant Evals paradigm — reach
the same conclusion this benchmark reached?
"""

from __future__ import annotations

import json

from jevdemo import judge


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


def _ok(content: str, *, prompt=120, completion=8, cost=0.00002) -> bytes:
    # The float cost is deliberate: it is the provider's wire format. cost_to_micro
    # converts it to integer micro-dollars at the boundary, and nothing downstream
    # of parse() ever sees a float.
    return _body({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost},
    })


# --- request shape ---------------------------------------------------------

def test_request_never_contains_the_gold_label():
    """The entire experiment is void if the gold label leaks into the prompt."""
    _, body = judge.build_request("anthropic/claude-sonnet-5", "Charged twice.", "billing")
    blob = json.dumps(body)
    assert "gold" not in blob.lower()
    # The predicted label is present because the judge is grading it.
    assert "billing" in blob


def test_request_shows_the_department_definitions():
    """An operator writing this question would define the taxonomy. Withholding it
    would handicap the paradigm and make the comparison unfair to it."""
    _, body = judge.build_request("m", "t", "billing")
    system = body["messages"][0]["content"]
    for label in ("billing", "technical", "account", "sales", "other"):
        assert label in system


def test_request_asks_for_a_probability_not_only_a_verdict():
    """LangWatch returns calibrated probabilities per row; a bare yes/no would be
    testing a weaker product than the one being evaluated."""
    _, body = judge.build_request("m", "t", "billing")
    system = body["messages"][0]["content"]
    assert "probability" in system.lower() or "confidence" in system.lower()
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0


def test_request_targets_the_chat_endpoint():
    url, _ = judge.build_request("m", "t", "billing")
    assert url.endswith("/api/v1/chat/completions")


# --- parsing ---------------------------------------------------------------

def test_parses_a_correct_verdict_with_its_probability():
    v = judge.parse(_ok('{"correct": true, "probability": 0.93}'), 200)
    assert v.correct is True
    assert v.probability == 0.93
    assert v.failure is None
    assert v.input_tokens == 120
    assert v.reported_cost_micro == 20


def test_parses_an_incorrect_verdict():
    v = judge.parse(_ok('{"correct": false, "probability": 0.12}'), 200)
    assert v.correct is False
    assert v.probability == 0.12


def test_accepts_a_probability_only_response_and_derives_the_verdict():
    """Some judges answer with a score alone. 0.5 is the decision boundary."""
    v = judge.parse(_ok('{"probability": 0.8}'), 200)
    assert v.correct is True
    v = judge.parse(_ok('{"probability": 0.2}'), 200)
    assert v.correct is False


def test_rejects_a_probability_outside_the_unit_interval():
    v = judge.parse(_ok('{"correct": true, "probability": 7}'), 200)
    assert v.failure == judge.MALFORMED
    assert v.correct is None


def test_rejects_a_response_with_neither_verdict_nor_probability():
    v = judge.parse(_ok('{"reason": "looks fine"}'), 200)
    assert v.failure == judge.MALFORMED


def test_rejects_non_json_content():
    v = judge.parse(_ok("Yes, that routing looks correct."), 200)
    assert v.failure == judge.MALFORMED


def test_reports_an_http_failure_rather_than_guessing():
    v = judge.parse(b'{"error": "boom"}', 500)
    assert v.failure == judge.HTTP_ERROR
    assert v.correct is None


def test_reports_empty_body_as_malformed():
    v = judge.parse(b"", 200)
    assert v.failure == judge.MALFORMED


def test_missing_usage_is_a_failure_not_a_free_call():
    raw = _body({"choices": [{"message": {"content": '{"correct": true, "probability": 0.9}'}}]})
    v = judge.parse(raw, 200)
    assert v.failure == judge.MISSING_USAGE


def test_absent_cost_is_unverified_not_zero():
    raw = _body({
        "choices": [{"message": {"content": '{"correct": true, "probability": 0.9}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    })
    v = judge.parse(raw, 200)
    assert v.failure is None
    assert v.reported_cost_micro is None


# --- agreement -------------------------------------------------------------

def test_agreement_counts_matches_over_scored_rows_only():
    """A judge that failed to answer must not be scored as agreeing."""
    verdicts = [True, False, None, True]
    gold = [True, False, True, False]
    scored, agreed = judge.agreement(verdicts, gold)
    assert (scored, agreed) == (3, 2)


def test_agreement_rejects_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        judge.agreement([True], [True, False])


def test_agreement_on_an_empty_set_is_zero_over_zero():
    assert judge.agreement([], []) == (0, 0)
