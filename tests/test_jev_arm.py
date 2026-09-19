import json

import pytest

from jevdemo import jev_arm
from jevdemo.arms import by_name

ARM = by_name("jev")
TICKET = "My card was charged twice this month and the invoice only shows one line."


def read(name: str) -> bytes:
    with open(f"tests/fixtures/{name}", "rb") as fh:
        return fh.read()


def test_request_targets_the_decisions_endpoint():
    url, body = jev_arm.build_request(ARM, TICKET)
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert body["model"] == "typesafe/jev-1.13"


def test_request_carries_the_ticket_as_state_not_as_a_prompt():
    _, body = jev_arm.build_request(ARM, TICKET)
    assert body["state"] == {"ticket": TICKET}
    assert "messages" not in body
    assert "prompt" not in body


def test_criteria_are_the_five_label_definitions():
    _, body = jev_arm.build_request(ARM, TICKET)
    criteria = body["questions"]["department"]["criteria"]
    assert set(criteria) == {"billing", "technical", "account", "sales", "other"}
    assert body["questions"]["department"]["type"] == "choice"


def test_parses_the_real_captured_body():
    r = jev_arm.parse(read("jev_ok.json"), status=200)
    assert r.failure is None
    assert r.label == "billing"
    assert r.input_tokens == 395
    assert r.output_tokens == 52
    assert r.confidence == 1.0
    assert r.reported_cost_micro == 17  # 1.659e-05 USD, half-up


def test_an_out_of_set_choice_is_a_failure_and_is_never_coerced_to_other():
    body = json.loads(read("jev_ok.json"))
    body["answers"]["department"]["choice"] = "refunds"
    r = jev_arm.parse(json.dumps(body).encode(), status=200)
    assert r.failure == "invalid_label"
    assert r.label is None


def test_a_missing_answer_block_is_malformed_not_a_crash():
    r = jev_arm.parse(b'{"usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0}}', status=200)
    assert r.failure == "malformed_output"


def test_non_json_is_malformed():
    r = jev_arm.parse(read("empty.bin"), status=200)
    assert r.failure == "malformed_output"


def test_a_non_200_is_an_http_error_carrying_the_status():
    r = jev_arm.parse(read("http_500.json"), status=500)
    assert r.failure == "http_error"
    assert "500" in r.detail


@pytest.mark.parametrize("missing", ["input_tokens", "output_tokens"])
def test_missing_usage_is_reported_not_guessed(missing):
    body = json.loads(read("jev_ok.json"))
    del body["usage"][missing]
    r = jev_arm.parse(json.dumps(body).encode(), status=200)
    assert r.failure == "missing_usage"
