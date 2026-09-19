import json

import pytest

from jevdemo import chat_arm
from jevdemo.arms import ARMS, by_name

ARM = by_name("sonnet-5")
TICKET = "My card was charged twice this month and the invoice only shows one line."


def read(name: str) -> bytes:
    with open(f"tests/fixtures/{name}", "rb") as fh:
        return fh.read()


def test_request_targets_chat_completions_with_json_mode():
    url, body = chat_arm.build_request(ARM, TICKET)
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body["response_format"] == {"type": "json_object"}
    assert body["usage"] == {"include": True}
    assert body["max_tokens"] == 512


def test_every_chat_arm_sends_a_byte_identical_body_apart_from_model_and_reasoning():
    chat = [a for a in ARMS if a.kind == "chat"]
    assert len(chat) == 19
    shapes = set()
    for arm in chat:
        _, body = chat_arm.build_request(arm, TICKET)
        body.pop("model")
        body.pop("reasoning", None)
        shapes.add(json.dumps(body, sort_keys=True))
    assert len(shapes) == 1


def test_reasoning_is_sent_only_for_low_arms():
    _, default_body = chat_arm.build_request(by_name("gpt-5.3-codex"), TICKET)
    _, low_body = chat_arm.build_request(by_name("gpt-5.3-codex-low"), TICKET)
    assert "reasoning" not in default_body
    assert low_body["reasoning"] == {"enabled": False}


def test_reasoning_override_replaces_the_arm_default():
    _, body = chat_arm.build_request(
        by_name("gemini-3.8-flash-low"), TICKET, reasoning={"effort": "minimal"}
    )
    assert body["reasoning"] == {"effort": "minimal"}


def test_parses_the_real_captured_body():
    r = chat_arm.parse(read("chat_ok.json"), status=200)
    assert r.failure is None
    assert r.label == "billing"
    assert r.input_tokens == 71
    assert r.output_tokens == 11
    assert r.reasoning_tokens == 0
    assert r.reported_cost_micro == 252


def test_reasoning_tokens_are_read_from_the_details_block():
    r = chat_arm.parse(read("chat_reasoning.json"), status=200)
    assert r.reasoning_tokens == 171
    assert r.output_tokens == 176


def test_a_capitalised_label_is_invalid_and_is_not_normalised():
    r = chat_arm.parse(read("chat_wrong_case.json"), status=200)
    assert r.failure == "invalid_label"
    assert r.label is None


def test_a_label_outside_the_five_is_invalid():
    assert chat_arm.parse(read("chat_unknown_label.json"), status=200).failure == "invalid_label"


@pytest.mark.parametrize("name", ["chat_truncated.json", "chat_no_label.json", "empty.bin"])
def test_unusable_content_is_malformed(name):
    assert chat_arm.parse(read(name), status=200).failure == "malformed_output"


def test_a_200_without_usage_is_reported_not_assumed_free():
    r = chat_arm.parse(read("chat_no_usage.json"), status=200)
    assert r.failure == "missing_usage"


def test_a_400_naming_reasoning_is_classified_so_the_runner_can_renegotiate():
    r = chat_arm.parse(read("chat_400_reasoning.json"), status=400)
    assert r.failure == "http_error"
    assert r.reasoning_rejected is True


def test_an_unrelated_400_is_not_a_reasoning_rejection():
    r = chat_arm.parse(b'{"error": {"message": "context length exceeded"}}', status=400)
    assert r.reasoning_rejected is False
