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


# --- markdown fences -------------------------------------------------------
#
# Found in the field, not imagined: claude-sonnet-5 wrapped its JSON in a
# ```json fence on 29 of 100 calls in the first full run, and the parser scored
# every one of them as malformed. That is a 27-point error in the arm's reported
# accuracy caused entirely by our own reader. A fence is presentation, not
# content, so it is stripped before parsing rather than counted as a failure.


def _ok(content: str):
    body = json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 70, "completion_tokens": 11, "cost": 0.000252},
    }).encode()
    return chat_arm.parse(body, 200)


def test_a_json_fence_is_stripped_not_counted_as_malformed():
    p = _ok('```json\n{"label": "billing"}\n```')
    assert p.failure is None
    assert p.label == "billing"


def test_a_bare_fence_without_a_language_tag_is_stripped():
    p = _ok('```\n{"label": "technical"}\n```')
    assert p.failure is None
    assert p.label == "technical"


def test_unfenced_content_is_unaffected():
    p = _ok('{"label": "account"}')
    assert p.failure is None
    assert p.label == "account"


def test_a_fence_around_prose_is_still_malformed():
    """Stripping the fence must not turn unparseable content into a silent pass."""
    p = _ok("```json\nI think this is a billing issue.\n```")
    assert p.failure == "malformed_output"


def test_backticks_inside_the_json_are_not_treated_as_a_fence():
    p = _ok('{"label": "other", "note": "```"}')
    assert p.failure is None
    assert p.label == "other"


def test_an_unterminated_fence_is_still_stripped():
    """Truncation at max_tokens can drop the closing fence; the label is still there."""
    p = _ok('```json\n{"label": "sales"}')
    assert p.failure is None
    assert p.label == "sales"
