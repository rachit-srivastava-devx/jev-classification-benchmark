
from jevdemo import negotiate
from jevdemo.arms import LOW_REASONING, LOW_REASONING_FALLBACK, by_name
from jevdemo.transport import FixtureTransport

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
TICKET = "Charged twice."


def fixture(name: str) -> bytes:
    with open(f"tests/fixtures/{name}", "rb") as fh:
        return fh.read()


OK = fixture("chat_ok.json")
REJECT = fixture("chat_400_reasoning.json")


def test_a_non_low_arm_is_not_negotiated_and_costs_no_call():
    t = FixtureTransport({})
    outcome = negotiate.resolve(by_name("sonnet-5"), TICKET, t)
    assert outcome.reasoning is None
    assert outcome.calls == 0
    assert t.calls == []


def test_a_low_arm_that_accepts_enabled_false_keeps_it_after_one_call():
    t = FixtureTransport({CHAT_URL: (200, OK)})
    outcome = negotiate.resolve(by_name("gpt-5.3-codex-low"), TICKET, t)
    assert outcome.reasoning == LOW_REASONING
    assert outcome.calls == 1
    assert t.calls[0].body["reasoning"] == LOW_REASONING


def test_a_400_naming_reasoning_falls_back_to_minimal_effort():
    t = FixtureTransport({CHAT_URL: [(400, REJECT), (200, OK)]})
    outcome = negotiate.resolve(by_name("gemini-3.8-flash-low"), TICKET, t)
    assert outcome.reasoning == LOW_REASONING_FALLBACK
    assert outcome.calls == 2
    assert t.calls[1].body["reasoning"] == LOW_REASONING_FALLBACK


def test_a_500_is_never_retried_and_the_arm_is_left_unresolved():
    t = FixtureTransport({CHAT_URL: (500, fixture("http_500.json"))})
    outcome = negotiate.resolve(by_name("gpt-5.3-codex-low"), TICKET, t)
    assert outcome.calls == 1
    assert outcome.reasoning is None
    assert outcome.error is not None


def test_both_settings_rejected_leaves_the_arm_unresolved_rather_than_guessing():
    t = FixtureTransport({CHAT_URL: [(400, REJECT), (400, REJECT)]})
    outcome = negotiate.resolve(by_name("gemini-3.8-flash-low"), TICKET, t)
    assert outcome.calls == 2
    assert outcome.reasoning is None
    assert outcome.error is not None
