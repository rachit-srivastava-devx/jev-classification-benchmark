"""A2 — the seam that keeps the whole suite offline."""
import pathlib

import pytest

from jevdemo import transport

PKG = pathlib.Path(__file__).resolve().parents[1] / "jevdemo"


def test_urllib_is_imported_in_exactly_one_module():
    """The property that makes every other test offline, asserted rather than trusted."""
    importers = sorted(
        p.name for p in PKG.glob("*.py") if "import urllib" in p.read_text()
    )
    assert importers == ["transport.py"], f"urllib leaked into {importers}"


def test_fixture_transport_replays_body_and_scripted_latency():
    t = transport.FixtureTransport({"https://x/y": (200, b'{"ok": true}')}, wall_ms=42.0)
    status, body, wall_ms = t("https://x/y", {}, {})
    assert (status, body, wall_ms) == (200, b'{"ok": true}', 42.0)


def test_fixture_transport_cycles_scripted_responses_in_order():
    """Reasoning negotiation needs a 400 then a 200 from the same URL."""
    t = transport.FixtureTransport(
        {"https://x/y": [(400, b'{"error": "nope"}'), (200, b'{"ok": true}')]}
    )
    assert t("https://x/y", {}, {})[0] == 400
    assert t("https://x/y", {}, {})[0] == 200


def test_fixture_transport_raises_on_an_unscripted_url():
    """A silent empty response would let a broken test pass."""
    t = transport.FixtureTransport({})
    with pytest.raises(KeyError):
        t("https://unscripted", {}, {})


def test_fixture_transport_records_the_requests_it_received():
    t = transport.FixtureTransport({"https://x/y": (200, b"{}")})
    t("https://x/y", {"model": "m"}, {"Authorization": "Bearer secret"})
    assert t.calls[0].url == "https://x/y"
    assert t.calls[0].body == {"model": "m"}


def test_recorded_headers_never_retain_the_bearer_token():
    """Fixtures get written to disk and tests get pasted into reports."""
    t = transport.FixtureTransport({"https://x/y": (200, b"{}")})
    t("https://x/y", {}, {"Authorization": "Bearer sk-or-v1-realkey"})
    assert "sk-or-v1-realkey" not in repr(t.calls[0])


# --- rate-limit handling -------------------------------------------------------------------
# A 429 is our own throttle, not a property of the model. It is the one status worth retrying:
# reporting it as a model failure would measure the rate limiter and call it accuracy.

class _Scripted:
    def __init__(self, responses):
        self.responses = list(responses)
        self.attempts = 0

    def __call__(self, url, body, headers):
        self.attempts += 1
        return self.responses.pop(0)


def test_a_429_is_retried_until_it_succeeds():
    slept = []
    inner = _Scripted([(429, b"slow down", 5.0), (429, b"slow down", 5.0), (200, b"{}", 12.0)])
    t = transport.Throttled(inner, sleep=slept.append)
    status, raw, wall_ms = t("https://x/y", {}, {})
    assert (status, raw) == (200, b"{}")
    assert wall_ms == 12.0, "latency must be the successful attempt, not the sum of the retries"
    assert inner.attempts == 3
    assert slept == [2.0, 4.0]


def test_retries_are_bounded_and_the_last_429_is_returned():
    inner = _Scripted([(429, b"slow down", 1.0)] * 9)
    t = transport.Throttled(inner, sleep=lambda s: None, max_attempts=4)
    status, _, _ = t("https://x/y", {}, {})
    assert status == 429
    assert inner.attempts == 4


def test_a_500_is_never_retried():
    inner = _Scripted([(500, b"boom", 1.0), (200, b"{}", 1.0)])
    t = transport.Throttled(inner, sleep=lambda s: None)
    assert t("https://x/y", {}, {})[0] == 500
    assert inner.attempts == 1


def test_a_200_costs_no_sleep_and_no_extra_attempt():
    slept = []
    inner = _Scripted([(200, b"{}", 1.0)])
    t = transport.Throttled(inner, sleep=slept.append)
    assert t("https://x/y", {}, {})[0] == 200
    assert (inner.attempts, slept) == (1, [])


def test_the_wrapper_counts_the_retries_it_spent():
    inner = _Scripted([(429, b"x", 1.0), (200, b"{}", 1.0)])
    t = transport.Throttled(inner, sleep=lambda s: None)
    t("https://x/y", {}, {})
    assert t.retries == 1
