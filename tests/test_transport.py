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
