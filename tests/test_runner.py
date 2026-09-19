import json

from jevdemo import runner
from jevdemo.arms import by_name
from jevdemo.metrics import BULK_PASS, LATENCY_PASS, aggregate
from jevdemo.report import rows, write_json
from jevdemo.transport import FixtureTransport

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
TICKETS = [{"id": f"t{i:03d}", "text": "Charged twice this month.", "label": "billing"}
           for i in range(1, 101)]


def fixture(name):
    with open(f"tests/fixtures/{name}", "rb") as fh:
        return fh.read()


def test_dry_run_makes_no_calls_and_reports_the_plan(capsys):
    assert runner.main(["--dry-run", "--arm", "jev"]) == 0
    out = capsys.readouterr().out
    assert "1 arms x (20 latency + 100 bulk) = 120 calls" in out


def test_a_missing_key_exits_two_rather_than_running(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(runner, "ENV_PATH", type(runner.ENV_PATH)("/nonexistent/.env"))
    assert runner.main(["--arm", "jev"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_the_two_passes_produce_one_record_per_ticket_plus_the_latency_head():
    t = FixtureTransport({JEV_URL: [(200, fixture("jev_ok.json"))] * 120})
    records, outcome = run_silently(by_name("jev"), t)
    assert outcome.calls == 0  # jev is not negotiated
    assert len(records) == 120
    assert sum(1 for r in records if r.pass_name == LATENCY_PASS) == 20
    assert sum(1 for r in records if r.pass_name == BULK_PASS) == 100
    assert len(t.calls) == 120


def test_a_low_arm_negotiates_once_and_reuses_the_answer_for_every_ticket():
    bodies = [(400, fixture("chat_400_reasoning.json"))] + [(200, fixture("chat_ok.json"))] * 121
    t = FixtureTransport({CHAT_URL: bodies})
    records, outcome = run_silently(by_name("gemini-3.8-flash-low"), t)
    assert outcome.calls == 2
    assert outcome.reasoning == {"effort": "minimal"}
    assert len(records) == 120
    assert all(c.body.get("reasoning") == {"effort": "minimal"} for c in t.calls[1:])


def test_a_transport_failure_becomes_a_failed_record_not_an_exception():
    class Boom:
        calls = []

        def __call__(self, url, body, headers):
            from jevdemo.transport import TransportError
            raise TransportError("connection reset")

    records, _ = run_silently(by_name("jev"), Boom())
    assert len(records) == 120
    assert all(r.prediction.failure == "http_error" for r in records)


def test_results_json_carries_the_negotiated_setting_and_a_record_count(tmp_path):
    t = FixtureTransport({JEV_URL: [(200, fixture("jev_ok.json"))] * 120})
    records, outcome = run_silently(by_name("jev"), t)
    results = aggregate(records, {"jev": by_name("jev")})
    out = tmp_path / "results.json"
    write_json(out, results, {"jev": outcome}, records)
    payload = json.loads(out.read_text())
    assert payload["totals"]["records"] == 120
    assert payload["reasoning_negotiation"]["jev"] == {"reasoning": None, "calls": 0, "error": None}
    assert payload["arms"][0]["latency_samples"] == 20
    assert payload["arms"][0]["accuracy"] == 1.0


def test_the_report_rows_sort_cheapest_first():
    t = FixtureTransport({JEV_URL: [(200, fixture("jev_ok.json"))] * 120,
                          CHAT_URL: [(200, fixture("chat_ok.json"))] * 120})
    recs = run_silently(by_name("jev"), t)[0] + run_silently(by_name("opus-5"), t)[0]
    ordered = rows(aggregate(recs, {"jev": by_name("jev"), "opus-5": by_name("opus-5")}))
    assert [r["arm"] for r in ordered] == ["jev", "opus-5"]


def run_silently(arm, transport):
    return runner.run_arm(arm, TICKETS, transport, {}, log=lambda *a: None)
