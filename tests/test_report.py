"""The results file is the report's only source of truth, so its shape is tested.

Every claim in the written report must be traceable to a field here. That makes
two things non-negotiable: per-row predictions must be persisted (or the
label-free judge experiment has nothing to read), and every accuracy must carry
the interval that says whether it can be distinguished from its neighbour.
"""

from __future__ import annotations

import json

from jevdemo.arms import ARMS
from jevdemo.metrics import BULK_PASS, LATENCY_PASS, Record, aggregate
from jevdemo.report import rows, write_json
from jevdemo.result import INVALID_LABEL, Prediction, failed

BY_NAME = {a.name: a for a in ARMS}


def _pred(label="billing", *, tokens=(100, 10)) -> Prediction:
    return Prediction(
        label=label, input_tokens=tokens[0], output_tokens=tokens[1],
        reasoning_tokens=0, reported_cost_micro=None, confidence=None,
        failure=None, detail="",
    )


def _records(n_correct=8, n_wrong=2, arm="jev"):
    out = []
    for i in range(n_correct):
        out.append(Record(arm, f"t{i}", BULK_PASS, "billing", _pred("billing"), 100.0))
    for i in range(n_wrong):
        out.append(Record(arm, f"w{i}", BULK_PASS, "billing", _pred("sales"), 100.0))
    return out


def test_rows_carry_a_confidence_interval_on_accuracy():
    """A bare percentage invites the reader to rank arms that cannot be ranked."""
    results = aggregate(_records(), BY_NAME)
    row = rows(results)[0]
    assert row["accuracy"] == 0.8
    lo, hi = row["accuracy_ci95"]
    assert lo < 0.8 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_rows_carry_per_million_token_prices():
    """Cost per call is task-specific. Per-million prices let a reader transfer
    the result to a workload of a different size."""
    results = aggregate(_records(), BY_NAME)
    row = rows(results)[0]
    assert row["price_in_micro_per_mtok"] == BY_NAME["jev"].price_in_micro_per_mtok
    assert row["price_out_micro_per_mtok"] == BY_NAME["jev"].price_out_micro_per_mtok


def test_write_json_persists_every_prediction(tmp_path):
    """Without this the judge experiment and the per-item appendix are impossible,
    and the run would have to be paid for twice."""
    records = _records()
    results = aggregate(records, BY_NAME)
    out = tmp_path / "results.json"
    write_json(out, results, {}, records)

    payload = json.loads(out.read_text())
    assert len(payload["records"]) == len(records)

    first = payload["records"][0]
    assert first["arm"] == "jev"
    assert first["ticket_id"] == "t0"
    assert first["gold"] == "billing"
    assert first["predicted"] == "billing"
    assert first["pass"] == BULK_PASS
    assert first["elapsed_ms"] == 100.0
    assert first["failure"] is None


def test_persisted_records_keep_a_failure_and_its_null_label(tmp_path):
    records = [Record("jev", "t9", BULK_PASS, "billing",
                      failed(INVALID_LABEL, "model said 'refund'"), 120.0)]
    results = aggregate(records, BY_NAME)
    out = tmp_path / "results.json"
    write_json(out, results, {}, records)

    row = json.loads(out.read_text())["records"][0]
    assert row["predicted"] is None
    assert row["failure"] == INVALID_LABEL
    assert "refund" in row["detail"]


def test_latency_pass_records_are_persisted_and_marked(tmp_path):
    """They are excluded from accuracy, so the file must say which pass they came
    from or a reader recomputing accuracy from records would disagree with us."""
    records = _records() + [
        Record("jev", "t0", LATENCY_PASS, "billing", _pred("billing"), 90.0)
    ]
    results = aggregate(records, BY_NAME)
    out = tmp_path / "results.json"
    write_json(out, results, {}, records)

    payload = json.loads(out.read_text())
    passes = {r["pass"] for r in payload["records"]}
    assert passes == {BULK_PASS, LATENCY_PASS}
    # Accuracy still computed over the bulk pass only: 8 of 10, not 9 of 11.
    assert payload["arms"][0]["attempted"] == 10


def test_accuracy_is_none_and_interval_is_the_unit_interval_on_an_empty_arm():
    results = aggregate([], BY_NAME)
    assert results == {}


def test_each_record_carries_the_provider_cost(tmp_path):
    """A run-level cost divergence is only localisable if each call kept its bill."""
    records = _records()
    out = tmp_path / "r.json"
    write_json(out, aggregate(records, BY_NAME), {}, records)
    payload = json.loads(out.read_text())
    assert "reported_cost_micro" in payload["records"][0]
