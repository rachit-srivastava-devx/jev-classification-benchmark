"""Two outputs: a stdout table for the operator, results.json for everything else."""

from __future__ import annotations

import json
from pathlib import Path

from jevdemo.metrics import ArmMetrics, Record
from jevdemo.stats import wilson_interval
from jevdemo.pricing import usd


def _fmt(value, spec="", dash="—"):
    return dash if value is None else format(value, spec)


def rows(results: dict[str, ArmMetrics]) -> list[dict]:
    out = []
    for m in sorted(results.values(), key=lambda m: m.computed_cost_micro):
        rec = m.reconciliation()
        out.append({
            "arm": m.arm,
            "model": m.model,
            "kind": m.kind,
            "reasoning": m.reasoning,
            "attempted": m.attempted,
            "scored": m.scored,
            "correct": m.correct,
            "accuracy": m.accuracy,
            "measured": m.measured,
            "failures": m.failures,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "reasoning_tokens": m.reasoning_tokens,
            "computed_cost_micro": m.computed_cost_micro,
            "reported_cost_micro": m.reported_cost_micro if m.reported_cost_complete else None,
            "cost_reconciled": rec.ok,
            "accuracy_ci95": list(wilson_interval(m.correct, m.attempted)),
            "price_in_micro_per_mtok": m.spec.price_in_micro_per_mtok,
            "price_out_micro_per_mtok": m.spec.price_out_micro_per_mtok,
            "cost_per_1000_micro": m.cost_per_1000_micro,
            "p50_latency_ms": m.p50_latency_ms,
            "p95_latency_ms": m.p95_latency_ms,
            "latency_samples": len(m.latencies),
        })
    return out


def print_table(results: dict[str, ArmMetrics]) -> None:
    head = f"{'arm':22} {'acc':>6} {'n':>4} {'p50 ms':>7} {'p95 ms':>7} {'in':>8} {'out':>8} {'think':>7} {'$/1k':>9}"
    print()
    print(head)
    print("-" * len(head))
    for r in rows(results):
        acc = "—" if r["accuracy"] is None else f"{r['accuracy'] * 100:.0f}%"
        print(
            f"{r['arm']:22} {acc:>6} {r['attempted']:>4} "
            f"{_fmt(r['p50_latency_ms'], '.0f'):>7} {_fmt(r['p95_latency_ms'], '.0f'):>7} "
            f"{r['input_tokens']:>8} {r['output_tokens']:>8} {r['reasoning_tokens']:>7} "
            f"{_fmt(r['cost_per_1000_micro'] and usd(r['cost_per_1000_micro'])):>9}"
        )
    unreconciled = [r["arm"] for r in rows(results) if not r["cost_reconciled"]]
    if unreconciled:
        print(f"\ncost not reconciled against the provider for: {', '.join(unreconciled)}")


def write_json(path: Path, results: dict[str, ArmMetrics], negotiated: dict,
               records: list[Record], rate_limit_retries: int = 0) -> None:
    payload = {
        "arms": rows(results),
        "reasoning_negotiation": {
            name: {"reasoning": o.reasoning, "calls": o.calls, "error": o.error}
            for name, o in negotiated.items()
        },
        "records": [
            {
                "arm": r.arm,
                "ticket_id": r.ticket_id,
                "pass": r.pass_name,
                "gold": r.gold,
                "predicted": r.prediction.label,
                "failure": r.prediction.failure,
                "detail": r.prediction.detail,
                "input_tokens": r.prediction.input_tokens,
                "output_tokens": r.prediction.output_tokens,
                "reasoning_tokens": r.prediction.reasoning_tokens,
                # Serialised per call so a divergence between the provider's bill
                # and the list price can be localised to the calls that caused it.
                # Only arm-level totals were stored for the first corrected run,
                # which is why that run can name the divergence but not explain it.
                "reported_cost_micro": r.prediction.reported_cost_micro,
                "confidence": r.prediction.confidence,
                "elapsed_ms": r.elapsed_ms,
            }
            for r in records
        ],
        "totals": {
            "records": len(records),
            "negotiation_calls": sum(o.calls for o in negotiated.values()),
            "computed_cost_micro": sum(m.computed_cost_micro for m in results.values()),
            "rate_limit_retries": rate_limit_retries,
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
