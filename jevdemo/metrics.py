"""Pure aggregation. No I/O, no clock, no network — everything is passed in."""

from __future__ import annotations

from dataclasses import dataclass, field

from jevdemo.arms import Arm
from jevdemo.pricing import Reconciliation, compute_micro, per_thousand_micro, reconcile
from jevdemo.result import Prediction

LATENCY_PASS = "latency"
BULK_PASS = "bulk"


@dataclass(frozen=True)
class Record:
    arm: str
    ticket_id: str
    pass_name: str
    gold: str
    prediction: Prediction
    elapsed_ms: float


@dataclass
class ArmMetrics:
    spec: Arm
    scored: int = 0
    correct: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    computed_cost_micro: int = 0
    reported_cost_micro: int = 0
    reported_cost_complete: bool = True
    latencies: list[float] = field(default_factory=list)

    @property
    def arm(self) -> str:
        return self.spec.name

    @property
    def model(self) -> str:
        return self.spec.model

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def reasoning(self) -> dict | None:
        return self.spec.reasoning

    @property
    def attempted(self) -> int:
        return self.scored + sum(self.failures.values())

    @property
    def accuracy(self) -> float | None:
        """Accuracy over every attempt, so a failure is never free. None on an empty set."""
        return None if self.attempted == 0 else self.correct / self.attempted

    @property
    def cost_per_1000_micro(self) -> int | None:
        return None if self.attempted == 0 else per_thousand_micro(
            self.computed_cost_micro, self.attempted
        )

    @property
    def p50_latency_ms(self) -> float | None:
        return _percentile(self.latencies, 0.50)

    @property
    def p95_latency_ms(self) -> float | None:
        return _percentile(self.latencies, 0.95)

    def reconciliation(self) -> Reconciliation:
        reported = (
            self.reported_cost_micro / 1_000_000 if self.reported_cost_complete else None
        )
        return reconcile(self.spec, self.input_tokens, self.output_tokens, reported)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank. With 20 samples per arm, interpolation would invent precision.
    index = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def aggregate(records: list[Record], arms: dict[str, Arm]) -> dict[str, ArmMetrics]:
    out: dict[str, ArmMetrics] = {}
    for record in records:
        arm = arms[record.arm]
        m = out.get(record.arm)
        if m is None:
            m = out[record.arm] = ArmMetrics(spec=arm)

        # Latency is reported only from the sequential pass. Timings taken under
        # concurrency measure the pool, not the model.
        if record.pass_name == LATENCY_PASS:
            if record.prediction.failure is None:
                m.latencies.append(record.elapsed_ms)
            continue

        p = record.prediction
        if p.failure is not None:
            m.failures[p.failure] = m.failures.get(p.failure, 0) + 1
            continue

        m.scored += 1
        if p.label == record.gold:
            m.correct += 1
        m.input_tokens += p.input_tokens
        m.output_tokens += p.output_tokens
        m.reasoning_tokens += p.reasoning_tokens
        m.computed_cost_micro += compute_micro(arm, p.input_tokens, p.output_tokens)
        if p.reported_cost_micro is None:
            m.reported_cost_complete = False
        else:
            m.reported_cost_micro += p.reported_cost_micro
    return out
