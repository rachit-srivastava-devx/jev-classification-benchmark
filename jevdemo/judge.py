"""Label-free evaluation, in the shape LangWatch Instant Evals sells.

The paradigm: instead of labelling a dataset, you write the question in plain
words — "was this ticket routed to the right department?" — and an LLM judge
scores every row, returning a calibrated probability. LangWatch reports 97%
agreement with human labels on their own judge.

This module reproduces that paradigm against arms whose gold labels we already
hold but never show the judge. That makes the interesting question answerable:
would a team with no labelled data have reached this benchmark's conclusion?

The judge is deliberately given every advantage an operator would give it — the
full department taxonomy, a temperature of zero, JSON mode, and a strong model —
because a weak judge failing would say nothing about the paradigm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from jevdemo import labels
from jevdemo.arms import CHAT_URL
from jevdemo.result import HTTP_ERROR, MALFORMED, MISSING_USAGE, cost_to_micro

#: A strong judge, so that a negative result indicts the paradigm and not the
#: model. Named here rather than in the runner so the report can cite it.
DEFAULT_JUDGE = "anthropic/claude-sonnet-5"

MAX_TOKENS = 256

#: Verdict boundary when a judge returns a probability but no explicit verdict.
DECISION_BOUNDARY = 0.5

SYSTEM = (
    "You are grading how a support ticket was routed. You will be shown the ticket "
    "and the department it was routed to. Decide whether that routing is correct.\n\n"
    "The departments are:\n"
    f"{labels.render_for_prompt()}\n\n"
    'Reply with JSON only: {"correct": true or false, "probability": a number from '
    "0 to 1 giving your confidence that the routing shown is correct}."
)


@dataclass(frozen=True)
class Verdict:
    """One judged row. `correct` is the judge's opinion, never the gold label."""

    correct: bool | None
    probability: float | None
    input_tokens: int
    output_tokens: int
    reported_cost_micro: int | None
    failure: str | None
    detail: str = ""


def _failed(kind: str, detail: str = "") -> Verdict:
    return Verdict(None, None, 0, 0, None, kind, detail)


def build_request(model: str, ticket: str, predicted_label: str) -> tuple[str, dict]:
    """The judge sees the ticket and the predicted label. It never sees the gold
    label — that is the experiment's one inviolable rule."""
    return CHAT_URL, {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Ticket:\n{ticket}\n\nRouted to: {predicted_label}",
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }


def parse(raw: bytes, status: int) -> Verdict:
    """Same refusal order as the classification arms: transport, envelope,
    content, then accounting. A judge that did not answer is a failure, never a
    default verdict — scoring an absent answer as agreement would inflate exactly
    the number this experiment reports."""
    if status != 200:
        return _failed(HTTP_ERROR, f"status {status}: {raw[:200]!r}")

    try:
        envelope = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return _failed(MALFORMED, f"envelope not JSON: {exc}")

    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        return _failed(MALFORMED, f"no message content: {exc}")

    try:
        answer = json.loads(content)
    except (ValueError, TypeError) as exc:
        return _failed(MALFORMED, f"content not JSON: {exc}")
    if not isinstance(answer, dict):
        return _failed(MALFORMED, f"content is {type(answer).__name__}, not an object")

    probability = answer.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        probability = None
    elif not 0.0 <= probability <= 1.0:
        return _failed(MALFORMED, f"probability {probability} outside [0, 1]")

    verdict = answer.get("correct")
    if not isinstance(verdict, bool):
        verdict = None if probability is None else probability >= DECISION_BOUNDARY

    if verdict is None and probability is None:
        return _failed(MALFORMED, f"neither verdict nor probability in {content[:120]!r}")

    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return _failed(MISSING_USAGE, "no usage block")
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return _failed(MISSING_USAGE, f"token counts absent from {usage!r}")

    return Verdict(
        correct=verdict,
        probability=probability,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reported_cost_micro=cost_to_micro(usage.get("cost")),
        failure=None,
    )


def agreement(verdicts: list[bool | None], gold_correct: list[bool]) -> tuple[int, int]:
    """Returns (scored, agreed).

    Rows the judge failed to score are excluded from both, and the report prints
    `scored` alongside the rate so the denominator is never implied.
    """
    if len(verdicts) != len(gold_correct):
        raise ValueError(f"length mismatch: {len(verdicts)} vs {len(gold_correct)}")
    scored = 0
    agreed = 0
    for verdict, gold in zip(verdicts, gold_correct):
        if verdict is None:
            continue
        scored += 1
        if verdict is gold:
            agreed += 1
    return scored, agreed
