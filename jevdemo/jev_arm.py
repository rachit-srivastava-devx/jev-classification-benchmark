"""The Jev protocol: typed questions in, typed answers out. No answer text."""

from __future__ import annotations

import json

from jevdemo import labels
from jevdemo.arms import Arm
from jevdemo.result import (
    HTTP_ERROR,
    INVALID_LABEL,
    MALFORMED,
    MISSING_USAGE,
    Prediction,
    cost_to_micro,
    failed,
)

QUESTION = "department"


def build_request(arm: Arm, ticket: str) -> tuple[str, dict]:
    body = {
        "model": arm.model,
        "state": {"ticket": ticket},
        "questions": {
            QUESTION: {
                "type": "choice",
                "instructions": "Which team should handle this ticket.",
                "criteria": labels.render_for_criteria(),
            }
        },
    }
    return arm.url, body


def parse(raw: bytes, status: int) -> Prediction:
    if status != 200:
        return failed(HTTP_ERROR, f"HTTP {status}: {raw[:200].decode('utf-8', 'replace')}")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return failed(MALFORMED, f"body is not JSON: {exc}")
    if not isinstance(body, dict):
        return failed(MALFORMED, "body is not an object")

    answer = (body.get("answers") or {}).get(QUESTION)
    if not isinstance(answer, dict) or "choice" not in answer:
        return failed(MALFORMED, f"no answers.{QUESTION}.choice in the response")

    choice = answer["choice"]
    if not labels.is_valid(choice):
        return failed(INVALID_LABEL, f"choice {choice!r} is not one of the five labels")

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return failed(MISSING_USAGE, "no usage block")
    try:
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError):
        return failed(MISSING_USAGE, f"incomplete usage block: {sorted(usage)}")

    confidence = answer.get("confidence")
    return Prediction(
        label=choice,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        reported_cost_micro=cost_to_micro(usage.get("cost")),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        failure=None,
        detail="",
    )
