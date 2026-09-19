"""The chat protocol: one JSON-mode completion per ticket, identical across all 19 arms."""

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

# 64 truncated gemini mid-JSON during probing (PROBE-RESULTS P2). 512 is slack, not a budget.
MAX_TOKENS = 512


def build_request(arm: Arm, ticket: str, reasoning: dict | None = None) -> tuple[str, dict]:
    body: dict = {
        "model": arm.model,
        "messages": [
            {"role": "system", "content": labels.render_for_prompt()},
            {"role": "user", "content": ticket},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    effective = reasoning if reasoning is not None else arm.reasoning
    if effective is not None:
        body["reasoning"] = effective
    return arm.url, body


def _rejects_reasoning(raw: bytes) -> bool:
    return b"reasoning" in raw.lower()


def strip_fence(content: str) -> str:
    """Remove a surrounding markdown code fence, if one is there.

    claude-sonnet-5 wrapped its JSON in a ```json fence on 29 of 100 calls in the
    first full run of this benchmark, and the parser counted every one as a
    malformed output — a 27-point error in that arm's accuracy, caused by our
    reader and not by the model. A fence is presentation; the content inside it
    is the answer.

    Only a fence that *opens the content* is stripped, so backticks appearing
    inside a JSON string are untouched. The closing fence is optional because
    truncation at max_tokens drops it while leaving a complete object behind.
    """
    text = content.strip()
    if not text.startswith("```"):
        return content
    # Drop the opening fence and its optional language tag, which runs to the newline.
    newline = text.find("\n")
    text = "" if newline == -1 else text[newline + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text


def parse(raw: bytes, status: int) -> Prediction:
    if status != 200:
        return failed(
            HTTP_ERROR,
            f"HTTP {status}: {raw[:200].decode('utf-8', 'replace')}",
            reasoning_rejected=status == 400 and _rejects_reasoning(raw),
        )
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return failed(MALFORMED, f"body is not JSON: {exc}")
    if not isinstance(body, dict):
        return failed(MALFORMED, "body is not an object")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return failed(MALFORMED, "no choices in the response")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return failed(MALFORMED, "empty message content")

    try:
        payload = json.loads(strip_fence(content))
    except ValueError as exc:
        return failed(MALFORMED, f"content is not JSON: {exc}")
    if not isinstance(payload, dict) or "label" not in payload:
        return failed(MALFORMED, "content has no label key")

    label = payload["label"]
    # Case-sensitive by design: "Billing" is a different string and the model was told the set.
    if not labels.is_valid(label):
        return failed(INVALID_LABEL, f"label {label!r} is not one of the five labels")

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return failed(MISSING_USAGE, "no usage block")
    try:
        input_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return failed(MISSING_USAGE, f"incomplete usage block: {sorted(usage)}")

    details = usage.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        reasoning_tokens = details["reasoning_tokens"]

    return Prediction(
        label=label,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        reported_cost_micro=cost_to_micro(usage.get("cost")),
        confidence=None,
        failure=None,
        detail="",
    )
