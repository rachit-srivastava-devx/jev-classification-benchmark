"""One-time, per-arm resolution of what "reasoning off" means for that provider.

`{"enabled": false}` works on gpt-5.3-codex and is a hard 400 on gemini, which
answers `Reasoning is mandatory for this endpoint and cannot be disabled.`
(PROBE-RESULTS P3). So the setting is negotiated once per arm, not per ticket,
and which one answered is recorded and printed — they are not the same setting.
"""

from __future__ import annotations

from dataclasses import dataclass

from jevdemo import chat_arm
from jevdemo.arms import LOW_REASONING, LOW_REASONING_FALLBACK, Arm
from jevdemo.transport import Transport, TransportError


@dataclass(frozen=True)
class Outcome:
    arm: str
    reasoning: dict | None
    calls: int
    error: str | None = None


def resolve(arm: Arm, ticket: str, transport: Transport, headers: dict | None = None) -> Outcome:
    if arm.kind != "chat" or arm.reasoning is None:
        return Outcome(arm.name, None, 0)

    calls = 0
    last = ""
    for setting in (LOW_REASONING, LOW_REASONING_FALLBACK):
        url, body = chat_arm.build_request(arm, ticket, reasoning=setting)
        calls += 1
        try:
            status, raw, _ = transport(url, body, headers or {})
        except TransportError as exc:
            return Outcome(arm.name, None, calls, f"transport failed: {exc}")
        result = chat_arm.parse(raw, status)
        if result.failure != "http_error":
            return Outcome(arm.name, setting, calls)
        last = result.detail
        if not result.reasoning_rejected:
            # A 500 or an unrelated 400 says nothing about the setting. Do not retry.
            return Outcome(arm.name, None, calls, last)
    return Outcome(arm.name, None, calls, f"both reasoning settings rejected: {last}")
