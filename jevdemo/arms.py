"""The arm registry: the only place a model id or a price appears.

Twenty arms. Every reasoning-capable model gets two — default and lowest — because
the gap between them is a configuration choice teams make without seeing its cost,
and a `-low` arm with no default counterpart would measure nothing to compare against.

Prices are integer micro-dollars per million tokens, read from OpenRouter's model
roster on 2026-09-20 and checked against the captured copy in
`tests/fixtures/models.json` by `test_arms.py`. They are not remembered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Requested first on a `-low` arm. Rejected with HTTP 400 by some providers
#: ("Reasoning is mandatory for this endpoint"), which is why `chat_arm` negotiates
#: a fallback to `{"effort": "minimal"}`. PROBE-RESULTS P3.
LOW_REASONING: dict = {"enabled": False}
LOW_REASONING_FALLBACK: dict = {"effort": "minimal"}

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class Arm:
    """One measured configuration.

    Attributes:
        name: Report identity. A `-low` suffix means a reasoning-reduced arm.
        model: The provider's model id.
        kind: Which protocol answers it.
        reasoning: The `reasoning` argument, or None for a default arm.
        price_in_micro_per_mtok: Integer micro-dollars per million input tokens.
        price_out_micro_per_mtok: Integer micro-dollars per million output tokens.
    """

    name: str
    model: str
    kind: Literal["jev", "chat"]
    reasoning: dict | None
    price_in_micro_per_mtok: int
    price_out_micro_per_mtok: int

    @property
    def url(self) -> str:
        """The endpoint this arm posts to."""
        return JEV_URL if self.kind == "jev" else CHAT_URL


ARMS: tuple[Arm, ...] = (
    # Jev is absent from /v1/models by design: a decisions model is not in the chat
    # catalogue. Its price is the one published on the model page. PROBE-RESULTS P1.
    Arm("jev", "typesafe/jev-1.13", "jev", None, 42_000, 0),
    Arm("sonnet-5", "anthropic/claude-sonnet-5", "chat", None, 2000000, 10000000),
    Arm("opus-5", "anthropic/claude-opus-5", "chat", None, 5000000, 25000000),
    Arm("gpt-5.3-codex", "openai/gpt-5.3-codex", "chat", None, 1750000, 14000000),
    Arm("gpt-5.3-codex-low", "openai/gpt-5.3-codex", "chat", LOW_REASONING, 1750000, 14000000),
    Arm("gemini-3.8-flash", "google/gemini-3.8-flash", "chat", None, 750000, 3750000),
    Arm("gemini-3.8-flash-low", "google/gemini-3.8-flash", "chat", LOW_REASONING, 750000, 3750000),
    Arm("glm-5.3-flash", "z-ai/glm-5.3-flash", "chat", None, 90000, 300000),
    Arm("glm-5.3-flash-low", "z-ai/glm-5.3-flash", "chat", LOW_REASONING, 90000, 300000),
    Arm("qwen3.8-flash", "qwen/qwen3.8-flash", "chat", None, 150000, 470000),
    Arm("qwen3.8-flash-low", "qwen/qwen3.8-flash", "chat", LOW_REASONING, 150000, 470000),
    Arm("deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash", "chat", None, 150000, 600000),
    Arm("deepseek-v4.1-flash-low", "deepseek/deepseek-v4.1-flash", "chat", LOW_REASONING, 150000, 600000),
    Arm("kimi-k3", "moonshotai/kimi-k3", "chat", None, 1700000, 8500000),
    Arm("kimi-k3-low", "moonshotai/kimi-k3", "chat", LOW_REASONING, 1700000, 8500000),
    Arm("grok-4.6", "x-ai/grok-4.6", "chat", None, 2000000, 6000000),
    Arm("grok-4.6-low", "x-ai/grok-4.6", "chat", LOW_REASONING, 2000000, 6000000),
    Arm("mistral-small", "mistralai/mistral-small-2603", "chat", None, 150000, 600000),
    Arm("mistral-small-low", "mistralai/mistral-small-2603", "chat", LOW_REASONING, 150000, 600000),
    Arm("llama-4-scout", "meta-llama/llama-4-scout", "chat", None, 100000, 300000),)

_BY_NAME = {arm.name: arm for arm in ARMS}


def by_name(name: str) -> Arm:
    """Look up an arm.

    Raises:
        KeyError: No arm by that name. Silently returning a default would run the
            wrong model for an entire pass.
    """
    if name not in _BY_NAME:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(_BY_NAME)}")
    return _BY_NAME[name]
