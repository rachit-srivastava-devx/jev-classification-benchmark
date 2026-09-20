"""Reranking through an ordinary chat completion.

The passages go into the prompt, the model is asked for a JSON list of ids in
order, and the reply is parsed back. This is how reranking with a general model
is actually done, so it is what the comparison has to measure — including the
part where the reply has to be dug out of prose and checked for invented ids.
"""

from __future__ import annotations

import json
import re

from jevdemo.arms import Arm
from jevdemo.rag.rank_result import (
    BAD_IDS,
    HTTP_ERROR,
    MALFORMED,
    MISSING_USAGE,
    RankResult,
    cost_to_micro,
    failed,
)

SYSTEM = (
    "You rank retrieved passages by how well they answer a question. "
    "Reply with JSON only: {\"ranking\": [\"id\", ...]}, best first."
)

#: Non-greedy so a reply containing several JSON blocks yields the first whole
#: one rather than everything between the first `{` and the last `}`.
_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


#: Sentinel: "use whatever the arm was defined with". `None` is a real value —
#: it means send no reasoning field at all — so it cannot double as "unset".
_FROM_ARM = object()

#: Reply budget for one ranking. The answer is `top_k` ids and a little JSON —
#: a few hundred tokens — so this is generous headroom, not a limit the task
#: bumps into. It is set explicitly for two reasons. Without it the provider
#: reserves credit against the model's *entire* context window, which made
#: expensive models fail with HTTP 402 on a healthy balance while actually
#: costing pennies. And an unbounded reply lets a reasoning model spend without
#: end on a task whose output is a list of ids.
#: Not lower: a reply cut off mid-array is unparseable, which is a real failure
#: mode already seen in this benchmark.
RANK_MAX_TOKENS = 1500


def build_request(arm: Arm, task: dict, top_k: int = 10,
                  reasoning: object = _FROM_ARM) -> tuple[str, dict]:
    passages = "\n\n".join(f"[{c['id']}] {c['text']}" for c in task["candidates"])
    prompt = (
        f"Question: {task['query']}\n\n"
        f"Passages:\n\n{passages}\n\n"
        f"Return the {top_k} passage ids that best answer the question, best "
        f"first, as {{\"ranking\": [\"id\", ...]}}. Use the ids exactly as given."
    )
    body = {
        "model": arm.model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": RANK_MAX_TOKENS,
    }
    setting = arm.reasoning if reasoning is _FROM_ARM else reasoning
    if setting is not None:
        body["reasoning"] = setting
    return arm.url, body


def _extract(content: str) -> list | None:
    """Find a ranking in the reply, whether or not it is wrapped in prose."""
    for candidate in [content, *(m.group(0) for m in _OBJECT.finditer(content))]:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("ranking"), list):
            return obj["ranking"]
    return None


def parse(raw: bytes, status: int, candidate_ids: list[str]) -> RankResult:
    if status != 200:
        return failed(HTTP_ERROR, f"HTTP {status}: {raw[:200].decode('utf-8', 'replace')}")
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

    raw_ranking = _extract(content)
    if raw_ranking is None:
        return failed(MALFORMED, f"no ranking object in content: {content[:120]!r}")

    known = set(candidate_ids)
    seen: set[str] = set()
    ranking: list[str] = []
    dropped = 0
    for item in raw_ranking:
        pid = str(item)
        if pid not in known:
            dropped += 1          # an id nobody retrieved: a hallucination, not a miss
        elif pid not in seen:
            seen.add(pid)
            ranking.append(pid)
    if not ranking:
        return failed(BAD_IDS, f"none of {len(raw_ranking)} returned ids exist")

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return failed(MISSING_USAGE, "no usage block")
    try:
        input_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return failed(MISSING_USAGE, f"incomplete usage block: {sorted(usage)}")
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)

    return RankResult(
        ranking=ranking,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        reported_cost_micro=cost_to_micro(usage.get("cost")),
        dropped=dropped,
        failure=None,
        detail="",
    )
