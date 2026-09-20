"""Reranking through Jev, by making the passages the criteria.

Jev answers a typed question by choosing among named criteria and returning a
probability for each one. Put one candidate passage under each criterion name
and that probability map is a ranking — which is the whole reason this comparison
is possible at all. No prompt engineering, no output format to negotiate: the
protocol returns a number per passage because that is what it always returns.
"""

from __future__ import annotations

import json

from jevdemo.arms import Arm
from jevdemo.rag.rank_result import (
    HTTP_ERROR,
    MALFORMED,
    MISSING_USAGE,
    RankResult,
    cost_to_micro,
    failed,
)

QUESTION = "best_passage"

INSTRUCTIONS = (
    "Which passage best answers the question in state.query? Score every passage "
    "by how much it helps answer that question."
)


def build_request(arm: Arm, task: dict, top_k: int | None = None,
                  reasoning: object = None) -> tuple[str, dict]:
    """`top_k` and `reasoning` are accepted and ignored.

    Jev scores every criterion either way, and has no reasoning setting to
    negotiate. Both are in the signature so the runner can call the two
    protocols identically instead of branching on which one it is talking to.
    """
    body = {
        "model": arm.model,
        "state": {"query": task["query"]},
        "questions": {
            QUESTION: {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {c["id"]: c["text"] for c in task["candidates"]},
            }
        },
    }
    return arm.url, body


def parse(raw: bytes, status: int, candidate_ids: list[str]) -> RankResult:
    if status != 200:
        return failed(HTTP_ERROR, f"HTTP {status}: {raw[:200].decode('utf-8', 'replace')}")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return failed(MALFORMED, f"body is not JSON: {exc}")
    if not isinstance(body, dict):
        return failed(MALFORMED, "body is not an object")

    answer = (body.get("answers") or {}).get(QUESTION)
    probs = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probs, dict) or not probs:
        return failed(MALFORMED, f"no answers.{QUESTION}.probabilities in the response")

    # Retrieval order is the tie-break. Without it two passages on the same
    # probability would be ordered by JSON key order, and the same reply could
    # score differently on two runs.
    position = {pid: i for i, pid in enumerate(candidate_ids)}
    scored = [(p, position[k]) for k, p in probs.items()
              if k in position and isinstance(p, (int, float))]
    dropped = len(probs) - len(scored)
    ranking = [candidate_ids[i] for _, i in sorted(scored, key=lambda t: (-t[0], t[1]))]

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return failed(MISSING_USAGE, "no usage block")
    try:
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError):
        return failed(MISSING_USAGE, f"incomplete usage block: {sorted(usage)}")

    return RankResult(
        ranking=ranking,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        reported_cost_micro=cost_to_micro(usage.get("cost")),
        dropped=dropped,
        failure=None,
        detail="",
    )
