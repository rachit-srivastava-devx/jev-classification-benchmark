"""How a ranking is scored.

The question this report asks is not "did the model pick the single best
passage". It is "did the passages needed to answer the question survive into the
short list the generator will see". That is recall at k, and it is the headline.
nDCG is reported beside it because recall is blind to order and a reranker that
puts gold at position 5 is worse than one that puts it at position 1.

All four functions take the ranking as a list of document ids, most relevant
first, and the gold set as a set of ids. Duplicates in the ranking are tolerated
because a model that pads its answer must not be able to score twice.
"""

from __future__ import annotations

import math


def _dedup(ranking: list[str]) -> list[str]:
    """First occurrence wins. A repeated id is a wasted slot, not a second hit."""
    seen: set[str] = set()
    out: list[str] = []
    for doc in ranking:
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def recall_at_k(ranking: list[str], gold: set[str], k: int) -> float | None:
    """Share of the gold passages that appear in the top k.

    Returns None when the query has no gold passage at all: that is a hole in
    the relevance judgments, and scoring it zero would charge the model for it.
    An empty ranking scores 0.0, not None — failing to rank is a real failure.
    """
    if not gold:
        return None
    top = _dedup(ranking)[:k]
    return len(gold.intersection(top)) / len(gold)


def mrr(ranking: list[str], gold: set[str]) -> float:
    """Reciprocal rank of the first gold passage; 0.0 if none is ranked."""
    for i, doc in enumerate(_dedup(ranking), start=1):
        if doc in gold:
            return 1.0 / i
    return 0.0


def dcg(ranking: list[str], gold: set[str], k: int | None = None) -> float:
    """Discounted cumulative gain with binary relevance.

    BEIR's qrels for this corpus are binary, so a graded gain would be inventing
    a distinction the judgments do not make.
    """
    top = _dedup(ranking)
    if k is not None:
        top = top[:k]
    return sum(1.0 / math.log2(i + 1) for i, doc in enumerate(top, start=1) if doc in gold)


def ndcg_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    """DCG over the best achievable DCG for this gold set at this k.

    Returns 0.0 rather than raising when nothing was ranked, and 0.0 when there
    is no gold — callers that need to tell 'no judgment' from 'scored zero'
    should use `recall_at_k`, which reports None for the former.
    """
    if not gold:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg(ranking, gold, k) / ideal if ideal else 0.0
