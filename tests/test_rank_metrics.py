"""Ranking metrics. These decide who wins the report, so they are tested against
hand-worked examples rather than against themselves.

The metric that matters for RAG is not 'was the top pick right'. It is 'did the
passages the answer needs survive into the short list the generator sees'.
"""

from __future__ import annotations

import pytest

from jevdemo.rag.rank_metrics import dcg, mrr, ndcg_at_k, recall_at_k


def test_recall_is_the_share_of_gold_that_survived_into_the_top_k():
    # Two gold passages; one of them is in the top 3.
    assert recall_at_k(["a", "b", "c", "d"], {"c", "z"}, 3) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], {"c", "d"}, 3) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], {"c", "d"}, 4) == 1.0


def test_recall_of_a_query_with_no_gold_is_undefined_not_zero():
    """Scoring it zero would punish a model for a hole in the judgments."""
    assert recall_at_k(["a"], set(), 3) is None


def test_recall_counts_each_gold_once_even_if_the_ranking_repeats_it():
    """A model that pads its list with the same id must not score twice."""
    assert recall_at_k(["c", "c", "c"], {"c", "d"}, 3) == 0.5


def test_mrr_is_one_over_the_rank_of_the_first_gold():
    assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert mrr(["c", "b", "a"], {"c"}) == 1.0
    assert mrr(["a", "b"], {"z"}) == 0.0


def test_dcg_discounts_by_log2_of_the_position():
    # One gold at position 1 and one at position 3: 1/log2(2) + 1/log2(4).
    assert dcg(["g", "x", "g2"], {"g", "g2"}) == pytest.approx(1.0 + 0.5)


def test_ndcg_is_one_when_every_gold_sits_at_the_top():
    assert ndcg_at_k(["g1", "g2", "x", "y"], {"g1", "g2"}, 4) == pytest.approx(1.0)


def test_ndcg_is_bounded_and_falls_as_gold_sinks():
    high = ndcg_at_k(["g", "x", "y", "z"], {"g"}, 4)
    low = ndcg_at_k(["x", "y", "z", "g"], {"g"}, 4)
    assert 0.0 <= low < high <= 1.0


def test_k_larger_than_the_ranking_is_not_an_error():
    """A model that returns three ids when asked for five is scored on three,
    not crashed on."""
    assert recall_at_k(["a", "b"], {"a"}, 5) == 1.0
    assert ndcg_at_k(["a", "b"], {"a"}, 5) == pytest.approx(1.0)


def test_an_empty_ranking_scores_zero_rather_than_raising():
    """A model that returned nothing usable has failed to rank, and failing to
    rank is a score of zero, not a missing row."""
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert mrr([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}, 5) == 0.0
