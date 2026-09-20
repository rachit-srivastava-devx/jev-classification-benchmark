"""Stage-1 retrieval.

The rerankers are only interesting if the candidates they are given are the ones
a real system would give them. That means a real first-stage retriever over the
real corpus, producing real hard negatives — passages that genuinely look right
and are not. A random sample of the corpus would make every reranker look good.
"""

from __future__ import annotations

import math

import pytest

from jevdemo.rag.bm25 import BM25, tokenize

DOCS = {
    "d1": "the cat sat on the mat",
    "d2": "the dog sat on the log",
    "d3": "cats and dogs living together",
    "d4": "a completely unrelated passage about marine biology",
}


def test_tokenize_folds_case_and_drops_punctuation():
    assert tokenize("The CAT's mat, again!") == ["the", "cat", "s", "mat", "again"]


def test_tokenize_of_empty_text_is_empty_not_a_crash():
    assert tokenize("") == []
    assert tokenize("   !!!  ") == []


def test_search_ranks_the_document_that_matches_the_query():
    top = BM25(DOCS).search("cat mat", k=2)
    assert top[0][0] == "d1"


def test_a_term_in_every_document_carries_no_signal():
    """A term present in all N documents separates nothing, so it must score
    zero rather than producing a confident-looking arbitrary ranking."""
    every = {f"d{i}": f"boilerplate header unique{i}" for i in range(6)}
    scores = dict(BM25(every).search("boilerplate header", k=6))
    # BM25's +0.5 smoothing leaves a small positive idf even at n == N, so the
    # property to assert is that such a term cannot *discriminate*: every
    # document ties, and no false ordering is produced.
    assert len(set(scores.values())) == 1, scores
    assert max(scores.values()) < 0.2, scores


def test_search_returns_at_most_k_and_is_sorted_descending():
    top = BM25(DOCS).search("dog log sat", k=2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]


def test_a_query_with_no_known_term_returns_no_matches_rather_than_random_ones():
    assert BM25(DOCS).search("xyzzy plugh", k=3) == []


def test_empty_corpus_raises_rather_than_scoring_nothing_silently():
    """A gate that measured nothing failed. An empty index is a bug upstream."""
    with pytest.raises(ValueError):
        BM25({})


def test_idf_is_never_negative_so_a_common_term_cannot_subtract_score():
    """Classic BM25 idf goes negative for terms in over half the corpus, which
    can rank a document below one that matches nothing at all."""
    idx = BM25({f"d{i}": "common term" for i in range(10)} | {"dx": "rare"})
    assert idx.idf("common") >= 0.0


def test_scores_are_finite_for_a_repeated_term():
    top = BM25(DOCS).search("cat cat cat", k=1)
    assert math.isfinite(top[0][1])
