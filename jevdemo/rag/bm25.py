"""BM25 over the corpus: the first stage of the pipeline.

This exists so the rerankers are handed the candidates a real system would hand
them. Twenty passages drawn at random from 50,000 would be trivially separable
and every model would score near-perfectly; twenty passages that BM25 ranked
highest are hard negatives, which is the whole difficulty of reranking.

Okapi BM25 with the standard k1=1.5, b=0.75. Pure stdlib, because the rest of
this project is and a retrieval index is not worth a dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Deliberately crude and deliberately fixed:
    the retriever must be identical for every model or the comparison is not one."""
    return _WORD.findall(text.lower())


class BM25:
    """An in-memory index. Build once per corpus, query many times.

    Building is O(total tokens). A search is O(sum of postings for the query's
    terms), which for a 50k-document corpus is milliseconds.

    Not thread-safe to build; safe to search concurrently once built, because
    every attribute is read-only after __init__.
    """

    def __init__(self, docs: dict[str, str]) -> None:
        if not docs:
            raise ValueError("BM25 over an empty corpus: nothing could ever be retrieved")
        self.doc_ids = list(docs)
        self.lengths: dict[str, int] = {}
        self.freqs: dict[str, Counter] = {}
        self.postings: dict[str, list[str]] = {}
        for doc_id, text in docs.items():
            terms = tokenize(text)
            self.lengths[doc_id] = len(terms)
            counts = Counter(terms)
            self.freqs[doc_id] = counts
            for term in counts:
                self.postings.setdefault(term, []).append(doc_id)
        total = sum(self.lengths.values())
        self.avg_len = total / len(docs) if total else 1.0

    def idf(self, term: str) -> float:
        """Inverse document frequency, floored at zero.

        Textbook BM25 idf turns negative once a term appears in more than half
        the corpus, which can rank a document that matches a common term *below*
        one that matches nothing. Clamping is the usual fix and is applied here.
        """
        n = len(self.postings.get(term, ()))
        if not n:
            return 0.0
        raw = math.log(1.0 + (len(self.doc_ids) - n + 0.5) / (n + 0.5))
        return max(0.0, raw)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Top k (doc_id, score), best first.

        Returns fewer than k, or nothing at all, when fewer documents match. An
        empty result is a real answer — it means the query shares no term with
        the corpus — and is never padded with arbitrary documents.
        """
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf(term)
            if not idf:
                continue
            for doc_id in self.postings.get(term, ()):
                tf = self.freqs[doc_id][term]
                norm = K1 * (1 - B + B * self.lengths[doc_id] / self.avg_len)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
