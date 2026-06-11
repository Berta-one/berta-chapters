"""
Hybrid retrieval primitives.

Okapi BM25 (sparse), cosine dense retrieval, Reciprocal Rank Fusion, and a
phrase-aware reranker. Pure NumPy + stdlib, deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 ranking over a list of documents."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [tokenize(d) for d in corpus]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in self.docs]
        self.df: Counter = Counter()
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, i: int) -> float:
        dl = len(self.docs[i])
        s = 0.0
        for term in tokenize(query):
            f = self.tf[i].get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf(term) * f * (self.k1 + 1) / denom
        return s

    def rank(self, query: str) -> list[int]:
        scored = sorted(range(self.N), key=lambda i: self.score(query, i), reverse=True)
        return scored


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


def dense_rank(query_vec, doc_vecs) -> list[int]:
    sims = [cosine(query_vec, v) for v in doc_vecs]
    return sorted(range(len(doc_vecs)), key=lambda i: sims[i], reverse=True)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse ranked lists of doc ids into one robust order (RRF)."""
    scores: Counter = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return [doc for doc, _ in scores.most_common()]


def rerank(query: str, candidates: list[int], corpus: list[str]) -> list[int]:
    """Re-score candidates: BM25-style overlap plus an exact-phrase bonus."""
    q_tokens = set(tokenize(query))
    q_text = query.lower()

    def score(i: int) -> float:
        toks = tokenize(corpus[i])
        overlap = len(q_tokens & set(toks))
        phrase_bonus = 2.0 if q_text in corpus[i].lower() else 0.0
        return overlap + phrase_bonus

    return sorted(candidates, key=score, reverse=True)
