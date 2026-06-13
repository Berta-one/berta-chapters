"""Retrieval evaluation metrics: Recall@k, MRR, nDCG."""

from __future__ import annotations

import math


def recall_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    top = set(ranking[:k])
    return len(top & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranking: list[int], relevant: set[int]) -> float:
    for rank, doc in enumerate(ranking, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: list[list[int]], rels: list[set[int]]) -> float:
    return sum(reciprocal_rank(r, s) for r, s in zip(rankings, rels)) / len(rankings)


def ndcg_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc in enumerate(ranking[:k], 1) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
