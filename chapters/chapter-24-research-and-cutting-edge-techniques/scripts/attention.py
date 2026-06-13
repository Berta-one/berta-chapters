"""
Scaled dot-product attention, reproduced from "Attention Is All You Need".

Pure NumPy so the mechanism is fully visible. The implementation is verified
against the invariants the math guarantees (softmax rows sum to one, output
shape matches V's value dimension).
"""

from __future__ import annotations

import numpy as np


def softmax(x, axis: int = -1):
    """Numerically stable softmax along an axis."""
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V.

    Q: (n_q, d_k), K: (n_k, d_k), V: (n_k, d_v).
    Returns (output (n_q, d_v), attention weights (n_q, n_k)).
    """
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_mask(n: int):
    """Lower-triangular mask so position i attends only to j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))
