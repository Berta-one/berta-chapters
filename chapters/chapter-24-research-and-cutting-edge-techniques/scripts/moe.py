"""
Mixture-of-experts top-k routing, the sparse-scaling trick behind many
frontier LLMs. Each token is routed to its top-k experts; their outputs are
combined with renormalized gate weights. Pure NumPy.
"""

from __future__ import annotations

import numpy as np

from attention import softmax


def top_k_gating(x, W_gate, k: int = 2):
    """Route each row of x to its top-k experts.

    x: (n, d), W_gate: (d, n_experts).
    Returns (expert_indices (n, k), gate_weights (n, k)) where each row of
    gate_weights is a probability distribution over the selected experts.
    """
    x = np.asarray(x, dtype=float)
    logits = x @ np.asarray(W_gate, dtype=float)         # (n, n_experts)
    idx = np.argsort(-logits, axis=1)[:, :k]             # top-k per row
    topk_logits = np.take_along_axis(logits, idx, axis=1)
    weights = softmax(topk_logits, axis=1)               # renormalize over k
    return idx, weights


def moe_forward(x, W_gate, experts, k: int = 2):
    """Sparse MoE forward.

    experts: list of callables mapping (n, d) -> (n, d_out).
    Returns (output (n, d_out), expert_indices, gate_weights).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    idx, weights = top_k_gating(x, W_gate, k)
    expert_outputs = [np.asarray(e(x), dtype=float) for e in experts]
    d_out = expert_outputs[0].shape[1]
    y = np.zeros((n, d_out))
    for i in range(n):
        for j in range(k):
            y[i] += weights[i, j] * expert_outputs[idx[i, j]][i]
    return y, idx, weights
