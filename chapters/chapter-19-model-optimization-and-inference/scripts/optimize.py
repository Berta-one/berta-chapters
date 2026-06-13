"""
Model optimization primitives: quantization, pruning, distillation, sizing.

All pure NumPy and deterministic. These mirror what production stacks
(bitsandbytes, ONNX Runtime, torch.ao) do under the hood.
"""

from __future__ import annotations

import numpy as np


def quantize_int8(W):
    """Symmetric per-tensor int8 quantization. Returns (q_int8, scale)."""
    W = np.asarray(W, dtype=float)
    maxabs = float(np.max(np.abs(W))) or 1.0
    scale = maxabs / 127.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(float) * scale


def quantization_error(W):
    """Max absolute reconstruction error after int8 round-trip."""
    q, scale = quantize_int8(W)
    return float(np.max(np.abs(np.asarray(W, float) - dequantize_int8(q, scale))))


def magnitude_prune(W, sparsity: float):
    """Zero the smallest-magnitude `sparsity` fraction of weights."""
    W = np.asarray(W, dtype=float).copy()
    k = int(round(sparsity * W.size))
    if k <= 0:
        return W
    thresh = np.partition(np.abs(W).ravel(), k - 1)[k - 1]
    W[np.abs(W) <= thresh] = 0.0
    return W


def measure_sparsity(W) -> float:
    W = np.asarray(W)
    return float(np.mean(W == 0))


def softmax_with_temperature(logits, T: float = 1.0):
    """Temperature-scaled softmax. High T -> softer distribution (distillation)."""
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def distillation_targets(teacher_logits, T: float = 2.0):
    """Soft targets a student is trained to match."""
    return softmax_with_temperature(teacher_logits, T=T)


def model_size_bytes(n_params: int, precision: str = "float32") -> int:
    from config import BYTES

    return n_params * BYTES[precision]
