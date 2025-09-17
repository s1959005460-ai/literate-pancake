"""
AdaptiveGradientCompressor

Provides adaptive magnitude-based compression (sparse top-k by threshold).
Expose compress/decompress APIs and a small helper to compute upstream compression_ratio.

No magic numbers inside: compression_ratio passed by caller.
"""

from __future__ import annotations
from typing import Tuple, Any, Dict
import numpy as np

from .. import constants, logger

class AdaptiveGradientCompressor:
    def __init__(self):
        # no default magic numbers here; user must supply compression_ratio or threshold
        pass

    def compress(self, gradients: Iterable[Any], compression_ratio: float = 0.01) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        """
        Compress list/iterable of gradient arrays into values + indices + original shape concatenation info.
        - gradients: iterable of numpy arrays / tensors (will be flattened)
        - compression_ratio: fraction of elements to keep (0<ratio<=1)
        Returns:
            values: 1D numpy array of kept values
            indices: 1D numpy array of integer indices into flattened vector
            original_shape: a tuple with single flattened length, useful for reconstruct
        """
        vecs = []
        for g in gradients:
            a = np.asarray(g, dtype=np.float32).ravel()
            vecs.append(a)
        if not vecs:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), (0,)
        flat = np.concatenate(vecs)
        n = flat.size
        k = max(1, int(max(1, n * float(compression_ratio))))
        # threshold by magnitude (top-k)
        if k >= n:
            inds = np.arange(n, dtype=np.int64)
            vals = flat.copy()
            return vals, inds, (n,)
        # use argpartition for efficiency
        thresh_index = n - k
        partitioned = np.argpartition(np.abs(flat), thresh_index)
        topk_idx = partitioned[thresh_index:]
        topk_idx_sorted = np.sort(topk_idx.astype(np.int64))
        vals = flat[topk_idx_sorted].astype(np.float32)
        return vals, topk_idx_sorted, (n,)

    def decompress(self, values: np.ndarray, indices: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct flattened vector of length original_shape[0] and return 1D numpy array.
        """
        n = int(original_shape[0])
        out = np.zeros(n, dtype=np.float32)
        if indices.size:
            out[indices] = values
        return out
