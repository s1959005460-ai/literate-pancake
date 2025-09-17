# FedGNN_advanced/aggregator.py
"""
Robust aggregator with validation, NaN handling, and multiple aggregation strategies.
This file is a production-ready replacement: input validation, per-coordinate clipping,
and logging included.

CRITICAL FIX: Validate dtype and shapes, enforce finite values, and provide robust
aggregation methods (fedavg, trimmed mean, median) with clipping.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np

logger = logging.getLogger("fedgnn.aggregator")
logger.setLevel(logging.INFO)


class Aggregator:
    def __init__(self, expected_shapes: Dict[str, Tuple[int, ...]],
                 method: str = "fedavg", clip_value: Optional[float] = None):
        self.expected_shapes = expected_shapes
        if method not in ("fedavg", "trimmed_mean", "median"):
            raise ValueError("unsupported aggregation method")
        self.method = method
        self.clip_value = clip_value

    def _validate_and_normalize(self, client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for cid, upd in client_updates.items():
            if not isinstance(upd, dict):
                raise TypeError(f"client {cid} update must be a dict")
            norm = {}
            for name, arr in upd.items():
                if name not in self.expected_shapes:
                    raise ValueError(f"unexpected parameter name {name} from client {cid}")
                if not isinstance(arr, np.ndarray):
                    raise TypeError(f"parameter {name} from client {cid} is not numpy array")
                # convert to float32 if numeric
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32)
                # shape check
                if arr.shape != self.expected_shapes[name]:
                    raise ValueError(f"client {cid} parameter {name} shape mismatch {arr.shape} != {self.expected_shapes[name]}")
                # finite check
                if not np.isfinite(arr).all():
                    logger.warning("non-finite values found in client %s param %s - replacing with zeros", cid, name)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # clipping
                if self.clip_value is not None:
                    arr = np.clip(arr, -self.clip_value, self.clip_value)
                norm[name] = arr.astype(np.float32)
            out[cid] = norm
        return out

    def aggregate(self, client_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates according to configured method.
        Returns dict name -> numpy array
        """
        validated = self._validate_and_normalize(client_updates)
        # collect per-key lists
        keys = list(self.expected_shapes.keys())
        results: Dict[str, np.ndarray] = {}
        for key in keys:
            arrs = [validated[cid][key] for cid in validated]
            stacked = np.stack(arrs, axis=0)  # shape (num_clients, *param_shape)
            if self.method == "fedavg":
                res = np.mean(stacked, axis=0)
            elif self.method == "trimmed_mean":
                # trim top/bottom 10% by default
                k = max(1, int(0.1 * stacked.shape[0]))
                # sort along client axis
                s = np.sort(stacked, axis=0)
                res = np.mean(s[k: stacked.shape[0] - k], axis=0)
            elif self.method == "median":
                res = np.median(stacked, axis=0)
            else:
                raise RuntimeError("unsupported aggregation method encountered")
            results[key] = res.astype(np.float32)
        return results


# Simple test helper if run as script
if __name__ == "__main__":
    expected_shapes = {"weights": (2, 2), "bias": (2,)}
    aggregator = Aggregator(expected_shapes, method="fedavg", clip_value=1e3)
    client_updates = {
        "c1": {"weights": np.ones((2, 2), dtype=np.float32), "bias": np.array([1.0, 2.0], dtype=np.float32)},
        "c2": {"weights": np.ones((2, 2), dtype=np.float32) * 3, "bias": np.array([3.0, 4.0], dtype=np.float32)},
    }
    print(aggregator.aggregate(client_updates))
