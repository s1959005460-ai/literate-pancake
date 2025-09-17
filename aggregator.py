# 文件: FedGNN_advanced/aggregator.py
"""
Aggregator implementing FedAvg (weighted), Trimmed Mean and Median with robust checks.

Standards:
- FedAvg weighting per McMahan et al. (2017): support client sample counts.
- Trimmed-mean robust fallback and instrumentation per OWASP operational observability requirements.
"""
from __future__ import annotations

import logging
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
from prometheus_client import Counter, Gauge

logger = logging.getLogger("fedgnn.aggregator")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":%(message)s}'))
    logger.addHandler(h)
logger.setLevel("INFO")

AGG_ANOMALIES = Counter("fedgnn_aggregation_anomalies_total", "Aggregation anomalies")
AGG_ROUNDS = Counter("fedgnn_aggregation_rounds_total", "Aggregation rounds")
AGG_CLIENTS = Gauge("fedgnn_aggregation_clients_gauge", "Number of clients in last aggregation")

class Aggregator:
    def __init__(self, expected_shapes: Dict[str, Tuple[int, ...]], method: str = "fedavg", clip_value: Optional[float] = None):
        if method not in ("fedavg", "trimmed_mean", "median"):
            raise ValueError("unsupported aggregation method")
        self.expected_shapes = expected_shapes
        self.method = method
        self.clip_value = clip_value

    def _validate(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for cid, upd in client_updates.items():
            if not isinstance(upd, dict):
                logger.error(json.dumps({"msg":"invalid client update type","cid":cid}))
                raise TypeError("client update must be dict")
            norm = {}
            for name, arr in upd.items():
                if name not in self.expected_shapes:
                    logger.error(json.dumps({"msg":"unexpected param","cid":cid,"param":name}))
                    raise ValueError("unexpected param")
                if not hasattr(arr, "dtype"):
                    raise TypeError("parameter must be numpy array")
                if arr.shape != self.expected_shapes[name]:
                    logger.error(json.dumps({"msg":"shape mismatch","cid":cid,"param":name,"shape":str(arr.shape)}))
                    raise ValueError("shape mismatch")
                if not np.isfinite(arr).all():
                    logger.warning(json.dumps({"msg":"non-finite replaced","cid":cid,"param":name}))
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.clip_value is not None:
                    arr = np.clip(arr, -self.clip_value, self.clip_value)
                norm[name] = arr.astype(np.float32)
            out[cid] = norm
        return out

    def aggregate(self, client_updates: Dict[str, Dict[str, Any]], weights: Optional[Dict[str, float]] = None):
        """
        client_updates: mapping client_id -> {param_name: np.array}
        weights: optional mapping client_id -> float, representing local dataset sizes. If None, equal weighting used.
        """
        AGG_ROUNDS.inc()
        validated = self._validate(client_updates)
        n = len(validated)
        AGG_CLIENTS.set(n)
        if n == 0:
            AGG_ANOMALIES.inc()
            raise ValueError("no clients")
        keys = list(self.expected_shapes.keys())
        results = {}
        # If weights provided, normalize
        if weights:
            total_w = sum(weights.get(cid, 0.0) for cid in validated)
            if total_w <= 0:
                raise ValueError("invalid weights")
            norm_weights = {cid: weights.get(cid, 0.0) / total_w for cid in validated}
        else:
            norm_weights = None

        for key in keys:
            arrs = [validated[cid][key] for cid in validated]
            stacked = np.stack(arrs, axis=0)
            if self.method == "fedavg":
                if norm_weights is None:
                    res = np.mean(stacked, axis=0)
                else:
                    # weighted mean
                    ws = np.array([norm_weights[cid] for cid in validated], dtype=np.float32).reshape((len(validated),) + (1,) * (stacked.ndim - 1))
                    res = np.sum(stacked * ws, axis=0)
            elif self.method == "trimmed_mean":
                k = max(1, int(0.1 * stacked.shape[0]))
                if 2 * k >= stacked.shape[0]:
                    logger.warning(json.dumps({"msg":"trimmed_mean fallback to median","n":stacked.shape[0],"k":k}))
                    AGG_ANOMALIES.inc()
                    res = np.median(stacked, axis=0)
                else:
                    s = np.sort(stacked, axis=0)
                    res = np.mean(s[k: stacked.shape[0] - k], axis=0)
            else:
                res = np.median(stacked, axis=0)
            results[key] = res.astype(np.float32)
        return results
