# FedGNN_advanced/topo_reg.py
"""
Topology regularization utilities with memory-aware batching.

Two modes:
 - Vectorized (fast): stacks client parameter arrays for each parameter when
   memory/shape makes stacking feasible.
 - Streaming (memory-efficient): incrementally accumulates weighted diffs per client.

API:
 - build_client_similarity(adj=None, client_embeddings=None, normalize=True)
 - apply_topo_reg_per_param(aggregate, client_states, similarity=None, lambda_topo=0.1,
                           max_stack_elements=64, max_param_bytes=50*1024*1024)

Parameters:
 - max_stack_elements: if number of clients <= this, stacking is allowed (subject to max_param_bytes)
 - max_param_bytes: if a parameter's flattened size * 8 bytes > this, use streaming for that parameter
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger("topo_reg")
logger.setLevel(logging.INFO)

def build_client_similarity(adj=None, client_embeddings=None, normalize=True):
    if adj is not None:
        A = np.array(adj, dtype=float)
        s = A.sum(axis=1)
        s = np.maximum(s, 0.0)
        if normalize and s.sum() > 0:
            s = s / float(s.sum())
        return s
    if client_embeddings is not None:
        mats = np.array(client_embeddings, dtype=float)
        centroid = mats.mean(axis=0, keepdims=True)
        num = (mats * centroid).sum(axis=1)
        den = np.linalg.norm(mats, axis=1) * (np.linalg.norm(centroid) + 1e-12)
        sims = num / (den + 1e-12)
        sims = np.maximum(sims, 0.0)
        if normalize and sims.sum() > 0:
            sims = sims / float(sims.sum())
        return sims
    return None

def apply_topo_reg_per_param(aggregate: Dict[str, np.ndarray],
                             client_states: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                             similarity: Optional[np.ndarray] = None,
                             lambda_topo: float = 0.1,
                             max_stack_elements: int = 64,
                             max_param_bytes: int = 50 * 1024 * 1024):
    """
    Apply topology regularization per parameter.

    Memory-aware: attempt vectorized stacking when safe; otherwise stream per client.
    """
    if client_states is None:
        logger.info("No client_states provided; topo_reg is a no-op")
        return aggregate

    client_ids = list(client_states.keys())
    K = len(client_ids)
    if K == 0:
        return aggregate

    # similarity vector
    if similarity is None:
        s = np.ones(K, dtype=float) / float(K)
    else:
        s = np.asarray(similarity, dtype=float)
        if s.shape[0] != K:
            logger.warning("similarity length mismatch; falling back to uniform")
            s = np.ones(K, dtype=float) / float(K)
        else:
            s = np.maximum(s, 0.0)
            if s.sum() > 0:
                s = s / float(s.sum())
            else:
                s = np.ones(K, dtype=float) / float(K)

    new_agg = {}

    # iterate parameters; do vectorized when safe
    for name, w_avg in aggregate.items():
        try:
            w_avg_arr = np.asarray(w_avg, dtype=np.float32)
            flat_size = int(np.prod(w_avg_arr.shape))
            bytes_needed = flat_size * 4  # float32 bytes
            can_stack = (K <= max_stack_elements) and (bytes_needed <= max_param_bytes)

            if can_stack:
                # collect client param arrays into stack
                mats = []
                for cid in client_ids:
                    p = client_states[cid].get(name)
                    if p is None:
                        mats.append(np.zeros_like(w_avg_arr, dtype=np.float32))
                    else:
                        mats.append(np.asarray(p, dtype=np.float32))
                # K x D stacked
                stacked = np.stack([m.reshape(-1) for m in mats], axis=0)  # (K, D)
                avg_flat = w_avg_arr.reshape(-1)
                diffs = stacked - avg_flat[None, :]
                # weighted sum across clients
                weighted = (s[:, None] * diffs).sum(axis=0)
                topo_shift = weighted.reshape(w_avg_arr.shape)
                new_agg[name] = (w_avg_arr - float(lambda_topo) * topo_shift).astype(np.float32)
            else:
                # streaming path: accumulate weighted diffs per-client to avoid stacking
                accum = None
                for i, cid in enumerate(client_ids):
                    p = client_states[cid].get(name)
                    if p is None:
                        continue
                    diff = np.asarray(p, dtype=np.float32) - w_avg_arr
                    weighted = s[i] * diff
                    if accum is None:
                        accum = weighted
                    else:
                        accum = accum + weighted
                if accum is None:
                    new_agg[name] = w_avg_arr
                else:
                    new_agg[name] = (w_avg_arr - float(lambda_topo) * accum).astype(np.float32)
        except Exception as e:
            logger.exception("topo_reg failed for param %s: %s", name, e)
            new_agg[name] = w_avg
    return new_agg
