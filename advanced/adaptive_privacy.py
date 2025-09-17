"""
AdaptivePrivacyAllocator

Implements an example adaptive privacy-budget allocation strategy.
This is meant as a reusable component to slot into your privacy accounting pipeline.

No magic numbers: defaults read from constants or passed in via constructor.
"""
from __future__ import annotations
from typing import Dict, Iterable, Any
import numpy as np

from .. import constants, logger

DEFAULT_TOTAL_EPSILON = 1.0  # fallback if not provided; caller should pass explicit value


class SensitivityAnalyzer:
    """
    Example sensitivity analyzer: computes L2 norm of client updates per-parameter.
    Implement more advanced analyses (per-feature sensitivity, fisher information, gradient variance, etc.)
    """

    def analyze(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        # returns a mapping param_name -> sensitivity_score (non-negative)
        # client_updates: client_id -> {param_name: numpy array or tensor-like}
        scores: Dict[str, float] = {}
        for cid, upd in client_updates.items():
            for pname, arr in upd.items():
                a = np.asarray(arr)
                norm = float(np.linalg.norm(a)) if a.size > 0 else 0.0
                scores[pname] = scores.get(pname, 0.0) + norm
        # average across clients
        n_clients = max(1, len(client_updates))
        for k in list(scores.keys()):
            scores[k] = scores[k] / n_clients
        return scores


class AdaptivePrivacyAllocator:
    def __init__(self, total_epsilon: float | None = None, sensitivity_analyzer: SensitivityAnalyzer | None = None):
        self.total_epsilon = float(total_epsilon) if total_epsilon is not None else float(getattr(constants, "DEFAULT_TOTAL_EPSILON", 1.0))
        self.sensitivity_analyzer = sensitivity_analyzer or SensitivityAnalyzer()

    def allocate_privacy_budget(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Allocate epsilon per parameter name based on sensitivities.

        Returns dict: param_name -> epsilon_share (sums to total_epsilon)
        """
        sensitivities = self.sensitivity_analyzer.analyze(client_updates)
        total = sum(sensitivities.values())
        if total <= 0.0:
            # fallback equal allocation
            n = max(1, len(sensitivities))
            equal = self.total_epsilon / n
            logger.secure_log("warning", "All sensitivities zero; using equal allocation", total=total, equal=equal)
            return {p: equal for p in sensitivities.keys()}
        allocation = {p: (s / total) * self.total_epsilon for p, s in sensitivities.items()}
        # ensure numerical sanity: sum to total_epsilon
        ssum = sum(allocation.values())
        if ssum == 0:
            # degenerate fallback
            n = max(1, len(allocation))
            return {p: self.total_epsilon / n for p in allocation.keys()}
        # normalize
        factor = self.total_epsilon / ssum
        for p in allocation:
            allocation[p] = float(allocation[p] * factor)
        return allocation
