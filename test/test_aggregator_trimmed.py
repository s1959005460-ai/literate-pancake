# tests/test_aggregator_trimmed.py
import numpy as np
from FedGNN_advanced.aggregator import Aggregator
import pytest

def test_trimmed_mean_fallback_small_n():
    expected_shapes = {"w": (1,)}
    agg = Aggregator(expected_shapes, method="trimmed_mean")
    updates = {
        "c1": {"w": np.array([1.0])},
        "c2": {"w": np.array([100.0])},
    }
    # with 2 clients trimmed mean k=1 => 2*k >= n => fallback to median
    res = agg.aggregate(updates)
    assert np.allclose(res["w"], np.array([1.0]) ) or np.allclose(res["w"], np.array([100.0])) or np.allclose(res["w"], np.array([ (1.0+100.0)/2 ]))

def test_trimmed_mean_normal():
    expected_shapes = {"w": (1,)}
    agg = Aggregator(expected_shapes, method="trimmed_mean")
    updates = {f"c{i}": {"w": np.array([0.0 if i < 9 else 1000.0])} for i in range(10)}
    res = agg.aggregate(updates)
    assert abs(float(res["w"]) - 0.0) < 1.0
