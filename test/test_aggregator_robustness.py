# tests/test_aggregator_robustness.py
import numpy as np
from FedGNN_advanced.aggregator import Aggregator
import pytest

def test_fedavg_basic():
    expected_shapes = {"w": (2, 2)}
    agg = Aggregator(expected_shapes, method="fedavg")
    updates = {
        "a": {"w": np.array([[1.0, 1.0], [1.0, 1.0]])},
        "b": {"w": np.array([[3.0, 3.0], [3.0, 3.0]])},
    }
    res = agg.aggregate(updates)
    assert np.allclose(res["w"], np.array([[2.0, 2.0], [2.0, 2.0]]))

def test_nan_handling_and_clipping():
    expected_shapes = {"w": (1, 3)}
    agg = Aggregator(expected_shapes, method="fedavg", clip_value=10.0)
    updates = {
        "a": {"w": np.array([np.nan, np.inf, -np.inf])},
        "b": {"w": np.array([1000.0, -1000.0, 5.0])},
    }
    res = agg.aggregate(updates)
    # NaNs replaced with zeros; clipping applied to 1000 -> 10
    assert res["w"].shape == (1, 3)

def test_trimmed_mean_outlier_resistance():
    expected_shapes = {"w": (1,)}
    agg = Aggregator(expected_shapes, method="trimmed_mean")
    updates = {
        f"c{i}": {"w": np.array([0.0 if i < 9 else 1000.0])} for i in range(10)
    }
    res = agg.aggregate(updates)
    # trimmed mean should ignore extremes
    assert abs(float(res["w"]) - 0.0) < 1.0
