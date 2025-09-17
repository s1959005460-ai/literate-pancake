# 文件: tests/test_aggregator.py
import numpy as np
from FedGNN_advanced.aggregator import Aggregator

def test_fedavg_weighted():
    expected_shapes = {"w": (1,)}
    agg = Aggregator(expected_shapes, method="fedavg")
    updates = {"c1":{"w":np.array([1.0])},"c2":{"w":np.array([3.0])}}
    weights = {"c1": 1.0, "c2": 3.0}
    res = agg.aggregate(updates, weights=weights)
    assert abs(float(res["w"]) - (1*1 + 3*3)/4) < 1e-6
