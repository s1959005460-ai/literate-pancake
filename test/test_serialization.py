# 文件: tests/test_serialization.py
import pytest
import numpy as np
from FedGNN_advanced.utils.serialization import serialize_state_dict, deserialize_state_bytes, DeserializationError

def test_roundtrip_basic():
    s = {"w": np.array([[1.0,2.0]], dtype=np.float32)}
    b = serialize_state_dict(s, compress=False)
    out = deserialize_state_bytes(b, compressed=False)
    assert "w" in out
    assert out["w"].shape == (1,2)

def test_truncated_fails():
    s = {"w": np.array([1.0,2.0], dtype=np.float32)}
    b = serialize_state_dict(s)
    with pytest.raises(Exception):
        _ = deserialize_state_bytes(b[:-5])
