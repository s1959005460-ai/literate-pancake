# 文件: tests/fuzz_tests.py
from hypothesis import given, strategies as st
from FedGNN_advanced.utils.serialization import deserialize_state_bytes, serialize_state_dict
import numpy as np

@given(st.binary(min_size=0, max_size=1024))
def test_fuzz_deserialize(data):
    try:
        # best effort: call deserialize; must not crash interpreter
        deserialize_state_bytes(data, compressed=None)
    except Exception:
        pass

def test_serialize_fuzz_shapes():
    # generate random small arrays to serialize
    for _ in range(20):
        arr = np.random.randn(2,2).astype(np.float32)
        b = serialize_state_dict({"a": arr})
        assert b is not None
