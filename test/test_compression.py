# tests/test_compression.py
from FedGNN_advanced import compression
import numpy as np

def test_roundtrip():
    x = np.random.randn(20,10).astype(np.float32)
    p, m = compression.serialize_sparse(x)
    y = compression.deserialize_sparse(p, m)
    assert (x == y).all()

if __name__ == "__main__":
    test_roundtrip()
    print("test_compression passed")
