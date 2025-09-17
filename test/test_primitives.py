# tests/test_primitives.py
import math
from FedGNN_advanced.fed_crypto.primitives import rand_scalar, DEFAULT_PRIME

def test_rand_scalar_range():
    s = rand_scalar(128)
    assert 0 < s < DEFAULT_PRIME

def test_rand_scalar_different_calls():
    a = rand_scalar(128)
    b = rand_scalar(128)
    assert a != b  # extremely unlikely to fail

def test_rand_scalar_deterministic_rng():
    # deterministic rng for testing
    def rng(n): return b"\x01" * n
    s = rand_scalar(16, rng=rng)
    assert 0 < s < DEFAULT_PRIME
