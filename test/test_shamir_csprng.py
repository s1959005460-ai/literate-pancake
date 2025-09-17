# tests/test_shamir_csprng.py
import pytest
from FedGNN_advanced.privacy import shamir
from FedGNN_advanced.privacy.shamir import split_secret_integer, reconstruct_from_map, ShamirError

def test_split_and_reconstruct():
    secret = 12345678901234567890 % shamir.VSS_MODULUS
    shares = split_secret_integer(secret, n=5, t=3)
    subset = {k: shares[k] for k in list(shares.keys())[:3]}
    rec = reconstruct_from_map(subset)
    assert rec == secret

def test_csprng_used_stability():
    # ensure split_secret_integer produces differing coefficients across calls (statistical)
    s1 = split_secret_integer(1, n=5, t=3)
    s2 = split_secret_integer(1, n=5, t=3)
    assert any(s1[i] != s2[i] for i in s1)
