# 文件: tests/test_shamir.py
import pytest
from FedGNN_advanced.privacy.shamir import split_secret_integer, reconstruct_from_map

def test_shamir_roundtrip():
    secret = 12345678901234567890
    shares = split_secret_integer(secret, n=5, t=3)
    subset = {k: shares[k] for k in list(shares.keys())[:3]}
    rec = reconstruct_from_map(subset)
    assert rec == secret % (1 << 256)  # modulus is large prime; check mod equivalence
