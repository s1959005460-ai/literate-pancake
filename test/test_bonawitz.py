# 文件: tests/test_bonawitz.py
import pytest
import secrets
from FedGNN_advanced.privacy.bonawitz import generate_x25519_keypair, derive_pairwise_seed, derive_pairwise_mask, generate_local_mask_vector, create_client_share_package, validate_share_package

def test_ecdh_seed_and_mask():
    a_priv, a_pub = generate_x25519_keypair()
    b_priv, b_pub = generate_x25519_keypair()
    seed_ab = derive_pairwise_seed(a_priv, b_pub)
    seed_ba = derive_pairwise_seed(b_priv, a_pub)
    assert seed_ab == seed_ba
    m = derive_pairwise_mask("a","b",seed_ab, 16)
    assert isinstance(m, bytes)
    arr = generate_local_mask_vector(seed_ab, (2,2))
    assert arr.shape == (2,2)

def test_mac_package():
    key = secrets.token_bytes(32)
    shares = {1: 123456789, 2: 987654321}
    pkg = create_client_share_package(key, shares)
    assert validate_share_package(key, pkg)
    # tamper
    bad = dict(pkg)
    first = next(iter(bad))
    bad[first] = (bad[first][0] + 1, bad[first][1])
    assert not validate_share_package(key, bad)
