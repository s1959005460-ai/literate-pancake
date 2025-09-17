# tests/test_secagg_correctness.py
# This supplements the shamir tests by testing MAC-on-shares and mask generation
import pytest
import numpy as np
from FedGNN_advanced.privacy.bonawitz import (
    mac_share, verify_share_mac, create_client_share_package, validate_share_package,
    generate_local_mask_vector,
)
from FedGNN_advanced.dp_rng import crypto_random_bytes

def test_mac_on_shares_roundtrip():
    hkey = crypto_random_bytes(32)
    shares = {1: 123456, 2: 98765, 3: 555}
    pkg = create_client_share_package(hkey, shares)
    assert validate_share_package(hkey, pkg)

def test_mac_mismatch_detection():
    hkey = crypto_random_bytes(32)
    shares = {1: 1, 2: 2}
    pkg = create_client_share_package(hkey, shares)
    # tamper one value
    bad_pkg = dict(pkg)
    idx = next(iter(bad_pkg))
    bad_pkg[idx] = (bad_pkg[idx][0] + 1, bad_pkg[idx][1])
    assert not validate_share_package(hkey, bad_pkg)

def test_generate_local_mask_vector_shape_and_type():
    seed = crypto_random_bytes(32)
    arr = generate_local_mask_vector(seed, shape=(2, 3), dtype="float32")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float32
