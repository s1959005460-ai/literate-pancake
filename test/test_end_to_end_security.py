# 文件: tests/test_end_to_end_security.py
import pytest
import secrets
import numpy as np
from FedGNN_advanced.utils.serialization import serialize_state_dict, deserialize_state_bytes
from FedGNN_advanced.privacy.bonawitz import create_client_share_package, validate_share_package
from FedGNN_advanced.privacy.shamir import split_secret_integer, reconstruct_from_map

def test_serialization_and_sharing_flow():
    state = {"w": np.array([[1.0,2.0]], dtype=np.float32)}
    data = serialize_state_dict(state, compress=False)
    out = deserialize_state_bytes(data, compressed=False)
    assert "w" in out

    # Shamir sharing roundtrip
    secret = 42
    shares = split_secret_integer(secret, n=5, t=3)
    subset = {k: shares[k] for k in list(shares.keys())[:3]}
    rec = reconstruct_from_map(subset)
    assert rec == secret

    # bonawitz MAC roundtrip
    h = secrets.token_bytes(32)
    pkg = create_client_share_package(h, {1: shares[1], 2:shares[2]})
    assert validate_share_package(h, pkg)
