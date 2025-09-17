# 文件: tests/test_he_kms_emulation.py
import os
import pytest
import numpy as np
from FedGNN_advanced.crypto import he_wrapper
import secrets

ts = pytest.importorskip("tenseal", reason="TenSEAL required for this test")

def test_public_private_flow_kms_emulation():
    # create a private context (simulate KMS enclave creation)
    ctx = he_wrapper.create_ckks_context(poly_modulus_degree=4096, coeff_mod_bit_sizes=[60, 30, 30], global_scale=2**20)
    public_bytes = he_wrapper.create_public_context_from_private(ctx)
    v = [1.0, 2.0, 3.0]
    ct = he_wrapper.encrypt_vector_public(public_bytes, v)
    dec = he_wrapper.decrypt_vector_local(ctx, ct)
    assert np.allclose(dec, np.array(v), atol=1e-3)

def test_private_serialize_policy():
    ctx = he_wrapper.create_ckks_context(poly_modulus_degree=4096, coeff_mod_bit_sizes=[60, 30, 30], global_scale=2**20)
    os.environ.pop("FEDGNN_ALLOW_PRIVATE_SERIALIZE", None)
    with pytest.raises(PermissionError):
        he_wrapper.serialize_context_private(ctx)
