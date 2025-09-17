# tests/test_he_smoke.py
import pytest
import numpy as np
ts = pytest.importorskip("tenseal", reason="TenSEAL not installed")

from FedGNN_advanced.crypto.he_wrapper import create_ckks_context, encrypt_vector, add_ciphertexts, decrypt_vector

def test_he_encrypt_aggregate_decrypt():
    ctx = create_ckks_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 60], global_scale=2**30)
    # create secret context by reloading serialized private version: TenSEAL specifics allow key present in ctx
    v1 = [1.0, 2.0, 3.0]
    v2 = [0.5, -1.0, 4.0]
    c1 = encrypt_vector(ctx, v1)
    c2 = encrypt_vector(ctx, v2)
    csum = add_ciphertexts([c1, c2])
    dec = decrypt_vector(ctx, csum)
    expected = np.array(v1) + np.array(v2)
    np.testing.assert_allclose(dec, expected, atol=1e-3, rtol=1e-3)
