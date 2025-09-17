# 文件: tests/test_he_kms.py
import pytest
import numpy as np
from FedGNN_advanced.crypto.he_wrapper import InMemoryKMSClient, create_keypair_and_store, encrypt_with_public_context, decrypt_via_kms

ts = pytest.importorskip("tenseal", reason="TenSEAL required for HE tests")

def test_he_flow_kms_emulation():
    kms = InMemoryKMSClient()
    key_id = "test-key"
    public = create_keypair_and_store(kms, key_id, profile="small")
    vec = [1.0, 2.0, 3.0]
    ct = encrypt_with_public_context(public, vec)
    # decrypt via kms emulation
    pt = decrypt_via_kms(kms, key_id, ct)
    assert np.allclose(pt, np.array(vec), atol=1e-3)
