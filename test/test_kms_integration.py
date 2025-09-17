

# 文件: tests/test_kms_integration.py
import os
import pytest

pytestmark = pytest.mark.skipif(os.getenv("FEDGNN_KMS_PROVIDER") is None, reason="KMS provider not configured")

def test_kms_provider_roundtrip():
    from FedGNN_advanced.crypto.he_wrapper import kms_client_factory, create_context_and_store, encrypt_with_public_bytes, decrypt_via_kms
    kms = kms_client_factory()
    key_id = "test-key-" + os.getenv("CI_COMMIT_SHA", "local")
    public = create_context_and_store(kms, key_id, profile="small")
    v = [1.0, 2.0, 3.0]
    ct = encrypt_with_public_bytes(public, v)
    pt = decrypt_via_kms(kms, key_id, ct)
    # numeric compare
    import numpy as np
    assert hasattr(pt, "__iter__")
    assert np.allclose(pt, np.array(v), atol=1e-3)
