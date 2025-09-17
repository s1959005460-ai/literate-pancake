# tests/test_bonawitz_server.py
import base64
import pytest
from FedGNN_advanced.privacy import bonawitz_server as bs
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

def make_client_payload(server_priv_bytes):
    client_priv = X25519PrivateKey.generate()
    client_pub = client_priv.public_key().public_bytes(encoding=__import__('cryptography').hazmat.primitives.serialization.Encoding.Raw, format=__import__('cryptography').hazmat.primitives.serialization.PublicFormat.Raw)
    # derive shared key
    from FedGNN_advanced.crypto.crypto_utils import derive_shared_key, hmac_sign
    shared = derive_shared_key(server_priv_bytes, client_pub, info=b"bonawitz-mac")
    masked = b"masked-bytes"
    seq = 1
    msg = seq.to_bytes(8,'big') + masked
    mac = hmac_sign(shared, msg)
    payload = {
        "seq": seq,
        "masked_share": base64.b64encode(masked).decode(),
        "sender_pub": base64.b64encode(client_pub).decode(),
        "mac": base64.b64encode(mac).decode()
    }
    return payload, client_priv, client_pub

@pytest.mark.skip(reason="Requires KMS stub or provide server_priv via monkeypatch")
def test_verify_store(monkeypatch):
    # This test is a template: in CI, provide a KMS stub that returns private key bytes for key_id.
    server_priv = X25519PrivateKey.generate()
    server_priv_bytes = server_priv.private_bytes(encoding=__import__('cryptography').hazmat.primitives.serialization.Encoding.Raw,
                                                 format=__import__('cryptography').hazmat.primitives.serialization.PrivateFormat.Raw,
                                                 encryption_algorithm=__import__('cryptography').hazmat.primitives.serialization.NoEncryption())
    # monkeypatch KMS factory to return a stub with get_private_key_bytes
    class StubKMS:
        def get_private_key_bytes(self, key_id): return server_priv_bytes
    from FedGNN_advanced.crypto import he_wrapper
    monkeypatch.setattr(he_wrapper, "kms_client_factory", lambda: StubKMS())
    payload, _, _ = make_client_payload(server_priv_bytes)
    bs.verify_and_store_masked_share("clientA", 1, payload)
    assert "clientA" in bs.get_masked_shares_for_round(1)
