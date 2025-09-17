# File: tests/test_crypto_utils.py
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization
from crypto.crypto_utils import derive_shared_key, hmac_sign, hmac_verify

def _priv_pub_bytes():
    a_priv = X25519PrivateKey.generate()
    b_priv = X25519PrivateKey.generate()
    a_priv_b = a_priv.private_bytes(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
    a_pub_b = a_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    b_priv_b = b_priv.private_bytes(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
    b_pub_b = b_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return a_priv_b, a_pub_b, b_priv_b, b_pub_b

def test_derive_shared_key_symmetry():
    a_priv_b, a_pub_b, b_priv_b, b_pub_b = _priv_pub_bytes()
    k1 = derive_shared_key(a_priv_b, b_pub_b)
    k2 = derive_shared_key(b_priv_b, a_pub_b)
    assert k1 == k2
    assert len(k1) == 32

def test_hmac_sign_verify():
    key = derive_shared_key(*_priv_pub_bytes()[:2])  # simple derive
    msg = b"hello"
    tag = hmac_sign(key, msg)
    assert hmac_verify(key, msg, tag)
    assert not hmac_verify(key, msg + b"x", tag)
