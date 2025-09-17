# crypto/crypto_utils.py
"""
安全工具：X25519 派生、HMAC、AEAD（AES-GCM）等。
- 严格不在日志中输出密钥/明文。
- 对异常进行明确捕获并返回结构化错误。
"""
from __future__ import annotations
import logging
import secrets
from typing import Optional
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("crypto_utils")
logger.setLevel(logging.INFO)


def load_x25519_priv(b: bytes) -> X25519PrivateKey:
    try:
        return X25519PrivateKey.from_private_bytes(b)
    except Exception as e:
        logger.exception("invalid x25519 private bytes")
        raise


def load_x25519_pub(b: bytes) -> X25519PublicKey:
    try:
        return X25519PublicKey.from_public_bytes(b)
    except Exception:
        logger.exception("invalid x25519 public bytes")
        raise


def derive_shared_key(my_priv_bytes: bytes, peer_pub_bytes: bytes, info: bytes = b"fedgnn") -> bytes:
    """
    使用 X25519 + HKDF -> 32 字节 key.
    """
    try:
        priv = load_x25519_priv(my_priv_bytes)
        pub = load_x25519_pub(peer_pub_bytes)
        shared = priv.exchange(pub)
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info)
        return hkdf.derive(shared)
    except Exception:
        logger.exception("derive_shared_key failed")
        raise


def hmac_sign(key: bytes, msg: bytes) -> bytes:
    try:
        h = hmac.HMAC(key, hashes.SHA256())
        h.update(msg)
        return h.finalize()
    except Exception:
        logger.exception("hmac_sign failed")
        raise


def hmac_verify(key: bytes, msg: bytes, tag: bytes) -> bool:
    try:
        h = hmac.HMAC(key, hashes.SHA256())
        h.update(msg)
        h.verify(tag)
        return True
    except Exception:
        # 不输出异常中的秘密
        logger.warning("hmac verify failed")
        return False


def aead_encrypt(key: bytes, plaintext: bytes, aad: Optional[bytes] = None) -> bytes:
    """
    返回 nonce || ciphertext
    """
    try:
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)
        ct = aesgcm.encrypt(nonce, plaintext, aad or b"")
        return nonce + ct
    except Exception:
        logger.exception("aead_encrypt failed")
        raise


def aead_decrypt(key: bytes, blob: bytes, aad: Optional[bytes] = None) -> bytes:
    try:
        nonce = blob[:12]
        ct = blob[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, aad or b"")
    except Exception:
        logger.exception("aead_decrypt failed")
        raise
