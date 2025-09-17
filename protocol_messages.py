# FedGNN_advanced/protocol_messages.py
import hmac
import hashlib
import logging
from typing import Optional

from FedGNN_advanced.privacy.ecdh import sign_ed25519, verify_ed25519
from FedGNN_advanced import constants

logger = logging.getLogger("fedgnn.protocol_messages")
logger.setLevel(getattr(logging, constants.LOG_LEVEL.upper(), logging.INFO))

NONCE_BYTES = getattr(constants, "NONCE_BYTES", 8)

def build_hmac_message(payload: bytes, nonce: int) -> bytes:
    return payload + int(nonce).to_bytes(NONCE_BYTES, "little")

def compute_hmac(key: bytes, message: bytes) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()

def verify_hmac(key: bytes, message: bytes, mac: bytes) -> bool:
    try:
        expected = compute_hmac(key, message)
        return hmac.compare_digest(expected, mac)
    except Exception:
        logger.exception("HMAC verify error")
        return False

def sign_payload(priv_sig_bytes: bytes, payload: bytes) -> bytes:
    return sign_ed25519(priv_sig_bytes, payload)

def verify_signature(pub_sig_bytes: bytes, signature: bytes, payload: bytes) -> bool:
    try:
        return verify_ed25519(pub_sig_bytes, signature, payload)
    except Exception:
        logger.exception("Signature verify error")
        return False
