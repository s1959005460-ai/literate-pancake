# 文件: FedGNN_advanced/privacy/bonawitz.py
"""
Bonawitz protocol helpers - final implementation pieces.

Covers:
- ECDH (X25519) seed derivation (pairwise) per Bonawitz 2017
- HKDF expansion to PRG (HMAC-DRBG style) to produce mask bytes
- Fixed-width MAC encoding for Shamir shares
- Client package creation and server-side verify helpers

Notes:
- This module provides cryptographic primitives and local helpers.
- The full interactive Bonawitz protocol (client/server orchestration, dropout handling)
  must use these primitives to implement the protocol flow per Bonawitz Section 3-5.
"""
from __future__ import annotations

import os
import json
import hashlib
import hmac
import logging
from typing import Tuple, Dict, Any, Optional

from prometheus_client import Counter
import numpy as np

# Use cryptography library for X25519 & HKDF
try:
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives import serialization
except Exception:
    raise ImportError("cryptography library with X25519 required")

logger = logging.getLogger("fedgnn.bonawitz")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":%(message)s}'))
    logger.addHandler(h)
logger.setLevel("INFO")

BONAWITZ_MAC_VALIDATIONS = Counter("fedgnn_bonawitz_mac_validations_total", "MAC validations")
BONAWITZ_MASK_GENERATIONS = Counter("fedgnn_bonawitz_mask_generations_total", "mask generations")

# Parameters
HKDF_INFO_LABEL = b"fedgnn-bonawitz-pairwise"
HKDF_LEN = 32  # we will expand to required bytes via iterated HMAC-DRBG

def generate_x25519_keypair() -> Tuple[bytes, bytes]:
    """
    Returns (private_bytes, public_bytes) for X25519 keypair.
    Private bytes are raw private key (32 bytes).
    Use secure storage for private bytes in production (KMS/HSM).
    """
    priv = x25519.X25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_bytes = pub.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return priv_bytes, pub_bytes

def derive_pairwise_seed(priv_a_bytes: bytes, pub_b_bytes: bytes) -> bytes:
    """
    Derive shared secret via X25519 and HKDF-SHA256 to produce seed for PRG.
    This follows Bonawitz recommendation: use DH shared secret then KDF.
    """
    priv = x25519.X25519PrivateKey.from_private_bytes(priv_a_bytes)
    pub_b = x25519.X25519PublicKey.from_public_bytes(pub_b_bytes)
    shared = priv.exchange(pub_b)
    # HKDF extract+expand
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=HKDF_LEN,
        salt=None,
        info=HKDF_INFO_LABEL,
    )
    seed = hkdf.derive(shared)
    return seed

def _hmac_expand(seed: bytes, info: bytes, length: int) -> bytes:
    """
    HMAC-DRBG-like expansion: iterate HMAC to produce 'length' bytes deterministically.
    """
    out = b""
    counter = 1
    while len(out) < length:
        ctr = counter.to_bytes(4, "little")
        out += hmac.new(seed, info + ctr, hashlib.sha256).digest()
        counter += 1
    res = out[:length]
    # pad to 4 bytes boundary if needed
    if len(res) % 4 != 0:
        res = res + b"\x00" * (4 - (len(res) % 4))
    return res

def derive_pairwise_mask(client_a_id: str, client_b_id: str, seed: bytes, length: int) -> bytes:
    """
    Expand seed into a deterministic mask of 'length' bytes using HKDF/HMAC-DRBG.
    client_a_id/client_b_id included in info to provide context separation.
    """
    if not isinstance(seed, (bytes, bytearray)):
        raise ValueError("seed must be bytes")
    info = client_a_id.encode("utf-8") + b"||" + client_b_id.encode("utf-8")
    out = _hmac_expand(seed, info, length)
    BONAWITZ_MASK_GENERATIONS.inc()
    return out

def generate_local_mask_vector(seed: bytes, shape: Tuple[int, ...], dtype="float32") -> np.ndarray:
    """
    Return masked numpy array derived deterministically from seed.
    Convert 4-byte chunks to uint32 and map to float range as small additive masks.
    """
    total = 1
    for d in shape:
        if d < 0:
            raise ValueError("negative dimension")
        total *= int(d)
    if total == 0:
        return np.zeros(shape, dtype=np.float32)
    byte_len = total * 4
    raw = derive_pairwise_mask("local", "local", seed, byte_len)
    if len(raw) % 4 != 0:
        raise RuntimeError("mask not 4-byte aligned")
    arr32 = np.frombuffer(raw, dtype=np.uint32, count=total).astype(np.uint32)
    scale = float(os.getenv("FEDGNN_MASK_SCALE", "1e-3"))
    arr_float = (arr32.astype(np.float64) / float(0xFFFFFFFF)) * (2.0 * scale) - scale
    return arr_float.astype(np.float32).reshape(shape)

def mac_share(hmac_key: bytes, share_index: int, share_value: int) -> bytes:
    """
    MAC a share using HMAC-SHA256 with fixed-width encoding.
    share_index: 4-byte little-endian
    share_value: 32-byte little-endian (canonical)
    """
    if not isinstance(hmac_key, (bytes, bytearray)):
        raise ValueError("hmac_key must be bytes")
    idx_bytes = int(share_index).to_bytes(4, "little", signed=False)
    val_bytes = int(share_value).to_bytes(32, "little", signed=False)
    payload = idx_bytes + val_bytes
    mac = hmac.new(hmac_key, payload, hashlib.sha256).digest()
    return mac

def verify_share_mac(hmac_key: bytes, share_index: int, share_value: int, mac: bytes) -> bool:
    expected = mac_share(hmac_key, share_index, share_value)
    ok = hmac.compare_digest(expected, mac)
    if ok:
        BONAWITZ_MAC_VALIDATIONS.inc()
    else:
        logger.warning(json.dumps({"msg":"mac verification failed","index":share_index}))
    return ok

def create_client_share_package(hmac_key: bytes, shares: Dict[int, int]) -> Dict[int, Tuple[int, bytes]]:
    out = {}
    for idx, val in shares.items():
        out[idx] = (val, mac_share(hmac_key, idx, val))
    return out

def validate_share_package(hmac_key: bytes, package: Dict[int, Tuple[int, bytes]]) -> bool:
    for idx, (val, mac) in package.items():
        if not verify_share_mac(hmac_key, idx, val, mac):
            return False
    return True
