# FedGNN_advanced/privacy/mask_manager.py
from typing import Dict, Tuple
import hashlib
import hmac
import logging
import numpy as np

from FedGNN_advanced.privacy.ecdh import shared_secret_to_hkdf, validate_public_key
from FedGNN_advanced import constants

logger = logging.getLogger("fedgnn.mask_manager")
logger.setLevel(getattr(logging, constants.LOG_LEVEL.upper(), logging.INFO))

SEED_INFO = b"fedgnn-pairwise-seed"
SEED_LEN = constants.SEED_BYTE_LEN

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    from cryptography.hazmat.backends import default_backend

    def _chacha20_prg(key: bytes, nonce: bytes, nbytes: int) -> bytes:
        cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        return cipher.encryptor().update(b"\x00" * nbytes)
    _HAS_CHACHA = True
except Exception:
    _HAS_CHACHA = False

def derive_pairwise_seed(own_priv_bytes: bytes, peer_pub_bytes: bytes, info: bytes = SEED_INFO, length: int = SEED_LEN) -> bytes:
    if not validate_public_key(peer_pub_bytes):
        raise ValueError("peer public key invalid")
    seed = shared_secret_to_hkdf(own_priv_bytes, peer_pub_bytes, length=length, info=info)
    return seed

def expand_seed_to_mask_bytes(seed: bytes, nbytes: int) -> bytes:
    if _HAS_CHACHA:
        key = hashlib.sha256(seed + b"chacha-key").digest()
        nonce = hashlib.sha256(seed + b"chacha-nonce").digest()[:16]
        return _chacha20_prg(key, nonce, nbytes)
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out.extend(hmac.new(seed, ctr.to_bytes(8, "little"), hashlib.sha256).digest())
        ctr += 1
    return bytes(out[:nbytes])

def mask_dict_from_seed(seed: bytes, shapes: Dict[str, Tuple[int, ...]], finite_field: bool = True, prime: int = None, scale: int = None) -> Dict[str, np.ndarray]:
    prime = prime or constants.FINITE_FIELD_PRIME
    scale = scale or constants.FLOAT_TO_INT_SCALE
    masks = {}
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        nbytes = size * 8
        out = expand_seed_to_mask_bytes(seed + name.encode("utf-8"), nbytes)
        ints = np.frombuffer(out, dtype=np.int64).copy()
        if finite_field:
            ints = (ints % prime).astype(np.int64)
            masks[name] = ints.reshape(shape)
        else:
            floats = (ints.astype(np.float32) / (2 ** 31)) * 1e-3
            masks[name] = floats.reshape(shape)
    return masks
