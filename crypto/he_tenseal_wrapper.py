# crypto/he_tenseal_wrapper.py
"""
TenSEAL CKKS wrapper with production considerations.

IMPORTANT:
 - In production, do NOT store secret_key on disk. Use KMS/HSM or a secure decryptor service.
 - This module exposes a simple API for encryption/aggregation/decryption:
    - setup_dev_context() -> development context, only for testing
    - encrypt_vector(ctx, vec) -> bytes
    - add_ciphertexts(ct1, ct2) -> bytes
    - decrypt_aggregate(ctx, ct) -> list[float]
 - For KMS-based decryption, implement a small secure decryption microservice that holds secret keys and exposes a restricted API.
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger("he_tenseal")
logger.setLevel(logging.INFO)

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except Exception:
    TENSEAL_AVAILABLE = False


class HEError(RuntimeError):
    pass


class HEContext:
    """
    Wrapper around TenSEAL context. Contains serialized public and secret key (if in dev mode).
    """
    def __init__(self, ctx: "ts.Context", store_secret: bool = False):
        self._ctx = ctx
        self.public_context = ctx.serialize()  # includes public key material
        self._store_secret = store_secret
        if store_secret:
            self.secret_context = ctx.serialize()  # dev-only: includes secret
        else:
            self.secret_context = None

    @classmethod
    def setup_dev_context(cls, poly_mod_degree: int = 8192, scale: float = 2**40) -> "HEContext":
        if not TENSEAL_AVAILABLE:
            raise HEError("TenSEAL not installed")
        # Create CKKS context with relin & galois keys
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree)
        ctx.global_scale = scale
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        return cls(ctx, store_secret=True)

    @classmethod
    def load_public_context(cls, serialized: bytes) -> "HEContext":
        if not TENSEAL_AVAILABLE:
            raise HEError("TenSEAL not installed")
        ctx = ts.context_from(serialized)
        return cls(ctx, store_secret=False)

    def encrypt_vector(self, vec: List[float]) -> bytes:
        if not TENSEAL_AVAILABLE:
            raise HEError("TenSEAL not installed")
        v = ts.ckks_vector(self._ctx, list(vec))
        return v.serialize()

    @staticmethod
    def add_ciphertexts(ct1: bytes, ct2: bytes) -> bytes:
        if not TENSEAL_AVAILABLE:
            raise HEError("TenSEAL not installed")
        a = ts.CKKSVector.load_from(ct1)
        b = ts.CKKSVector.load_from(ct2)
        a += b
        return a.serialize()

    def decrypt(self, ct: bytes) -> List[float]:
        if not TENSEAL_AVAILABLE:
            raise HEError("TenSEAL not installed")
        if not self.secret_context:
            raise HEError("Secret context not loaded. In production, perform decryption in secure node.")
        ctx = ts.context_from(self.secret_context)
        vec = ts.CKKSVector.load_from(ctx, ct)
        return vec.decrypt()
