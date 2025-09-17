# File: crypto/he_tenseal.py
"""
TenSEAL wrapper for CKKS operations.

Features:
 - Generate context & keys (for testing/development)
 - Serialize/deserialize context & ciphertext
 - Encrypt vector, add ciphertexts, decrypt (requires secret key)

Production note:
 - For production, the secret key must be protected (KMS/HSM). Do NOT persist raw secret keys.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except Exception:
    TENSEAL_AVAILABLE = False


class HETensealError(RuntimeError):
    pass


class TenSEALHE:
    def __init__(self, poly_mod_degree: int = 8192, coeff_mod_bit_sizes: Optional[List[int]] = None,
                 scale: float = 2 ** 40):
        """
        Create a TenSEAL CKKS context.

        - poly_mod_degree: e.g., 8192, 16384
        - coeff_mod_bit_sizes: list of bit sizes; if None, sensible defaults used.
        - scale: scaling factor
        """
        if not TENSEAL_AVAILABLE:
            raise HETensealError("TenSEAL not installed. Install tenseal or set HE_DISABLED=true for simulation.")
        if coeff_mod_bit_sizes is None:
            # default chain for depth ~2
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        self.context.global_scale = scale
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.public_key = self.context.serialize_public_key()
        self.secret_key = self.context.serialize_secret_key()
        logger.info("TenSEAL CKKS context generated")

    @staticmethod
    def load_public_context(b: bytes) -> Any:
        if not TENSEAL_AVAILABLE:
            raise HETensealError("TenSEAL not available")
        ctx = ts.context_from(b)  # expects serialized context with public key
        return ctx

    def encrypt_vector(self, vec: List[float]) -> bytes:
        """
        Encrypts a numpy-like vector and returns serialized ciphertext bytes.
        """
        if not TENSEAL_AVAILABLE:
            raise HETensealError("TenSEAL not available")
        plain = np.array(vec, dtype=float).tolist()
        ck = ts.ckks_vector(self.context, plain)
        return ck.serialize()

    @staticmethod
    def add_ciphertexts(ct1_bytes: bytes, ct2_bytes: bytes) -> bytes:
        if not TENSEAL_AVAILABLE:
            raise HETensealError("TenSEAL not available")
        ct1 = ts.CKKSVector.load_from(ct1_bytes)
        ct2 = ts.CKKSVector.load_from(ct2_bytes)
        ct1 += ct2
        return ct1.serialize()

    def decrypt(self, ct_bytes: bytes) -> List[float]:
        """
        Decrypt ciphertext bytes (requires secret key in this context).
        """
        if not TENSEAL_AVAILABLE:
            raise HETensealError("TenSEAL not available")
        # Restore context with secret key
        ctx = ts.context_from(self.context.serialize())
        ts.ckks_vector  # ensure import
        ck = ts.CKKSVector.load_from(ctx, ct_bytes)
        return ck.decrypt()
