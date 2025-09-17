# File: FedGNN_advanced/fed_crypto/primitives.py
"""
Cryptographic primitives used across FedGNN.

Modifications (audit):
 - Replaced insecure `random.getrandbits` with `secrets.token_bytes` for cryptographic randomness.
 - Provide optional injection of deterministic RNG for testing only (do not use in prod).

Theory / Rationale:
 - Python's `random` module is not cryptographically secure and must not be used for key/nonce generation.
 - Use os.urandom / secrets for cryptography-grade randomness.

Dependencies: standard library only (secrets, math)
"""
from __future__ import annotations

import secrets
import math
from typing import Callable, Optional

# Prime for finite field operations (placeholder; in production this should be a well-chosen large prime)
DEFAULT_PRIME = 2**521 - 1
DEFAULT_KEY_BITS = 256


def _default_crypto_rng(nbytes: int) -> bytes:
    """Return cryptographically secure random bytes."""
    return secrets.token_bytes(nbytes)


def rand_scalar(bits: int = DEFAULT_KEY_BITS, rng: Optional[Callable[[int], bytes]] = None) -> int:
    """
    Generate a random scalar in [1, DEFAULT_PRIME-1] using cryptographically secure RNG.

    Args:
        bits: requested bits of randomness
        rng: optional RNG(bytes) callable used for testing (must be deterministic if provided)

    Returns:
        int: scalar in field
    """
    if rng is None:
        rng = _default_crypto_rng
    byte_len = math.ceil(bits / 8)
    # loop until scalar != 0 to avoid zero scalar edge-case
    while True:
        b = rng(byte_len)
        scalar = int.from_bytes(b, "big") % DEFAULT_PRIME
        if scalar != 0:
            return scalar


# Example: field add / mul helpers
def field_add(a: int, b: int) -> int:
    return (a + b) % DEFAULT_PRIME


def field_mul(a: int, b: int) -> int:
    return (a * b) % DEFAULT_PRIME
