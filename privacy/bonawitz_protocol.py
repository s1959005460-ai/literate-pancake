# -*- coding: utf-8 -*-
"""
Bonawitz-style secure aggregation protocol implementation.
Includes Shamir's secret sharing with proper error handling.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Dict, List, Tuple

from FedGNN_advanced.dp_rng import crypto_random_bytes
from FedGNN_advanced.privacy.errors import ProtocolAbortError

logger = logging.getLogger('FedGNN.privacy.bonawitz')
logger.setLevel(logging.INFO)

# Large prime for Shamir's secret sharing (2^61-1)
PRIME = 2305843009213693951


def _eval_poly(coeffs: List[int], x: int, prime: int) -> int:
    """
    Evaluate a polynomial at point x modulo prime.

    Args:
        coeffs: Polynomial coefficients (constant term first)
        x: Point to evaluate at
        prime: Prime modulus

    Returns:
        Polynomial value at x modulo prime
    """
    result = 0
    power = 1

    for coefficient in coeffs:
        result = (result + coefficient * power) % prime
        power = (power * x) % prime

    return result


def shamir_split(secret_int: int, n: int, t: int, prime: int = PRIME) -> Dict[int, int]:
    """
    Split a secret into n shares with threshold t using Shamir's secret sharing.

    Args:
        secret_int: Secret integer value
        n: Number of shares to create
        t: Threshold required for reconstruction
        prime: Prime modulus

    Returns:
        Dictionary mapping share indices to share values

    Raises:
        ValueError: If parameters are invalid
    """
    if not (0 <= secret_int < prime):
        raise ValueError("Secret must be in range [0, prime)")

    if not (1 <= t <= n):
        raise ValueError("Threshold must be between 1 and number of shares")

    # Generate random coefficients (degree t-1 polynomial)
    coefficients = [secret_int]
    for _ in range(t - 1):
        # Generate random coefficient using crypto RNG
        coeff_bytes = crypto_random_bytes(8)
        coefficient = int.from_bytes(coeff_bytes, 'big') % prime
        coefficients.append(coefficient)

    # Generate shares
    shares = {}
    for i in range(1, n + 1):
        shares[i] = _eval_poly(coefficients, i, prime)

    return shares


def _lagrange_interpolate(x: int, x_values: List[int], y_values: List[int], prime: int) -> int:
    """
    Perform Lagrange interpolation at point x.

    Args:
        x: Point to interpolate at
        x_values: X coordinates of known points
        y_values: Y coordinates of known points
        prime: Prime modulus

    Returns:
        Interpolated value at x
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")

    result = 0
    k = len(x_values)

    for i in range(k):
        numerator = 1
        denominator = 1

        for j in range(k):
            if i == j:
                continue

            numerator = (numerator * (x - x_values[j])) % prime
            denominator = (denominator * (x_values[i] - x_values[j])) % prime

        # Modular inverse of denominator
        denom_inv = pow(denominator, -1, prime)
        term = (y_values[i] * numerator * denom_inv) % prime
        result = (result + term) % prime

    return result


def shamir_reconstruct(shares: Dict[int, int], prime: int = PRIME) -> int:
    """
    Reconstruct a secret from shares using Lagrange interpolation.

    Args:
        shares: Dictionary mapping share indices to share values
        prime: Prime modulus

    Returns:
        Reconstructed secret integer

    Raises:
        ValueError: If insufficient shares are provided
    """
    if not shares:
        raise ValueError("No shares provided for reconstruction")

    x_values = list(shares.keys())
    y_values = [shares[x] for x in x_values]

    # Reconstruct the secret (value at x=0)
    return _lagrange_interpolate(0, x_values, y_values, prime)


def generate_round_secret(n: int, t: int, secret_bytes_len: int = 32) -> bytes:
    """
    Generate a cryptographically secure random secret for a round.

    Args:
        n: Number of shares to create
        t: Threshold required for reconstruction
        secret_bytes_len: Length of secret in bytes

    Returns:
        Random secret bytes

    Raises:
        ValueError: If parameters are invalid
    """
    if not (1 <= t <= n):
        raise ValueError("Threshold must be between 1 and number of shares")

    if secret_bytes_len <= 0:
        raise ValueError("Secret length must be positive")

    return crypto_random_bytes(secret_bytes_len)


def create_shares(secret_bytes: bytes, n: int, t: int) -> Dict[int, int]:
    """
    Create Shamir shares from secret bytes.

    Args:
        secret_bytes: Secret bytes to share
        n: Number of shares to create
        t: Threshold required for reconstruction

    Returns:
        Dictionary mapping share indices to share values

    Raises:
        ValueError: If secret is too large for the prime modulus
    """
    secret_int = int.from_bytes(secret_bytes, 'big')

    if secret_int >= PRIME:
        raise ValueError("Secret too large for chosen prime modulus")

    return shamir_split(secret_int, n, t, PRIME)


def reconstruct_secret_safe(shares: Dict[int, int]) -> bytes:
    """
    Reconstruct secret bytes from shares with proper error handling.

    Args:
        shares: Dictionary mapping share indices to share values

    Returns:
        Reconstructed secret bytes

    Raises:
        ProtocolAbortError: If reconstruction fails for any reason
    """
    try:
        secret_int = shamir_reconstruct(shares, PRIME)
    except Exception as e:
        logger.error("Shamir reconstruction failed: %s", e)
        raise ProtocolAbortError(f"Shamir reconstruction failed: {e}") from e

    try:
        # Convert back to bytes
        byte_length = (secret_int.bit_length() + 7) // 8
        return secret_int.to_bytes(byte_length or 1, 'big')
    except Exception as e:
        logger.error("Failed to convert reconstructed secret to bytes: %s", e)
        raise ProtocolAbortError("Reconstructed secret invalid") from e


def compute_update_mac(hmac_key: bytes, serialized_update: bytes, nonce: int) -> bytes:
    """
    Compute HMAC for a serialized update with nonce.

    Args:
        hmac_key: HMAC key bytes
        serialized_update: Serialized update bytes
        nonce: Nonce value

    Returns:
        HMAC digest bytes
    """
    nonce_bytes = nonce.to_bytes(8, 'little')
    message = serialized_update + nonce_bytes
    return hmac.new(hmac_key, message, hashlib.sha256).digest()


# Self-test and demonstration
if __name__ == '__main__':
    print("Running Bonawitz protocol self-test...")

    try:
        # Test basic secret sharing and reconstruction
        secret = generate_round_secret(5, 3, 16)
        shares = create_shares(secret, 5, 3)

        # Reconstruct with minimum shares
        subset = {k: shares[k] for k in list(shares.keys())[:3]}
        reconstructed = reconstruct_secret_safe(subset)

        assert secret == reconstructed
        print("✓ Basic secret sharing test passed")

        # Test HMAC computation
        key = b'test_key_32_bytes_long_enough_'
        message = b'test_message'
        nonce = 12345
        mac = compute_update_mac(key, message, nonce)

        assert len(mac) == 32  # SHA256 digest length
        print("✓ HMAC computation test passed")

        # Test error handling
        try:
            # Test with invalid shares
            invalid_shares = {1: 999999999999999999, 2: 888888888888888888}
            reconstruct_secret_safe(invalid_shares)
            assert False, "Should have raised ProtocolAbortError"
        except ProtocolAbortError:
            print("✓ Error handling test passed")

        print("All Bonawitz protocol tests passed successfully!")

    except Exception as e:
        print(f"Bonawitz protocol test failed: {e}")
        raise
