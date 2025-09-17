# -*- coding: utf-8 -*-
"""
Secure RNG management for FedGNN_advanced.
Distinguishes between cryptographic RNG (for production) and reproducible RNG (for experiments).
"""

from __future__ import annotations

import secrets
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

__all__ = [
    'set_reproducible_seed',
    'get_reproducible_numpy',
    'get_reproducible_torch_generator',
    'crypto_random_bytes',
    'crypto_random_int',
    'get_rng_source_info'
]

# Global state for reproducible RNG
_repro_seed: Optional[int] = None
_numpy_rng: Optional[np.random.Generator] = None
_torch_gen: Optional[torch.Generator] = None


def set_reproducible_seed(seed: int) -> None:
    """
    Set a reproducible seed for experimental runs.

    Args:
        seed: Integer seed value

    Raises:
        ValueError: If seed is not an integer
    """
    global _repro_seed, _numpy_rng, _torch_gen

    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")

    _repro_seed = seed
    random.seed(_repro_seed)
    _numpy_rng = np.random.default_rng(_repro_seed)

    if torch is not None:
        _torch_gen = torch.Generator().manual_seed(_repro_seed)
    else:
        _torch_gen = None


def get_reproducible_numpy() -> np.random.Generator:
    """
    Get the reproducible numpy random number generator.

    Returns:
        numpy.random.Generator instance

    Raises:
        RuntimeError: If reproducible seed has not been set
    """
    if _numpy_rng is None:
        raise RuntimeError(
            "Reproducible numpy RNG not initialized. "
            "Call set_reproducible_seed(seed) first."
        )
    return _numpy_rng


def get_reproducible_torch_generator() -> torch.Generator:
    """
    Get the reproducible torch random number generator.

    Returns:
        torch.Generator instance

    Raises:
        RuntimeError: If torch is not available or seed not set
    """
    if torch is None:
        raise RuntimeError("Torch is not available in this environment")

    if _torch_gen is None:
        raise RuntimeError(
            "Reproducible torch generator not initialized. "
            "Call set_reproducible_seed(seed) first."
        )
    return _torch_gen


def crypto_random_bytes(nbytes: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        nbytes: Number of bytes to generate (must be positive)

    Returns:
        Random bytes

    Raises:
        ValueError: If nbytes is not a positive integer
    """
    if not isinstance(nbytes, int) or nbytes <= 0:
        raise ValueError("nbytes must be a positive integer")

    return secrets.token_bytes(nbytes)


def crypto_random_int(max_inclusive: int) -> int:
    """
    Generate a cryptographically secure random integer.

    Args:
        max_inclusive: Maximum inclusive value (must be non-negative)

    Returns:
        Random integer in range [0, max_inclusive]

    Raises:
        ValueError: If max_inclusive is not a non-negative integer
    """
    if not isinstance(max_inclusive, int) or max_inclusive < 0:
        raise ValueError("max_inclusive must be a non-negative integer")

    if max_inclusive == 0:
        return 0

    return secrets.randbelow(max_inclusive + 1)


def get_rng_source_info() -> dict:
    """
    Get information about the current RNG state.

    Returns:
        Dictionary with RNG state information
    """
    return {'reproducible_seed_set': _repro_seed is not None}


# Self-test and demonstration
if __name__ == '__main__':
    print("Running dp_rng self-test...")

    # Test cryptographic RNG
    b1 = crypto_random_bytes(16)
    b2 = crypto_random_bytes(16)
    assert len(b1) == 16 and len(b2) == 16
    assert b1 != b2
    print("✓ crypto_random_bytes test passed")

    # Test crypto random int
    for max_val in [0, 1, 10, 100]:
        val = crypto_random_int(max_val)
        assert 0 <= val <= max_val
    print("✓ crypto_random_int test passed")

    # Test reproducible RNG
    set_reproducible_seed(42)
    rng1 = get_reproducible_numpy()
    values1 = rng1.integers(0, 100, size=5)

    set_reproducible_seed(42)
    rng2 = get_reproducible_numpy()
    values2 = rng2.integers(0, 100, size=5)

    assert np.array_equal(values1, values2)
    print("✓ reproducible RNG test passed")

    # Test torch generator if available
    if torch is not None:
        try:
            gen = get_reproducible_torch_generator()
            print("✓ torch generator test passed")
        except RuntimeError:
            print("⚠ torch generator not available")
    else:
        print("⚠ torch not available, skipping torch tests")

    print("All dp_rng tests passed successfully!")
