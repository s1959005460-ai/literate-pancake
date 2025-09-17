# -*- coding: utf-8 -*-
"""
Tests for dp_rng module.
"""

import pytest
from FedGNN_advanced import dp_rng


def test_crypto_random_bytes():
    """Test that crypto_random_bytes returns distinct bytes."""
    b1 = dp_rng.crypto_random_bytes(16)
    b2 = dp_rng.crypto_random_bytes(16)

    assert isinstance(b1, bytes)
    assert isinstance(b2, bytes)
    assert len(b1) == 16
    assert len(b2) == 16
    assert b1 != b2


def test_crypto_random_int():
    """Test that crypto_random_int returns values in correct range."""
    # Test edge cases
    assert dp_rng.crypto_random_int(0) == 0

    # Test various ranges
    for max_val in [1, 10, 100]:
        val = dp_rng.crypto_random_int(max_val)
        assert 0 <= val <= max_val


def test_reproducible_rng():
    """Test that reproducible RNG produces deterministic results."""
    # First run
    dp_rng.set_reproducible_seed(42)
    rng1 = dp_rng.get_reproducible_numpy()
    values1 = rng1.integers(0, 100, size=5)

    # Second run with same seed
    dp_rng.set_reproducible_seed(42)
    rng2 = dp_rng.get_reproducible_numpy()
    values2 = rng2.integers(0, 100, size=5)

    # Should be identical
    assert np.array_equal(values1, values2)


def test_reproducible_rng_not_initialized():
    """Test error when reproducible RNG is accessed before initialization."""
    # Reset state
    dp_rng.set_reproducible_seed(42)  # Initialize first

    # Now test error case
    with pytest.raises(RuntimeError, match="not initialized"):
        # This should fail because we haven't called set_reproducible_seed
        # Reset by creating a new instance (simulate uninitialized state)
        import importlib
        importlib.reload(dp_rng)
        dp_rng.get_reproducible_numpy()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
