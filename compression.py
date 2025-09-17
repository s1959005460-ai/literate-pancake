# -*- coding: utf-8 -*-
"""
Compression utilities with zstandard support and fallback to zlib.
Provides efficient serialization and deserialization of model parameters.
"""

from __future__ import annotations

import importlib
import zlib
import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger('FedGNN.compression')
logger.setLevel(logging.INFO)

# Check if zstandard is available
_zstd_available = importlib.util.find_spec('zstandard') is not None

if _zstd_available:
    import zstandard as zstd


def serialize_sparse(
    arr: Any,
    method: str = 'zstd' if _zstd_available else 'zlib',
    zstd_level: int = 3
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize and compress a numpy array.

    Args:
        arr: Array-like data to compress
        method: Compression method ('zstd' or 'zlib')
        zstd_level: zstd compression level (1-22)

    Returns:
        Tuple of (compressed_bytes, metadata_dict)

    Raises:
        ValueError: If method is not supported
    """
    # Convert to numpy array with consistent dtype
    a = np.asarray(arr, dtype=np.float32)
    raw_bytes = a.tobytes()
    original_size = len(raw_bytes)

    # Apply compression
    if method == 'zstd' and _zstd_available:
        compressor = zstd.ZstdCompressor(level=zstd_level)
        payload = compressor.compress(raw_bytes)
        used_method = 'zstd'
    elif method == 'zlib' or (method == 'zstd' and not _zstd_available):
        payload = zlib.compress(raw_bytes)
        used_method = 'zlib'
    else:
        raise ValueError(f"Unsupported compression method: {method}")

    # Create metadata
    meta = {
        'method': used_method,
        'dtype': 'float32',
        'shape': a.shape,
        'original_size': original_size,
        'compressed_size': len(payload)
    }

    logger.debug(
        "Compression: %s, original: %d bytes, compressed: %d bytes (ratio: %.2f)",
        used_method, original_size, len(payload), len(payload) / max(1, original_size)
    )

    return payload, meta


def deserialize_sparse(payload: bytes, meta: Dict[str, Any]) -> np.ndarray:
    """
    Decompress and reconstruct a numpy array.

    Args:
        payload: Compressed bytes
        meta: Metadata dictionary from serialize_sparse

    Returns:
        Reconstructed numpy array

    Raises:
        ValueError: If metadata is invalid or method not supported
    """
    method = meta.get('method', 'zlib')

    # Decompress based on method
    if method == 'zstd' and _zstd_available:
        decompressor = zstd.ZstdDecompressor()
        raw_bytes = decompressor.decompress(payload)
    elif method == 'zlib':
        raw_bytes = zlib.decompress(payload)
    else:
        raise ValueError(f"Unsupported compression method in metadata: {method}")

    # Reconstruct array
    dtype = meta.get('dtype', 'float32')
    shape = meta.get('shape')

    if shape is None:
        raise ValueError("Shape metadata is required")

    arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    return arr


def get_compression_ratio(meta: Dict[str, Any]) -> float:
    """
    Calculate compression ratio from metadata.

    Args:
        meta: Metadata dictionary from serialize_sparse

    Returns:
        Compression ratio (compressed_size / original_size)
    """
    original = meta.get('original_size', 0)
    compressed = meta.get('compressed_size', 0)

    if original <= 0:
        return 1.0

    return compressed / original


# Self-test and demonstration
if __name__ == '__main__':
    print("Running Compression self-test...")

    import numpy as np

    # Test data
    test_data = np.random.randn(100, 50).astype(np.float32)

    # Test zstd if available, otherwise zlib
    method = 'zstd' if _zstd_available else 'zlib'
    print(f"Using compression method: {method}")

    # Test compression and decompression
    compressed, meta = serialize_sparse(test_data, method=method)
    decompressed = deserialize_sparse(compressed, meta)

    # Verify round-trip integrity
    assert np.allclose(test_data, decompressed), "Round-trip compression failed"

    # Verify compression ratio
    ratio = get_compression_ratio(meta)
    print(f"Compression ratio: {ratio:.3f}")

    # Test with different data types and shapes
    test_cases = [
        np.random.randn(10, 10).astype(np.float32),
        np.random.randn(100).astype(np.float32),
        np.random.randn(5, 5, 5).astype(np.float32)
    ]

    for i, data in enumerate(test_cases):
        compressed, meta = serialize_sparse(data, method=method)
        decompressed = deserialize_sparse(compressed, meta)
        assert np.allclose(data, decompressed), f"Test case {i + 1} failed"

    print("All Compression tests passed successfully!")
