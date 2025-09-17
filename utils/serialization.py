# File: FedGNN_advanced/utils/serialization.py
"""
Safe serialization and deserialization utilities.

Features:
 - Limit uncompressed size and per-item size to prevent zip-bomb attacks.
 - Avoid logging raw bytes or large payloads (redact).
 - Provide robust error messages.

Audit references:
 - Strengthen decompression/deserialize boundaries (audit mid #2).
"""
from __future__ import annotations

import io
import json
import logging
import zlib
import struct
from typing import Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    import sys
    ch = logging.StreamHandler(sys.stdout := sys.stdout) if False else logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel("INFO")

# Tunable limits
MAX_PER_ITEM_BYTES = 50 * 1024 * 1024    # 50 MB per item
MAX_TOTAL_BYTES = 200 * 1024 * 1024      # 200 MB total
MAX_DECOMPRESS_BYTES = 500 * 1024 * 1024 # 500 MB after decompression (safety cap)


class SerializationError(Exception):
    pass


def safe_decompress(compressed: bytes) -> bytes:
    """
    Safe decompression using zlib with streaming bounds checks to avoid zip bombs.
    """
    if not isinstance(compressed, (bytes, bytearray)):
        raise SerializationError("compressed must be bytes")

    # Quick length check before expensive decompression
    if len(compressed) > MAX_PER_ITEM_BYTES:
        logger.warning("Compressed payload exceeds per-item limit (redacted size)")
        raise SerializationError("compressed payload too large")

    decompressor = zlib.decompressobj()
    total_out = 0
    out_chunks = []
    CHUNK = 16384
    start = 0
    # Stream decompress
    try:
        while start < len(compressed):
            chunk_in = compressed[start:start+CHUNK]
            start += CHUNK
            chunk_out = decompressor.decompress(chunk_in, MAX_DECOMPRESS_BYTES - total_out)
            out_chunks.append(chunk_out)
            total_out += len(chunk_out)
            if total_out > MAX_DECOMPRESS_BYTES:
                logger.warning("Decompressed payload exceeds MAX_DECOMPRESS_BYTES (redacted)")
                raise SerializationError("decompressed payload too large")
        # flush remainder
        tail = decompressor.flush()
        total_out += len(tail)
        if total_out > MAX_DECOMPRESS_BYTES:
            raise SerializationError("decompressed payload too large")
        out_chunks.append(tail)
    except zlib.error:
        logger.exception("zlib decompression failure (redacted)")
        raise SerializationError("decompression failed")

    out = b"".join(out_chunks)
    if len(out) > MAX_TOTAL_BYTES:
        logger.warning("Decompressed payload exceeds MAX_TOTAL_BYTES (redacted)")
        raise SerializationError("decompressed payload too large")
    return out


def deserialize_state_bytes(b: bytes) -> Any:
    """
    Deserialize bytes into python object. Bytes are expected to be zlib-compressed JSON by default.
    """
    try:
        raw = safe_decompress(b)
    except SerializationError:
        raise

    try:
        # for JSON
        s = raw.decode("utf-8")
        obj = json.loads(s)
        return obj
    except Exception:
        logger.exception("Failed to decode/parse deserialized object (redacted)")
        raise SerializationError("Invalid serialized payload")


def serialize_state(obj: Any) -> bytes:
    """
    Serialize python object to compressed bytes.
    """
    try:
        s = json.dumps(obj).encode("utf-8")
        compressed = zlib.compress(s)
        if len(compressed) > MAX_PER_ITEM_BYTES:
            logger.warning("Serialized object too large (redacted)")
            raise SerializationError("serialized object too large")
        return compressed
    except Exception:
        logger.exception("Serialization failed (redacted)")
        raise SerializationError("Serialization failed")
