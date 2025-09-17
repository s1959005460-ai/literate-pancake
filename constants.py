"""
FedGNN_advanced/constants.py

Central constants and configuration used across modules.
Designed for Python 3.9 compatibility with production-ready values.
"""

from typing import Any, Dict
import os
import torch

# Environment variable based configuration with sensible defaults
def get_env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ('true', '1', 'yes', 'y')

def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

# Production-ready defaults with environment variable overrides
DEFAULTS: Dict[str, Any] = {
    # Randomness
    "DEFAULT_SEED": get_env_int("DEFAULT_SEED", 42),
    "SEED_BYTE_LEN": get_env_int("SEED_BYTE_LEN", 32),

    # Finite field arithmetic (fits into signed int64 for many operations)
    "FINITE_FIELD_PRIME": get_env_int("FINITE_FIELD_PRIME", (1 << 61) - 1),  # 2**61 - 1, a Mersenne prime
    "FLOAT_TO_INT_SCALE": get_env_int("FLOAT_TO_INT_SCALE", 10**6),

    # Feldman VSS parameters (big-int modulus, generator)
    "VSS_MODULUS": get_env_int("VSS_MODULUS", 2**256 - 2**32 - 977),
    "VSS_GENERATOR": get_env_int("VSS_GENERATOR", 3),

    # Shamir threshold fraction for reconstruction
    "SHAMIR_THRESHOLD_FRACTION": get_env_float("SHAMIR_THRESHOLD_FRACTION", 0.5),

    # PRG / hashing
    "PRG_HASH": os.getenv("PRG_HASH", "sha256"),

    # Privacy defaults
    "DEFAULT_DELTA": get_env_float("DEFAULT_DELTA", 1e-5),

    # Aggregation / timeouts
    "DEFAULT_INBOX_DRAIN_TIMEOUT": get_env_float("DEFAULT_INBOX_DRAIN_TIMEOUT", 0.5),
    "DEFAULT_UNMASK_REQUEST_TIMEOUT": get_env_float("DEFAULT_UNMASK_REQUEST_TIMEOUT", 2.0),
    "ROUND_TIMEOUT": get_env_float("ROUND_TIMEOUT", 30.0),

    # Device
    "DEFAULT_DEVICE": os.getenv("DEFAULT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),

    # Demo dataset
    "DEFAULT_DATASET": os.getenv("DEFAULT_DATASET", "Cora"),

    # Logging
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),

    # Security
    "REQUIRE_CLIENT_AUTH": get_env_bool("REQUIRE_CLIENT_AUTH", True),
    "MAX_CLIENTS": get_env_int("MAX_CLIENTS", 100),

    # Performance
    "BATCH_SIZE": get_env_int("BATCH_SIZE", 32),
    "PREFETCH_FACTOR": get_env_int("PREFETCH_FACTOR", 2),

    # Model
    "HIDDEN_DIM": get_env_int("HIDDEN_DIM", 64),
    "LORA_RANK": get_env_int("LORA_RANK", 8),
}

# Convenience values with type annotations
SEED_BYTE_LEN: int = int(DEFAULTS["SEED_BYTE_LEN"])
FINITE_FIELD_PRIME: int = int(DEFAULTS["FINITE_FIELD_PRIME"])
FLOAT_TO_INT_SCALE: int = int(DEFAULTS["FLOAT_TO_INT_SCALE"])
VSS_MODULUS: int = int(DEFAULTS["VSS_MODULUS"])
VSS_GENERATOR: int = int(DEFAULTS["VSS_GENERATOR"])
SHAMIR_THRESHOLD_FRACTION: float = float(DEFAULTS["SHAMIR_THRESHOLD_FRACTION"])
DEFAULT_DELTA: float = float(DEFAULTS["DEFAULT_DELTA"])
DEFAULT_DEVICE: str = str(DEFAULTS["DEFAULT_DEVICE"])
ROUND_TIMEOUT: float = float(DEFAULTS["ROUND_TIMEOUT"])
LOG_LEVEL: str = str(DEFAULTS["LOG_LEVEL"])
REQUIRE_CLIENT_AUTH: bool = bool(DEFAULTS["REQUIRE_CLIENT_AUTH"])
MAX_CLIENTS: int = int(DEFAULTS["MAX_CLIENTS"])
BATCH_SIZE: int = int(DEFAULTS["BATCH_SIZE"])
PREFETCH_FACTOR: int = int(DEFAULTS["PREFETCH_FACTOR"])
HIDDEN_DIM: int = int(DEFAULTS["HIDDEN_DIM"])
LORA_RANK: int = int(DEFAULTS["LORA_RANK"])

# Export all constants for easy import
__all__ = [
    'SEED_BYTE_LEN', 'FINITE_FIELD_PRIME', 'FLOAT_TO_INT_SCALE',
    'VSS_MODULUS', 'VSS_GENERATOR', 'SHAMIR_THRESHOLD_FRACTION',
    'DEFAULT_DELTA', 'DEFAULT_DEVICE', 'ROUND_TIMEOUT', 'LOG_LEVEL',
    'REQUIRE_CLIENT_AUTH', 'MAX_CLIENTS', 'BATCH_SIZE', 'PREFETCH_FACTOR',
    'HIDDEN_DIM', 'LORA_RANK'
]
