# FedGNN_advanced/protocols.py
from enum import Enum, auto

class UpdateFormat(Enum):
    FLOAT32 = auto()
    FINITE_FIELD_INT64 = auto()

# Standard message types for HMAC / nonce window key names
NONCE_WINDOW_SIZE = 1024  # per-client recent nonces to track (simple replay protection)
