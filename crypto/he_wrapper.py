# File: FedGNN_advanced/crypto/he_wrapper.py
"""
HE wrapper + KMS factory.

Revisions:
 - Enforce explicit FEDGNN_KMS_PROVIDER (no insecure defaults).
 - Validate required env vars for each provider (fail-fast).
 - Provide clear warnings for in-memory test provider only.

Audit references:
 - High risk: defaulting to InMemory KMS (audit high #1). This file enforces safe defaults and fails fast.
 - Uses KMS adapters (kms_aws.py, kms_vault.py) that must implement minimal interface:
     - get_private_key_bytes(key_id) -> bytes or raise
     - encrypt/decrypt wrappers as needed.

Dependencies: standard library only.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Minimal safe logger initialization
    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("FEDGNN_LOG_LEVEL", "INFO"))

# Allowed providers: require explicit config in env
ALLOWED_KMS_PROVIDERS = {"aws", "vault", "inmemory-test-only"}

KMS_PROVIDER = os.getenv("FEDGNN_KMS_PROVIDER")
if KMS_PROVIDER is None:
    # Fail fast: do not allow unspecified provider
    raise RuntimeError(
        "FEDGNN_KMS_PROVIDER is NOT set. Must be one of: 'aws', 'vault'. "
        "Do NOT use in-memory provider in production. See DEPLOYMENT.md for guidance."
    )
KMS_PROVIDER = KMS_PROVIDER.lower()
if KMS_PROVIDER not in ALLOWED_KMS_PROVIDERS:
    raise RuntimeError(f"Unsupported FEDGNN_KMS_PROVIDER='{KMS_PROVIDER}'. Allowed: {ALLOWED_KMS_PROVIDERS}")


class KMSClientProtocol(Protocol):
    """Protocol interface for KMS clients used by he_wrapper."""
    def get_private_key_bytes(self, key_id: str) -> bytes: ...
    def encrypt(self, key_id: str, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes: ...
    def decrypt(self, key_id: str, ciphertext: bytes, associated_data: Optional[bytes] = None) -> bytes: ...


def kms_client_factory() -> KMSClientProtocol:
    """
    Factory to return a KMS client instance. Performs fail-fast validation of required env vars.

    Returns:
        Instance implementing KMSClientProtocol.

    Raises:
        RuntimeError if required env is missing or adapter import fails.
    """
    global KMS_PROVIDER
    if KMS_PROVIDER == "aws":
        # Required env: AWS_KMS_KEY_ID not strictly required here, but helpful to validate
        if not os.getenv("AWS_REGION"):
            raise RuntimeError("AWS_REGION must be set when FEDGNN_KMS_PROVIDER=aws")
        try:
            from .kms_aws import AWSKMSClient  # type: ignore
        except Exception as e:
            logger.exception("Failed importing AWS KMS adapter")
            raise RuntimeError("AWS KMS adapter import failed") from e
        client = AWSKMSClient()
        logger.info("Using AWS KMS client")
        return client
    if KMS_PROVIDER == "vault":
        if not os.getenv("VAULT_ADDR") or not os.getenv("VAULT_TOKEN"):
            raise RuntimeError("VAULT_ADDR and VAULT_TOKEN must be set when FEDGNN_KMS_PROVIDER=vault")
        try:
            from .kms_vault import VaultKMSClient  # type: ignore
        except Exception as e:
            logger.exception("Failed importing Vault KMS adapter")
            raise RuntimeError("Vault KMS adapter import failed") from e
        client = VaultKMSClient()
        logger.info("Using Vault KMS client")
        return client
    # explicit in-memory test-only provider
    if KMS_PROVIDER == "inmemory-test-only":
        logger.warning("IN-MEMORY KMS selected. THIS IS FOR TESTING ONLY AND MUST NOT BE USED IN PRODUCTION.")
        try:
            from .kms_inmemory import InMemoryKMSClient  # type: ignore
        except Exception as e:
            logger.exception("Failed importing InMemory KMS adapter")
            raise RuntimeError("InMemory KMS adapter import failed") from e
        client = InMemoryKMSClient()
        return client

    # Should not reach here due to earlier validation
    raise RuntimeError(f"Unhandled KMS provider: {KMS_PROVIDER}")


# HE wrapper placeholder functions (exposed to rest of repo)
def encrypt_with_he_backend(plaintext: bytes, key_id: str, kms_client: Optional[KMSClientProtocol] = None) -> bytes:
    """
    High-level helper to encrypt a payload with an HE-compatible wrapper.
    In production, this may leverage TenSEAL or similar; we keep a clean separation
    between HE algorithm and key retrieval (from KMS).

    Note: This file focuses on KMS provider enforcement & safe factory; actual HE integration
    should be implemented in a dedicated module that imports TenSEAL, etc.
    """
    if kms_client is None:
        kms_client = kms_client_factory()
    # This wrapper returns plaintext as-is for now if no HE backend available.
    # Real HE implementation should call TenSEAL APIs and use KMS to fetch keys.
    logger.debug("encrypt_with_he_backend: no HE backend implemented in this wrapper")
    return plaintext


def decrypt_with_he_backend(ciphertext: bytes, key_id: str, kms_client: Optional[KMSClientProtocol] = None) -> bytes:
    """
    Symmetric helper to decrypt HE ciphertext. For real HE, use TenSEAL's decrypt calls and
    ensure private key material is properly handled via KMS.
    """
    if kms_client is None:
        kms_client = kms_client_factory()
    logger.debug("decrypt_with_he_backend: no HE backend implemented in this wrapper")
    return ciphertext
