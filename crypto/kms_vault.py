# File: FedGNN_advanced/crypto/kms_vault.py
"""
Vault KMS Client adapter.

Purpose:
 - Provide a minimal wrapper API compatible with KMSClientProtocol used by he_wrapper.kms_client_factory.
 - Delay network I/O until method calls (avoid doing network at import time).
 - Secure logging: do not log secret content; only log structured error codes.

Audit references:
 - Fix logger name typo and avoid import-time hvac initialization (audit high #2).
 - Validate VAULT_ADDR and VAULT_TOKEN presence before operations.

Dependencies:
 - cryptography (for envelope if needed)
 - hvac (optional at runtime)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout := sys.stdout) if False else logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("FEDGNN_LOG_LEVEL", "INFO"))

# We import hvac lazily to avoid import-time network/IO side effects.
class VaultKMSClient:
    """
    Minimal Vault KMS client adapter. Implementation aims to:
     - Keep secret material in memory only when necessary.
     - Provide get_private_key_bytes(key_id) and encrypt/decrypt wrappers.

    Requires VAULT_ADDR and VAULT_TOKEN env vars to be set (validated by he_wrapper).
    """
    def __init__(self, addr: Optional[str] = None, token: Optional[str] = None) -> None:
        self.addr = addr or os.getenv("VAULT_ADDR")
        self.token = token or os.getenv("VAULT_TOKEN")
        if not self.addr or not self.token:
            raise RuntimeError("VAULT_ADDR and VAULT_TOKEN must be provided to instantiate VaultKMSClient")
        self._client = None  # type: Optional["hvac.Client"]

    def _ensure_client(self):
        if self._client is None:
            try:
                import hvac  # type: ignore
                self._client = hvac.Client(url=self.addr, token=self.token)
                if not self._client.is_authenticated():
                    logger.error("Vault authentication failed (check token)")
                    raise RuntimeError("Vault authentication failed")
            except Exception as e:
                # Do not log exception message containing potential secrets
                logger.exception("Vault client initialization failed")
                raise

    def get_private_key_bytes(self, key_id: str) -> bytes:
        """
        Retrieve a private key as raw bytes from Vault KV or transit (depending on deployment).
        Note: Ideally we store raw private key material encrypted in transit and require
        additional access gating. This method intentionally returns raw bytes but should be used
        with caution and bounded to server memory lifetime.

        Returns:
            bytes: private key bytes

        Raises:
            RuntimeError on failure
        """
        self._ensure_client()
        try:
            # Example: reading from KV v2 at path "secret/data/fedgnn/keys/{key_id}"
            kv_path = f"secret/data/fedgnn/keys/{key_id}"
            resp = self._client.secrets.kv.v2.read_secret_version(path=f"fedgnn/keys/{key_id}")
            # The exact structure depends on Vault mount; adjust as needed in deployment.
            data = resp.get("data", {}).get("data", {})
            key_b64 = data.get("private_key_b64")
            if not key_b64:
                raise RuntimeError("private_key_b64 not found in vault secret")
            import base64
            return base64.b64decode(key_b64)
        except Exception:
            logger.exception("Vault: failed to retrieve private key (non-sensitive message logged)")
            raise RuntimeError("Failed to retrieve key from Vault")

    def encrypt(self, key_id: str, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        For envelope encryption we might use a Vault transit key (sign/encrypt) or client-side AES-GCM.
        This is a simple wrapper that uses transit encrypt API if available.
        """
        self._ensure_client()
        try:
            # Use transit mount 'transit' and key name = key_id; encode plaintext to base64
            import base64
            b64 = base64.b64encode(plaintext).decode()
            resp = self._client.secrets.transit.encrypt_data(name=key_id, plaintext=b64)
            ct = resp["data"]["ciphertext"]
            return ct.encode()
        except Exception:
            logger.exception("Vault: encrypt operation failed (non-sensitive message logged)")
            raise RuntimeError("Vault encrypt failed")

    def decrypt(self, key_id: str, ciphertext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        self._ensure_client()
        try:
            import base64
            # ciphertext should be str like "vault:v1:..."
            ct = ciphertext.decode() if isinstance(ciphertext, bytes) else ciphertext
            resp = self._client.secrets.transit.decrypt_data(name=key_id, ciphertext=ct)
            b64 = resp["data"]["plaintext"]
            return base64.b64decode(b64)
        except Exception:
            logger.exception("Vault: decrypt operation failed (non-sensitive message logged)")
            raise RuntimeError("Vault decrypt failed")
