# File: FedGNN_advanced/crypto/kms_aws.py
"""
AWS KMS adapter implementing minimal client interface:
 - get_private_key_bytes(key_id)
 - encrypt(key_id, plaintext, associated_data)
 - decrypt(key_id, ciphertext, associated_data)

Improvements:
 - Use envelope encryption pattern: generate data key with KMS, use AES-GCM for content encryption.
 - Use AAD (associated data) to bind metadata (e.g., key_id) to ciphertext.
 - Avoid logging secret material (redaction).

Dependencies: boto3, cryptography
"""
from __future__ import annotations

import base64
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

try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except Exception:
    boto3 = None
    ClientError = Exception


from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AWSKMSClient:
    """
    Minimal AWS KMS wrapper.

    Expected env:
      - AWS_REGION or default AWS config
    """
    def __init__(self):
        if boto3 is None:
            raise RuntimeError("boto3 is required for AWSKMSClient")
        self.kms = boto3.client("kms", region_name=os.getenv("AWS_REGION"))
        self.s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    def get_private_key_bytes(self, key_id: str) -> bytes:
        """
        Example: For private keys stored wrapped, you may call KMS to decrypt wrapped key material.
        For simplicity this example assumes there is a DynamoDB or S3 object encrypted with KMS
        which we can get and decrypt. Adapt to your storage model.
        """
        try:
            # Example path: S3 object key "keys/{key_id}.b64"
            bucket = os.getenv("FEDGNN_KEYS_BUCKET")
            if not bucket:
                raise RuntimeError("FEDGNN_KEYS_BUCKET must be set")
            obj = self.s3.get_object(Bucket=bucket, Key=f"keys/{key_id}.b64")
            b64 = obj["Body"].read()
            wrapped = base64.b64decode(b64)
            # Decrypt via KMS
            resp = self.kms.decrypt(CiphertextBlob=wrapped)
            plaintext = resp["Plaintext"]
            return plaintext
        except ClientError:
            logger.exception("AWSKMSClient.get_private_key_bytes failed (redacted)")
            raise RuntimeError("AWS KMS decrypt failed")
        except Exception:
            logger.exception("AWSKMSClient.get_private_key_bytes unexpected error (redacted)")
            raise RuntimeError("AWS KMS adapter failure")

    def encrypt(self, key_id: str, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Envelope encryption:
         - Generate a data key via KMS (GenerateDataKey)
         - Encrypt plaintext with AES-GCM using data key
         - Return concatenation: b64(kms_encrypted_key) || nonce || ciphertext

        associated_data should be bytes (example: b"model:v1|key_id")
        """
        try:
            resp = self.kms.generate_data_key(KeyId=key_id, KeySpec="AES_256")
            plaintext_key = resp["Plaintext"]
            encrypted_key = resp["CiphertextBlob"]
            aesgcm = AESGCM(plaintext_key)
            nonce = os.urandom(12)
            aad = associated_data or b""
            ct = aesgcm.encrypt(nonce, plaintext, aad)
            # Pack: base64(encrypted_key) || b"::" || nonce || ct
            packed = base64.b64encode(encrypted_key) + b"::" + nonce + ct
            return packed
        except ClientError:
            logger.exception("AWSKMSClient.encrypt failed (redacted)")
            raise RuntimeError("AWS KMS generate_data_key or encrypt failed")
        except Exception:
            logger.exception("AWSKMSClient.encrypt unexpected error (redacted)")
            raise RuntimeError("AWS KMS adapter failure")

    def decrypt(self, key_id: str, packed: bytes, associated_data: Optional[bytes] = None) -> bytes:
        try:
            import base64
            enc_key_b64, rest = packed.split(b"::", 1)
            encrypted_key = base64.b64decode(enc_key_b64)
            resp = self.kms.decrypt(CiphertextBlob=encrypted_key)
            plaintext_key = resp["Plaintext"]
            nonce = rest[:12]
            ct = rest[12:]
            aad = associated_data or b""
            aesgcm = AESGCM(plaintext_key)
            return aesgcm.decrypt(nonce, ct, aad)
        except ClientError:
            logger.exception("AWSKMSClient.decrypt failed (redacted)")
            raise RuntimeError("AWS KMS decrypt failed")
        except Exception:
            logger.exception("AWSKMSClient.decrypt unexpected error (redacted)")
            raise RuntimeError("AWS KMS adapter failure")
