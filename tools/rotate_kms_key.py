#!/usr/bin/env python3
# File: tools/rotate_kms_key.py
"""
Key rotation / re-wrap tool for FedGNN project.

Supports:
 - AWS: S3 objects holding base64-wrapped key material, re-wrap via KMS (encrypt with new key)
 - Vault: KV entries storing base64 ciphertext and Vault transit decrypt/encrypt

Usage examples:
  AWS:
    FEDGNN_KEYS_BUCKET=fedgnn-keys python tools/rotate_kms_key.py \
        --provider aws --bucket fedgnn-keys --keys key1,key2 \
        --new-key-id alias/new-kms-key --dry-run

  Vault:
    VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=root \
    python tools/rotate_kms_key.py --provider vault --vault-path secret/data/fedgnn/keys \
        --keys key1,key2 --new-key-name new-transit-key --dry-run

Notes:
 - The script performs a safe backup of original ciphertext (saves .bak).
 - Permissions required:
   * AWS: s3:GetObject, s3:PutObject, kms:Decrypt (for old key), kms:Encrypt (for new key)
   * Vault: kv read/write, transit/decrypt, transit/encrypt for the respective keys
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger("rotate_kms_key")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# local imports for optional dependencies
try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except Exception:
    boto3 = None
    ClientError = Exception

try:
    import hvac  # type: ignore
except Exception:
    hvac = None


def aws_rotate(
    bucket: str,
    keys: List[str],
    new_kms_key_id: str,
    dry_run: bool = True,
    backup: bool = True,
    s3_prefix: str = "keys/",
):
    if boto3 is None:
        raise RuntimeError("boto3 required for AWS rotation")

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    kms = boto3.client("kms", region_name=os.getenv("AWS_REGION"))

    for key_id in keys:
        object_key = f"{s3_prefix}{key_id}.b64"
        logger.info("AWS: Processing key_id=%s object=%s", key_id, object_key)
        try:
            obj = s3.get_object(Bucket=bucket, Key=object_key)
            b64_wrapped = obj["Body"].read()
        except ClientError as e:
            logger.error("Failed to fetch S3 object for %s (skipping): %s", key_id, str(e))
            continue

        # decode the wrapped key
        try:
            wrapped = base64.b64decode(b64_wrapped)
        except Exception:
            logger.error("Object %s is not valid base64 (skipping)", object_key)
            continue

        # Decrypt using KMS (old wrap key)
        try:
            resp = kms.decrypt(CiphertextBlob=wrapped)
            plaintext_key = resp["Plaintext"]
            logger.info("Decrypted wrapped key for %s (plaintext length=%d bytes, redacted)", key_id, len(plaintext_key))
        except ClientError as e:
            logger.error("KMS decrypt failed for %s: %s", key_id, str(e))
            continue

        # Encrypt with new KMS key (re-wrap)
        try:
            resp2 = kms.encrypt(KeyId=new_kms_key_id, Plaintext=plaintext_key)
            new_wrapped = resp2["CiphertextBlob"]
            new_b64 = base64.b64encode(new_wrapped)
            logger.info("Re-wrapped key for %s with new_kms_key_id=%s (ciphertext length=%d, redacted)", key_id, new_kms_key_id, len(new_wrapped))
        except ClientError as e:
            logger.error("KMS encrypt failed for %s with new key %s: %s", key_id, new_kms_key_id, str(e))
            continue

        # Backup and write back
        if dry_run:
            logger.info("Dry-run enabled: would backup original object and write new wrapped key for %s", key_id)
            continue

        # backup original
        if backup:
            bak_key = f"{object_key}.bak"
            try:
                s3.put_object(Bucket=bucket, Key=bak_key, Body=b64_wrapped)
                logger.info("Backup written to %s", bak_key)
            except ClientError as e:
                logger.error("Failed to write backup for %s: %s", key_id, str(e))
                continue

        # write new wrapped
        try:
            s3.put_object(Bucket=bucket, Key=object_key, Body=new_b64)
            logger.info("Replaced wrapped key in S3 for %s", key_id)
        except ClientError as e:
            logger.error("Failed to write new wrapped key for %s: %s", key_id, str(e))
            # Attempt to restore backup if backup requested
            if backup:
                try:
                    orig_bak = s3.get_object(Bucket=bucket, Key=f"{object_key}.bak")["Body"].read()
                    s3.put_object(Bucket=bucket, Key=object_key, Body=orig_bak)
                    logger.info("Restored original object for %s from backup after failure", key_id)
                except Exception:
                    logger.critical("Failed to restore backup for %s; manual intervention required", key_id)
            continue


def vault_rotate(
    vault_addr: str,
    token: str,
    kv_prefix: str,
    keys: List[str],
    new_transit_key_name: str,
    dry_run: bool = True,
    backup: bool = True,
):
    if hvac is None:
        raise RuntimeError("hvac required for Vault rotation")
    client = hvac.Client(url=vault_addr, token=token)
    if not client.is_authenticated():
        raise RuntimeError("Vault authentication failed")

    for key_id in keys:
        kv_path = f"{kv_prefix}/{key_id}"
        logger.info("Vault: Processing key_id=%s at kv_path=%s", key_id, kv_path)
        try:
            read = client.secrets.kv.v2.read_secret_version(path=kv_path.replace("secret/data/", "").replace("secret/", ""))
            data = read.get("data", {}).get("data", {})
            b64_wrapped = data.get("wrapped_b64")
            if b64_wrapped is None:
                logger.error("No 'wrapped_b64' field in KV for %s (skipping)", key_id)
                continue
            wrapped = base64.b64decode(b64_wrapped)
        except Exception as e:
            logger.error("Failed to read KV for %s: %s", key_id, str(e))
            continue

        # decrypt via transit (assuming ciphertext is in transit format)
        try:
            # transit.decrypt_data expects ciphertext string
            ciphertext_str = wrapped.decode() if isinstance(wrapped, bytes) else wrapped
            dec = client.secrets.transit.decrypt_data(name="transit", ciphertext=ciphertext_str)
            plaintext_b64 = dec["data"]["plaintext"]
            plaintext = base64.b64decode(plaintext_b64)
            logger.info("Vault transit decrypt succeeded for %s (plaintext length=%d bytes, redacted)", key_id, len(plaintext))
        except Exception as e:
            logger.error("Vault transit decrypt failed for %s: %s", key_id, str(e))
            continue

        # re-encrypt with new transit key
        try:
            new_enc = client.secrets.transit.encrypt_data(name=new_transit_key_name, plaintext=base64.b64encode(plaintext).decode())
            new_ciphertext = new_enc["data"]["ciphertext"]
            new_wrapped_b64 = base64.b64encode(new_ciphertext.encode())
            logger.info("Re-encrypted %s with new transit key %s (redacted)", key_id, new_transit_key_name)
        except Exception as e:
            logger.error("Vault transit encrypt failed for %s: %s", key_id, str(e))
            continue

        if dry_run:
            logger.info("Dry-run: would update KV for %s with new wrapped value", key_id)
            continue

        # backup (write to kv at {kv_prefix}/{key_id}.bak)
        if backup:
            try:
                backup_path = f"{kv_prefix}/{key_id}.bak"
                client.secrets.kv.v2.create_or_update_secret(path=backup_path.replace("secret/data/", "").replace("secret/", ""), secret={"wrapped_b64": b64_wrapped.decode() if isinstance(b64_wrapped, (bytes, bytearray)) else b64_wrapped})
                logger.info("Vault backup written to %s", backup_path)
            except Exception as e:
                logger.error("Failed to write vault backup for %s: %s", key_id, str(e))
                continue

        # write new wrapped
        try:
            client.secrets.kv.v2.create_or_update_secret(path=kv_path.replace("secret/data/", "").replace("secret/", ""), secret={"wrapped_b64": new_wrapped_b64.decode()})
            logger.info("Replaced wrapped key in Vault for %s", key_id)
        except Exception as e:
            logger.error("Failed to write new wrapped value for %s: %s", key_id, str(e))
            # attempt restore from backup
            if backup:
                try:
                    bak_read = client.secrets.kv.v2.read_secret_version(path=f"{kv_prefix}/{key_id}.bak".replace("secret/data/", "").replace("secret/", ""))
                    bak_data = bak_read.get("data", {}).get("data", {})
                    orig = bak_data.get("wrapped_b64")
                    client.secrets.kv.v2.create_or_update_secret(path=kv_path.replace("secret/data/", "").replace("secret/", ""), secret={"wrapped_b64": orig})
                    logger.info("Restored original KV for %s from backup", key_id)
                except Exception:
                    logger.critical("Failed to restore backup for %s; manual intervention required", key_id)
            continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotate / re-wrap wrapped key materials stored in S3 or Vault KV.")
    p.add_argument("--provider", choices=["aws", "vault"], required=True, help="KMS provider / storage backend")
    p.add_argument("--keys", required=True, help="Comma-separated list of key identifiers (e.g., key1,key2)")
    p.add_argument("--dry-run", action="store_true", help="Dry run (do not write changes)")
    p.add_argument("--backup", action="store_true", default=False, help="Create backups before replacing (default: False). Use with caution.")
    # AWS specific
    p.add_argument("--bucket", help="S3 bucket (for AWS provider)")
    p.add_argument("--s3-prefix", default="keys/", help="S3 prefix for key objects (default: keys/)")
    p.add_argument("--new-kms-key-id", help="New AWS KMS key id / alias to rewrap with (for AWS)")
    # Vault specific
    p.add_argument("--vault-addr", help="Vault address (for Vault provider)")
    p.add_argument("--vault-token", help="Vault token (for Vault provider)")
    p.add_argument("--vault-kv-prefix", help="Vault KV prefix path (e.g., secret/data/fedgnn/keys)")
    p.add_argument("--new-transit-key-name", help="New Vault transit key name to rewrap with")

    return p.parse_args()


def main():
    args = parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        logger.error("No keys specified")
        sys.exit(2)

    if args.provider == "aws":
        if not args.bucket:
            logger.error("AWS provider requires --bucket")
            sys.exit(2)
        if not args.new_kms_key_id:
            logger.error("AWS provider requires --new-kms-key-id")
            sys.exit(2)
        aws_rotate(bucket=args.bucket, keys=keys, new_kms_key_id=args.new_kms_key_id, dry_run=args.dry_run, backup=args.backup, s3_prefix=args.s3_prefix)
    elif args.provider == "vault":
        vault_addr = args.vault_addr or os.getenv("VAULT_ADDR")
        vault_token = args.vault_token or os.getenv("VAULT_TOKEN")
        if not vault_addr or not vault_token:
            logger.error("Vault provider requires VAULT_ADDR and VAULT_TOKEN")
            sys.exit(2)
        if not args.vault_kv_prefix:
            logger.error("Vault provider requires --vault-kv-prefix")
            sys.exit(2)
        if not args.new_transit_key_name:
            logger.error("Vault provider requires --new-transit-key-name")
            sys.exit(2)
        vault_rotate(vault_addr=vault_addr, token=vault_token, kv_prefix=args.vault_kv_prefix, keys=keys, new_transit_key_name=args.new_transit_key_name, dry_run=args.dry_run, backup=args.backup)
    else:
        logger.error("Unsupported provider: %s", args.provider)
        sys.exit(2)


if __name__ == "__main__":
    main()
