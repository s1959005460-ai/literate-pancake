# File: decryptor/decryptor_service.py
"""
Decryptor service for TenSEAL aggregated ciphertext.

Security model (recommended):
 - Secrets storing TenSEAL secret context are encrypted with KMS or stored in Vault
 - Decryptor will call KMS/Vault to obtain plaintext context just-in-time (JIT),
   keep in memory, use for decryption, and then zero memory if required.
 - Decryptor should run on isolated nodes with strict network policies.

Environment variables:
 - MODE: 'aws_kms' or 'vault' (choose secret backend)
 - AWS_REGION: required for AWS KMS
 - KMS_CIPHER_SECRET_KEY_NAME: S3/keyname or parameter to find ciphertext blob (depends on implementation)
 - VAULT_ADDR, VAULT_TOKEN, VAULT_SECRET_PATH: for Vault backend
 - SECRET_KEY_REF: path or identifier of stored encrypted TenSEAL context
 - AUTH_TOKEN: optional bearer token required for client calls to decryptor (for simple auth)
"""
from __future__ import annotations
import os
import base64
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, start_http_server
import boto3
import hvac
import time

# TenSEAL import (must exist in runtime)
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except Exception:
    TENSEAL_AVAILABLE = False

# logging (structured)
logger = logging.getLogger("decryptor")
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Metrics
REQUESTS = Counter("decryptor_requests_total", "Total decryptor requests")
FAILURES = Counter("decryptor_failures_total", "Total decryptor failures")
LATENCY_MS = Counter("decryptor_latency_ms_total", "decryptor latency ms sum (for avg calculation)")

# Config
MODE = os.getenv("MODE", "aws_kms")  # 'aws_kms' or 'vault'
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KMS_SECRET_S3_URI = os.getenv("KMS_SECRET_S3_URI")  # optional: where encrypted context is stored
KMS_SECRET_ALIAS = os.getenv("KMS_SECRET_ALIAS")    # or KMS alias / key id for decrypt
VAULT_ADDR = os.getenv("VAULT_ADDR")
VAULT_TOKEN = os.getenv("VAULT_TOKEN")
VAULT_SECRET_PATH = os.getenv("VAULT_SECRET_PATH", "secret/data/tenseal/context")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # Simple bearer token for api access (recommend mTLS instead)
PROM_PORT = int(os.getenv("PROM_PORT", "9200"))

app = FastAPI(title="TenSEAL Decryptor")

# Pydantic schemas
class DecryptRequest(BaseModel):
    round: int
    agg_cipher_b64: str  # base64 of aggregated ciphertext (may be concatenated pieces)
    meta: Optional[Dict[str, str]] = None


class DecryptResponse(BaseModel):
    round: int
    model_delta: Any  # JSON serializable; typically list[float] or compressed bytes (base64)


# Helper: simple auth dependency
async def check_auth(authorization: Optional[str] = Header(None)):
    if AUTH_TOKEN:
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header required")
        if authorization.strip() not in (f"Bearer {AUTH_TOKEN}", AUTH_TOKEN):
            raise HTTPException(status_code=403, detail="Forbidden")
    # otherwise no auth required (dev only)


# KMS helper: use boto3 KMS to decrypt stored ciphertext (expects base64-encoded ciphertext blob)
def decrypt_secret_from_kms(kms_cipher_b64: str) -> bytes:
    """
    kms_cipher_b64: base64 of the ciphertext blob encrypted by KMS (Encrypt API)
    returns: plaintext bytes (the tenseal serialized context)
    """
    try:
        kms = boto3.client("kms", region_name=AWS_REGION)
        cipher_blob = base64.b64decode(kms_cipher_b64)
        # Call KMS decrypt
        resp = kms.decrypt(CiphertextBlob=cipher_blob)
        plaintext = resp["Plaintext"]
        logger.info("KMS decrypt successful (plaintext len=%d)", len(plaintext))
        return plaintext
    except Exception as e:
        logger.exception("KMS decrypt failed")
        raise


def fetch_encrypted_context_from_s3(s3_uri: str) -> str:
    """
    optional helper: s3://bucket/path returns base64 ciphertext string
    """
    import boto3
    import re
    m = re.match(r"s3://([^/]+)/(.+)", s3_uri)
    if not m:
        raise ValueError("invalid s3 uri")
    bucket, key = m.group(1), m.group(2)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    # assume object contains base64-encoded ciphertext
    return data.decode().strip()


def decrypt_secret_from_vault(vault_secret_path: str) -> bytes:
    """
    Example with KV v2: path like 'secret/data/tenseal/context' where data: { 'ciphertext_b64': '...' }
    Or if using transit, one would use transit/decrypt with ciphertext string.
    """
    try:
        if not VAULT_ADDR or not VAULT_TOKEN:
            raise RuntimeError("Vault config missing")
        client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)
        if not client.is_authenticated():
            raise RuntimeError("Vault auth failed")
        # Read secret (KV v2)
        # vault kv v2 read_secret_version expects path without 'secret/data/' prefix
        path = vault_secret_path
        # tolerate several forms
        if path.startswith("secret/data/"):
            key_path = path[len("secret/data/"):]
        elif path.startswith("secret/"):
            key_path = path[len("secret/"):]
        else:
            key_path = path
        resp = client.secrets.kv.v2.read_secret_version(path=key_path)
        data = resp.get("data", {}).get("data", {})
        cipher_b64 = data.get("ciphertext_b64")
        if not cipher_b64:
            raise RuntimeError("ciphertext_b64 not found in Vault secret")
        # If cipher_b64 is actually KMS wrapped ciphertext, call decrypt_secret_from_kms
        # Here we assume cipher_b64 is the KMS-wrapped blob or raw TenSEAL serialized secret base64
        if cipher_b64.startswith("kms:"):
            # format kms:BASE64
            b64 = cipher_b64[len("kms:"):]
            return decrypt_secret_from_kms(b64)
        return base64.b64decode(cipher_b64)
    except Exception:
        logger.exception("Vault decrypt failed")
        raise


# TenSEAL context loader (dev & production)
def load_tenseal_context_from_bytes(ctx_bytes: bytes):
    if not TENSEAL_AVAILABLE:
        logger.error("TenSEAL not installed in runtime")
        raise RuntimeError("TenSEAL unavailable")
    try:
        # The TenSEAL API provides ts.context_from(serialized) or similar
        ctx = ts.context_from(ctx_bytes)
        # ensure secret loaded if present
        return ctx
    except Exception:
        logger.exception("failed to load tenseal context")
        raise


# Decrypt endpoint
@app.post("/decrypt", response_model=DecryptResponse)
async def decrypt_endpoint(req: DecryptRequest, authorized: None = Depends(check_auth)):
    start = time.time()
    REQUESTS.inc()
    try:
        agg_b64 = req.agg_cipher_b64
        # Step 1: fetch encrypted TenSEAL secret context
        # Production design: secret is stored encrypted (KMS) in S3 or Vault.
        secret_bytes: Optional[bytes] = None
        if MODE == "aws_kms":
            if KMS_SECRET_S3_URI:
                logger.info("fetching encrypted context from S3 URI")
                cipher_b64 = fetch_encrypted_context_from_s3(KMS_SECRET_S3_URI)
                secret_bytes = decrypt_secret_from_kms(cipher_b64)
            else:
                # Alternatively user may provide KMS_CIPHER value in env directly (not recommended)
                kms_cipher_b64 = os.getenv("KMS_CIPHER_B64")
                if not kms_cipher_b64:
                    raise HTTPException(status_code=500, detail="KMS cipher not configured")
                secret_bytes = decrypt_secret_from_kms(kms_cipher_b64)
        elif MODE == "vault":
            # read from Vault KV
            secret_bytes = decrypt_secret_from_vault(VAULT_SECRET_PATH)
        else:
            raise HTTPException(status_code=500, detail="Invalid MODE for decryptor")

        # Step 2: load TenSEAL context (in-memory)
        ctx = load_tenseal_context_from_bytes(secret_bytes)

        # Step 3: decode ciphertext(s)
        # If agg_b64 is concatenated ciphertexts, you must follow your serialization contract.
        # For demo we assume single ciphertext serialized as base64.
        try:
            ct_bytes = base64.b64decode(agg_b64)
        except Exception:
            logger.exception("invalid base64 ciphertext")
            raise HTTPException(status_code=400, detail="invalid base64 ciphertext")

        # Step 4: decrypt (TenSEAL ckks vector)
        try:
            # Depending on TenSEAL version, decrypt method varies. Example:
            vec = ts.CKKSVector.load_from(ctx, ct_bytes)
            plaintext = vec.decrypt()
            # plaintext is list of floats
        except Exception:
            # fallback alternative API: ctx.ckks_vector_from(...)
            try:
                # Alternative loading:
                v = ts.ckks_vector(ctx, [])
                v = ts.CKKSVector.load_from(ctx, ct_bytes)
                plaintext = v.decrypt()
            except Exception:
                logger.exception("tenseal decryption failed")
                raise HTTPException(status_code=500, detail="tenseal decrypt failed")

        # Optional: post-process (e.g., convert to quantized/serialized form)
        resp = DecryptResponse(round=req.round, model_delta=plaintext)
        elapsed = (time.time() - start) * 1000.0
        LATENCY_MS.inc(elapsed)
        return resp
    except HTTPException:
        FAILURES.inc()
        raise
    except Exception:
        FAILURES.inc()
        logger.exception("decrypt endpoint failed")
        raise HTTPException(status_code=500, detail="internal error")


# Health endpoints
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    # perform quick KMS/Vault access check depending on MODE (non-blocking)
    try:
        if MODE == "aws_kms":
            # try a lightweight KMS call with DryRun? There's no dry run; so either validate env variables
            return {"ready": True}
        else:
            # vault check
            client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)
            if not client.is_authenticated():
                return {"ready": False}
            return {"ready": True}
    except Exception:
        return {"ready": False}


# Startup tasks
@app.on_event("startup")
async def startup_event():
    # start prometheus metrics server on PROM_PORT
    start_http_server(PROM_PORT)
    logger.info("Prometheus metrics started on port %d", PROM_PORT)
    # any further initialization can be done here


# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("decryptor shutting down")


# If running as main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("decryptor_service:app", host="0.0.0.0", port=int(os.getenv("PORT", "9000")), log_level="info")
