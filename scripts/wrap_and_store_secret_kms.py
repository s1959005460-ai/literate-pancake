# File: scripts/wrap_and_store_secret_kms.py
"""
Usage:
  AWS_REGION=us-east-1 python scripts/wrap_and_store_secret_kms.py \
    --ctx-file ./secrets/tenseal_context.bin --s3-uri s3://my-bucket/tenseal/context_cipher_b64 \
    --kms-key-id alias/my-kms-key

This reads local context bytes, calls KMS.Encrypt to wrap it, base64-encodes ciphertext, and writes to S3 key.
"""
import argparse
import base64
import boto3
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx-file", required=True)
    p.add_argument("--s3-uri", required=True)
    p.add_argument("--kms-key-id", required=True)
    args = p.parse_args()

    with open(args.ctx_file, "rb") as f:
        plaintext = f.read()

    kms = boto3.client("kms", region_name=os.getenv("AWS_REGION", "us-east-1"))
    resp = kms.encrypt(KeyId=args.kms_key_id, Plaintext=plaintext)
    ciphertext = resp["CiphertextBlob"]
    b64 = base64.b64encode(ciphertext).decode()
    # write to S3
    import re
    m = re.match(r"s3://([^/]+)/(.*)", args.s3_uri)
    if not m:
        raise RuntimeError("invalid s3 uri")
    bucket, key = m.group(1), m.group(2)
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    s3.put_object(Bucket=bucket, Key=key, Body=b64)
    print("Stored encrypted context to", args.s3_uri)

if __name__ == "__main__":
    main()
