# File: scripts/store_secret_vault.py
"""
Store TenSEAL context into Vault KV v2 under path.
Usage:
  VAULT_ADDR=... VAULT_TOKEN=... python scripts/store_secret_vault.py --ctx-file ./secrets/tenseal_context.bin --vault-path secret/data/tenseal/context
"""
import argparse
import base64
import hvac
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx-file", required=True)
    p.add_argument("--vault-path", required=True)
    args = p.parse_args()

    with open(args.ctx_file, "rb") as f:
        ctx = f.read()
    b64 = base64.b64encode(ctx).decode()
    client = hvac.Client(url=os.environ.get("VAULT_ADDR"), token=os.environ.get("VAULT_TOKEN"))
    # KV v2 expects data: { 'data': { ... } } when using API; hvac handles it via create_or_update_secret(...)
    # but we assume kv v2 mount at 'secret'
    key_path = args.vault_path
    if key_path.startswith("secret/data/"):
        key_path = key_path[len("secret/data/"):]
    client.secrets.kv.v2.create_or_update_secret(path=key_path, secret={"ciphertext_b64": b64})
    print("Stored secret to vault path:", args.vault_path)

if __name__ == "__main__":
    main()
