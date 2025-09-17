# File: scripts/generate_keys.py
"""
Generate X25519 server priv/pub and TenSEAL context (for local testing).
Outputs:
 - server_x25519_priv.bin
 - server_x25519_pub.bin
 - tenseal_context.bin (if TenSEAL installed)
"""
from __future__ import annotations

import os
import sys
import base64
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization

OUT_DIR = os.getenv("OUT_DIR", "./secrets")
os.makedirs(OUT_DIR, exist_ok=True)

def gen_x25519():
    priv = X25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
    pub_bytes = priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    with open(os.path.join(OUT_DIR, "server_x25519_priv.bin"), "wb") as f:
        f.write(priv_bytes)
    with open(os.path.join(OUT_DIR, "server_x25519_pub.bin"), "wb") as f:
        f.write(pub_bytes)
    print("Generated X25519 keys in", OUT_DIR)

def gen_tenseal():
    try:
        import tenseal as ts
    except Exception:
        print("TenSEAL not available; skipping HE context generation")
        return
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    ctx_ser = context.serialize()
    with open(os.path.join(OUT_DIR, "tenseal_context.bin"), "wb") as f:
        f.write(ctx_ser)
    print("Generated TenSEAL context in", OUT_DIR)

if __name__ == "__main__":
    gen_x25519()
    gen_tenseal()
