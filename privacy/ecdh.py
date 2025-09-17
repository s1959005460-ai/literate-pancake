# FedGNN_advanced/privacy/ecdh.py
import logging
from typing import Tuple
try:
    from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.exceptions import InvalidSignature
except Exception as e:
    raise ImportError("cryptography is required for ecdh utilities") from e

logger = logging.getLogger("fedgnn.ecdh")

def generate_keypair() -> Tuple[bytes, bytes]:
    priv = x25519.X25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    return priv_bytes, pub_bytes

def validate_public_key(pub_bytes: bytes) -> bool:
    try:
        x25519.X25519PublicKey.from_public_bytes(pub_bytes)
        return True
    except Exception:
        return False

def shared_secret_to_hkdf(priv_bytes: bytes, peer_pub_bytes: bytes, length: int = 32, info: bytes = b"") -> bytes:
    if not validate_public_key(peer_pub_bytes):
        raise ValueError("invalid peer public key")
    priv = x25519.X25519PrivateKey.from_private_bytes(priv_bytes)
    peer = x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes)
    shared = priv.exchange(peer)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info)
    return hkdf.derive(shared)

# Optional Ed25519 helpers to prove authenticity of public keys (recommended)
def sign_ed25519(priv_sig_bytes: bytes, message: bytes) -> bytes:
    sk = ed25519.Ed25519PrivateKey.from_private_bytes(priv_sig_bytes)
    return sk.sign(message)

def verify_ed25519(pub_sig_bytes: bytes, signature: bytes, message: bytes) -> bool:
    try:
        pk = ed25519.Ed25519PublicKey.from_public_bytes(pub_sig_bytes)
        pk.verify(signature, message)
        return True
    except InvalidSignature:
        return False
    except Exception:
        return False
