# client/client_uploader.py
"""
客户端示例：创建 gRPC secure channel with mTLS, produce UploadRequest with proper HMAC.
- Client holds its X25519 key pair (private) and server's public for shared key derivation or receives server pub.
"""
import os
import grpc
import time
import base64
from proto import federated_service_pb2 as pb
from proto import federated_service_pb2_grpc as rpc
from crypto.crypto_utils import hmac_sign
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

SERVER_ADDR = os.getenv("SERVER_ADDR", "edge-aggregator:50051")
CLIENT_KEY = os.getenv("CLIENT_PRIV_KEY", "./client_priv.bin")
CLIENT_CERT = os.getenv("CLIENT_CERT", "./client.crt")
CLIENT_KEY_FILE = os.getenv("CLIENT_KEY_FILE", "./client.key")
CA_CERT_FILE = os.getenv("CA_CERT", "./ca.crt")
SERVER_PUB_B64 = os.getenv("SERVER_PUB_B64", None)  # optional

def load_priv():
    with open(CLIENT_KEY, "rb") as f:
        return f.read()

def main():
    # setup mTLS credentials
    with open(CLIENT_CERT, "rb") as f:
        cert = f.read()
    with open(CLIENT_KEY_FILE, "rb") as f:
        key = f.read()
    with open(CA_CERT_FILE, "rb") as f:
        ca = f.read()

    creds = grpc.ssl_channel_credentials(root_certificates=ca, private_key=key, certificate_chain=cert)
    channel = grpc.secure_channel(SERVER_ADDR, creds)
    stub = rpc.FederatedServiceStub(channel)

    # build request
    client_priv = load_priv()
    # generate ephemeral pub (for demonstration)
    priv = X25519PrivateKey.from_private_bytes(client_priv)
    pub = priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

    payload = b"client masked or HE ciphertext"  # in real: encrypt/pack/compress
    seq = 1
    seq_bytes = seq.to_bytes(8, "big")
    # derive shared key with server pub provided by env (here server pub must be known)
    if SERVER_PUB_B64:
        server_pub = base64.b64decode(SERVER_PUB_B64)
        shared = derive_shared_key(client_priv, server_pub)
        mac = hmac_sign(shared, seq_bytes + payload)
    else:
        mac = b"\x00" * 32

    req = pb.UploadRequest(client_id="client-1", round=1, seq=seq, sender_pub=pub, payload=payload, mac=mac)
    resp = stub.UploadUpdate(req, timeout=10)
    print("Upload response:", resp.ok, resp.message)
