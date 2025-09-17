# File: locust/locustfile.py
from locust import HttpUser, task, between
import base64, secrets, json
from crypto.crypto_utils import hmac_sign
import os

SERVER_PUB = os.getenv("SERVER_PUB_B64")  # not used here in simulation

class FedClient(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def upload(self):
        client_id = "c-" + secrets.token_hex(8)
        seq = 1
        payload = secrets.token_bytes(2048)
        # For simulation generate random mac (servers will reject), in real test set proper mac via shared key
        mac = secrets.token_bytes(32)
        req = {
            "client_id": client_id,
            "round": 1,
            "seq": seq,
            "sender_pub": base64.b64encode(secrets.token_bytes(32)).decode(),
            "payload": base64.b64encode(payload).decode(),
            "mac": base64.b64encode(mac).decode()
        }
        self.client.post("/upload_update", json=req)
