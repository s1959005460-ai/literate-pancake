# -*- coding: utf-8 -*-
"""
Integration negative tests: simulate adverse conditions like HMAC tampering and replay.
These are not exhaustive but cover key negative paths expected by the audit.

Run:
    python tests/test_integration_negative.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os

from FedGNN_advanced.server.receiver import PersistedStore, Receiver

def test_hmac_tamper_and_replay():
    with TemporaryDirectory() as td:
        db = Path(td) / "store.db"
        store = PersistedStore(db)
        recv = Receiver(store)
        client_id = "bad_client"
        payload = json.dumps({"data": 123}).encode("utf-8")
        key = b"testkey_for_unit_test_32bytes____"
        nonce = 10
        mac = recv.compute_hmac(key, payload, nonce)
        # tamper mac
        tampered = bytearray(mac)
        tampered[0] ^= 0xFF
        res = recv.handle_client_message(client_id, payload, bytes(tampered), nonce, key, lambda p: None)
        assert res["status"] == "rejected" and res["reason"] == "hmac_mismatch"
        # proper message accepted
        res2 = recv.handle_client_message(client_id, payload, mac, nonce, key, lambda p: None)
        assert res2["status"] == "accepted"
        # replay should be rejected
        res3 = recv.handle_client_message(client_id, payload, mac, nonce, key, lambda p: None)
        assert res3["status"] == "rejected" and res3["reason"] == "replay_or_stale"

if __name__ == "__main__":
    print("Running negative integration tests...")
    test_hmac_tamper_and_replay()
    print("Negative integration tests passed.")
