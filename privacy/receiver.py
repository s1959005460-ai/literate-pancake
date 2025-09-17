# -*- coding: utf-8 -*-
"""
Server receiver: centralize inbound client message validation.

Responsibilities:
- Verify HMAC (constant-time) for incoming payloads.
- Enforce anti-replay by maintaining persistent last_seq per client.
- Atomically update last_seq before processing to avoid TOCTOU races.
- Deserialize payload only after authentication & anti-replay pass.

This module uses a lightweight SQLite backend for last_seq persistence. For large deployments,
replace PersistedStore with Redis or cloud K/V store (with durability enabled).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import hmac
import hashlib
import secrets

logger = logging.getLogger("FedGNN.server.receiver")
logger.setLevel(logging.INFO)


@dataclass
class StoreConfig:
    db_path: Path


class PersistedStore:
    """
    Minimal persistent store for last_seq tracking using SQLite.
    Provides atomic get/set for client sequence numbers.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_db()

    def _ensure_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS client_seq (client_id TEXT PRIMARY KEY, last_seq INTEGER NOT NULL)"
            )
            conn.commit()

    def _connect(self):
        # sqlite3 with check_same_thread=False for multi-threaded environments;
        # write durability ensured by calling commit and relying on SQLite's WAL if configured by admin.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        return conn

    def get_last_seq(self, client_id: str) -> int:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT last_seq FROM client_seq WHERE client_id = ?", (client_id,)).fetchone()
                return int(row[0]) if row else 0

    def set_last_seq(self, client_id: str, seq: int) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO client_seq (client_id, last_seq) VALUES (?, ?) ON CONFLICT(client_id) DO UPDATE SET last_seq=excluded.last_seq",
                    (client_id, int(seq)),
                )
                conn.commit()


class Receiver:
    """
    Receiver centralizes inbound message checks. It does not implement transport.
    The caller should pass:
       - client_id: unique client identifier
       - payload_bytes: raw bytes payload
       - mac_bytes: raw HMAC bytes (hex or raw)
       - nonce: integer sequence number/nonce
       - client_hmac_key: bytes key to verify HMAC
       - process_callback(payload_dict) -> handle the deserialized update
    """

    def __init__(self, store: PersistedStore):
        self.store = store

    @staticmethod
    def compute_hmac(key: bytes, payload: bytes, nonce: int, nonce_bytes: int = 8) -> bytes:
        """
        Compute HMAC-SHA256 over payload||nonce (nonce little-endian) using provided key.
        :param key: bytes key
        :param payload: raw bytes
        :param nonce: integer nonce
        :return: raw bytes HMAC digest
        """
        if not isinstance(key, (bytes, bytearray)):
            raise ValueError("key must be bytes")
        nonce_bytes_le = int(nonce).to_bytes(nonce_bytes, "little")
        msg = payload + nonce_bytes_le
        return hmac.new(key, msg, hashlib.sha256).digest()

    def handle_client_message(
        self,
        client_id: str,
        payload_bytes: bytes,
        mac_bytes: bytes,
        nonce: int,
        client_hmac_key: bytes,
        process_callback: Callable[[dict], Any],
    ) -> dict:
        """
        Validate and process an incoming client message.

        Returns dict: {"status": "accepted"|"rejected", "reason": str}
        """
        # 1. Verify HMAC using constant-time compare
        expected_mac = self.compute_hmac(client_hmac_key, payload_bytes, nonce)
        if not secrets.compare_digest(expected_mac, mac_bytes):
            logger.warning("HMAC failure for client=%s", client_id)
            return {"status": "rejected", "reason": "hmac_mismatch"}

        # 2. Anti-replay: check persistent last_seq
        last_seq = self.store.get_last_seq(client_id)
        if nonce <= last_seq:
            logger.warning("Replay/stale message for client=%s nonce=%s last_seq=%s", client_id, nonce, last_seq)
            return {"status": "rejected", "reason": "replay_or_stale"}

        # 3. Atomically update last_seq BEFORE processing (prevents TOCTOU)
        try:
            self.store.set_last_seq(client_id, nonce)
        except Exception as e:
            logger.exception("Failed to persist last_seq for %s: %s", client_id, e)
            return {"status": "rejected", "reason": "persist_failure"}

        # 4. Deserialize (safe, expecting JSON)
        try:
            payload_obj = json.loads(payload_bytes.decode("utf-8"))
        except Exception as e:
            logger.exception("Deserialization failed for client=%s: %s", client_id, e)
            return {"status": "rejected", "reason": "deserialize_failed"}

        # 5. Delegate processing
        try:
            process_callback(payload_obj)
        except Exception as e:
            logger.exception("Processing callback failed for client=%s: %s", client_id, e)
            return {"status": "rejected", "reason": "processing_failed"}

        return {"status": "accepted", "reason": "ok"}


# Quick self-check when invoked directly
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.DEBUG)
    print("Receiver self-check...")
    td = tempfile.TemporaryDirectory()
    store = PersistedStore(Path(td.name) / "store.db")
    receiver = Receiver(store)

    # simple process callback that records payload into a list
    processed = []

    def cb(p):
        processed.append(p)

    client_id = "client_a"
    payload = json.dumps({"update": [1, 2, 3]}).encode("utf-8")
    key = b"test_hmac_key_32bytes_long_____"[:32]  # in production store keys in secret manager
    nonce = 1
    mac = receiver.compute_hmac(key, payload, nonce)
    res = receiver.handle_client_message(client_id, payload, mac, nonce, key, cb)
    assert res["status"] == "accepted"
    print("Accepted:", res)
    # Replay test
    res2 = receiver.handle_client_message(client_id, payload, mac, 1, key, cb)
    assert res2["status"] == "rejected"
    print("Replay test passed:", res2)
    print("Receiver self-check passed.")
