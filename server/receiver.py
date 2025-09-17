# -*- coding: utf-8 -*-
"""
Server-side receiver with HMAC verification and persistent anti-replay protection.
Uses SQLite for persistent storage of client sequence numbers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import hmac
import hashlib
import secrets

logger = logging.getLogger('FedGNN.server.receiver')
logger.setLevel(logging.INFO)


class PersistedStore:
    """
    Persistent storage for client sequence numbers using SQLite.

    Args:
        db_path: Path to SQLite database file
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        return sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0
        )

    def _ensure_db(self) -> None:
        """Ensure the database table exists."""
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS client_seq (
                    client_id TEXT PRIMARY KEY,
                    last_seq INTEGER NOT NULL
                )
            ''')
            conn.commit()

    def get_last_seq(self, client_id: str) -> int:
        """
        Get the last sequence number for a client.

        Args:
            client_id: Client identifier

        Returns:
            Last sequence number, or 0 if not found
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                'SELECT last_seq FROM client_seq WHERE client_id = ?',
                (client_id,)
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def set_last_seq(self, client_id: str, seq: int) -> None:
        """
        Set the last sequence number for a client.

        Args:
            client_id: Client identifier
            seq: Sequence number to set
        """
        with self._lock, self._connect() as conn:
            conn.execute('''
                INSERT INTO client_seq (client_id, last_seq)
                VALUES (?, ?)
                ON CONFLICT(client_id) DO UPDATE SET last_seq = excluded.last_seq
            ''', (client_id, int(seq)))
            conn.commit()


class Receiver:
    """
    Server-side message receiver with HMAC verification and anti-replay protection.

    Args:
        store: PersistedStore instance for sequence number storage
    """

    def __init__(self, store: PersistedStore):
        self.store = store

    @staticmethod
    def compute_hmac(key: bytes, payload: bytes, nonce: int, nonce_bytes: int = 8) -> bytes:
        """
        Compute HMAC-SHA256 for a payload with nonce.

        Args:
            key: HMAC key bytes
            payload: Message payload bytes
            nonce: Nonce value
            nonce_bytes: Number of bytes to use for nonce encoding

        Returns:
            HMAC digest bytes
        """
        if not isinstance(key, (bytes, bytearray)):
            raise ValueError("HMAC key must be bytes")

        nonce_bytes_le = nonce.to_bytes(nonce_bytes, 'little')
        message = payload + nonce_bytes_le
        return hmac.new(key, message, hashlib.sha256).digest()

    def handle_client_message(
        self,
        client_id: str,
        payload_bytes: bytes,
        mac_bytes: bytes,
        nonce: int,
        client_hmac_key: bytes,
        process_callback: Callable[[dict], Any]
    ) -> dict:
        """
        Process a client message with security validation.

        Args:
            client_id: Client identifier
            payload_bytes: Message payload bytes
            mac_bytes: Received HMAC bytes
            nonce: Message nonce
            client_hmac_key: HMAC key for this client
            process_callback: Callback to process valid messages

        Returns:
            Result dictionary with status and reason
        """
        # 1. Verify HMAC (constant-time comparison)
        expected_mac = self.compute_hmac(client_hmac_key, payload_bytes, nonce)
        if not secrets.compare_digest(expected_mac, mac_bytes):
            logger.warning("HMAC verification failed for client %s", client_id)
            return {'status': 'rejected', 'reason': 'hmac_mismatch'}

        # 2. Check anti-replay protection
        last_seq = self.store.get_last_seq(client_id)
        if nonce <= last_seq:
            logger.warning(
                "Replay or stale message from client %s: nonce=%d, last_seq=%d",
                client_id, nonce, last_seq
            )
            return {'status': 'rejected', 'reason': 'replay_or_stale'}

        # 3. Atomically update sequence number before processing
        try:
            self.store.set_last_seq(client_id, nonce)
        except Exception as e:
            logger.exception("Failed to persist sequence number for client %s: %s", client_id, e)
            return {'status': 'rejected', 'reason': 'persist_error'}

        # 4. Deserialize payload
        try:
            payload_obj = json.loads(payload_bytes.decode('utf-8'))
        except Exception as e:
            logger.exception("Deserialization failed for client %s: %s", client_id, e)
            return {'status': 'rejected', 'reason': 'deserialize_failed'}

        # 5. Process the message
        try:
            process_callback(payload_obj)
        except Exception as e:
            logger.exception("Processing callback failed for client %s: %s", client_id, e)
            return {'status': 'rejected', 'reason': 'processing_failed'}

        return {'status': 'accepted', 'reason': 'ok'}


# Self-test and demonstration
if __name__ == '__main__':
    print("Running Receiver self-test...")

    import tempfile

    # Set up logging for the test
    logging.basicConfig(level=logging.DEBUG)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create persisted store
        db_path = Path(temp_dir) / 'test.db'
        store = PersistedStore(db_path)
        receiver = Receiver(store)

        # Test data
        client_id = 'test_client'
        payload = {'update': [1.0, 2.0, 3.0]}
        payload_bytes = json.dumps(payload).encode('utf-8')
        key = b'test_hmac_key_32_bytes_long_______'
        nonce = 1

        # Compute HMAC
        mac = receiver.compute_hmac(key, payload_bytes, nonce)

        # Test processing callback
        processed_messages = []


        def test_callback(message):
            processed_messages.append(message)


        # Test valid message
        result = receiver.handle_client_message(
            client_id, payload_bytes, mac, nonce, key, test_callback
        )

        assert result['status'] == 'accepted'
        assert len(processed_messages) == 1
        assert processed_messages[0] == payload
        print("✓ Valid message test passed")

        # Test replay attack
        result2 = receiver.handle_client_message(
            client_id, payload_bytes, mac, nonce, key, test_callback
        )

        assert result2['status'] == 'rejected'
        assert result2['reason'] == 'replay_or_stale'
        print("✓ Replay protection test passed")

        # Test HMAC tampering
        tampered_mac = bytearray(mac)
        tampered_mac[0] ^= 0xFF  # Flip first bit

        result3 = receiver.handle_client_message(
            client_id, payload_bytes, bytes(tampered_mac), nonce + 1, key, test_callback
        )

        assert result3['status'] == 'rejected'
        assert result3['reason'] == 'hmac_mismatch'
        print("✓ HMAC tampering test passed")

    print("All Receiver tests passed successfully!")
