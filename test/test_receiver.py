# -*- coding: utf-8 -*-
"""
Tests for server receiver module.
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from FedGNN_advanced.server.receiver import Receiver, PersistedStore


def test_receiver_hmac_verification():
    """Test HMAC verification in receiver."""
    with TemporaryDirectory() as temp_dir:
        store = PersistedStore(Path(temp_dir) / 'test.db')
        receiver = Receiver(store)

        # Test data
        client_id = 'test_client'
        payload = {'data': 'test'}
        payload_bytes = json.dumps(payload).encode('utf-8')
        key = b'test_key_32_bytes_long_______'
        nonce = 1

        # Compute valid HMAC
        valid_mac = receiver.compute_hmac(key, payload_bytes, nonce)

        # Process callback
        processed = []

        def callback(data):
            processed.append(data)

        # Test valid message
        result = receiver.handle_client_message(
            client_id, payload_bytes, valid_mac, nonce, key, callback
        )

        assert result['status'] == 'accepted'
        assert len(processed) == 1
        assert processed[0] == payload


def test_receiver_replay_protection():
    """Test replay protection in receiver."""
    with TemporaryDirectory() as temp_dir:
        store = PersistedStore(Path(temp_dir) / 'test.db')
        receiver = Receiver(store)

        # Test data
        client_id = 'test_client'
        payload = {'data': 'test'}
        payload_bytes = json.dumps(payload).encode('utf-8')
        key = b'test_key_32_bytes_long_______'
        nonce = 1

        # Compute HMAC
        mac = receiver.compute_hmac(key, payload_bytes, nonce)

        # Process callback
        processed = []

        def callback(data):
            processed.append(data)

        # First message should be accepted
        result1 = receiver.handle_client_message(
            client_id, payload_bytes, mac, nonce, key, callback
        )
        assert result1['status'] == 'accepted'
        assert len(processed) == 1

        # Second message with same nonce should be rejected
        result2 = receiver.handle_client_message(
            client_id, payload_bytes, mac, nonce, key, callback
        )
        assert result2['status'] == 'rejected'
        assert result2['reason'] == 'replay_or_stale'
        assert len(processed) == 1  # Callback should not be called again


def test_receiver_hmac_tampering():
    """Test that tampered HMAC is rejected."""
    with TemporaryDirectory() as temp_dir:
        store = PersistedStore(Path(temp_dir) / 'test.db')
        receiver = Receiver(store)

        # Test data
        client_id = 'test_client'
        payload = {'data': 'test'}
        payload_bytes = json.dumps(payload).encode('utf-8')
        key = b'test_key_32_bytes_long_______'
        nonce = 1

        # Compute valid HMAC then tamper with it
        valid_mac = receiver.compute_hmac(key, payload_bytes, nonce)
        tampered_mac = bytearray(valid_mac)
        tampered_mac[0] ^= 0xFF  # Flip first bit

        # Process callback
        processed = []

        def callback(data):
            processed.append(data)

        # Tampered message should be rejected
        result = receiver.handle_client_message(
            client_id, payload_bytes, bytes(tampered_mac), nonce, key, callback
        )

        assert result['status'] == 'rejected'
        assert result['reason'] == 'hmac_mismatch'
        assert len(processed) == 0  # Callback should not be called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
