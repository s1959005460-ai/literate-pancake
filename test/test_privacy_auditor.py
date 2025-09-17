# -*- coding: utf-8 -*-
"""
Tests for privacy auditor module.
"""

import os
import json
import hmac
import hashlib
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from FedGNN_advanced.privacy.auditor import PrivacyAuditor


def test_auditor_signature():
    """Test that auditor creates and verifies signatures correctly."""
    os.environ['AUDIT_SECRET'] = 'test_secret'

    with TemporaryDirectory() as temp_dir:
        auditor = PrivacyAuditor(Path(temp_dir))

        # Record a round
        entry = auditor.record_round(
            round_idx=1,
            clients=['client1', 'client2'],
            clients_participation_rate=0.5,
            local_sample_rates={'client1': 0.1, 'client2': 0.2},
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )

        # Verify signature
        payload_bytes = json.dumps(
            entry['payload'], sort_keys=True, separators=(',', ':')
        ).encode('utf-8')

        expected_sig = hmac.new(
            os.environ['AUDIT_SECRET'].encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()

        assert entry['sig'] == expected_sig


def test_auditor_verification():
    """Test that auditor can verify its own entries."""
    os.environ['AUDIT_SECRET'] = 'test_secret'

    with TemporaryDirectory() as temp_dir:
        auditor = PrivacyAuditor(Path(temp_dir))

        # Record multiple rounds
        for i in range(3):
            auditor.record_round(
                round_idx=i,
                clients=[f'client{i}'],
                clients_participation_rate=0.5,
                local_sample_rates={f'client{i}': 0.1},
                noise_multiplier=1.0,
                max_grad_norm=1.0
            )

        # Verify all entries
        assert auditor.verify_entries()


def test_auditor_missing_secret():
    """Test error when audit secret is not set."""
    # Temporarily remove secret
    if 'AUDIT_SECRET' in os.environ:
        del os.environ['AUDIT_SECRET']

    with TemporaryDirectory() as temp_dir:
        with pytest.raises(RuntimeError, match="must be set"):
            PrivacyAuditor(Path(temp_dir))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
