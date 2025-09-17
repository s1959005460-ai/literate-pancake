# -*- coding: utf-8 -*-
"""
Privacy auditor for FedGNN_advanced.
Provides append-only, signed audit logging for privacy-sensitive operations.
"""

from __future__ import annotations

import os
import json
import time
import hmac
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# Environment variable name for audit secret
AUDIT_SECRET_ENV = 'AUDIT_SECRET'


class PrivacyAuditor:
    """
    Privacy auditor that writes signed audit entries to a JSON-lines file.

    Args:
        run_dir: Directory where audit logs will be stored
        audit_secret_env: Environment variable name for the audit secret

    Raises:
        RuntimeError: If audit secret is not set in environment
    """

    def __init__(self, run_dir: Path, audit_secret_env: str = AUDIT_SECRET_ENV):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.run_dir / 'privacy_audit.jsonl'

        # Get audit secret from environment
        secret = os.getenv(audit_secret_env)
        if not secret:
            raise RuntimeError(
                f"Environment variable {audit_secret_env} must be set. "
                "Use a secret manager in production."
            )

        self._key = secret.encode('utf-8')

    def _sign(self, payload_bytes: bytes) -> str:
        """
        Generate HMAC-SHA256 signature for payload.

        Args:
            payload_bytes: Payload bytes to sign

        Returns:
            Hex digest of the signature
        """
        return hmac.new(self._key, payload_bytes, hashlib.sha256).hexdigest()

    def record_round(
        self,
        round_idx: int,
        clients: List[str],
        clients_participation_rate: float,
        local_sample_rates: Dict[str, float],
        noise_multiplier: float,
        max_grad_norm: float,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a privacy audit entry for a round.

        Args:
            round_idx: Round index
            clients: List of client IDs that participated
            clients_participation_rate: Fraction of clients that participated
            local_sample_rates: Dictionary mapping client IDs to local sample rates
            noise_multiplier: Noise multiplier used in DP
            max_grad_norm: Maximum gradient norm used in DP
            extra: Additional metadata

        Returns:
            The recorded audit entry
        """
        payload = {
            'ts': int(time.time()),
            'round': int(round_idx),
            'clients': clients,
            'clients_participation_rate': float(clients_participation_rate),
            'local_sample_rates': {str(k): float(v) for k, v in local_sample_rates.items()},
            'noise_multiplier': float(noise_multiplier),
            'max_grad_norm': float(max_grad_norm),
            'extra': extra or {}
        }

        # Serialize and sign payload
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        signature = self._sign(payload_bytes)

        # Create entry
        entry = {'payload': payload, 'sig': signature}

        # Append to audit file with durability
        with open(self.audit_file, 'ab') as f:
            line = json.dumps(entry, ensure_ascii=False) + '\n'
            f.write(line.encode('utf-8'))
            f.flush()
            os.fsync(f.fileno())

        return entry

    def verify_entries(self) -> bool:
        """
        Verify all entries in the audit log.

        Returns:
            True if all entries are valid, False otherwise
        """
        if not self.audit_file.exists():
            return True

        with open(self.audit_file, 'rb') as f:
            for line in f:
                entry = json.loads(line.decode('utf-8'))
                payload_bytes = json.dumps(
                    entry['payload'], sort_keys=True, separators=(',', ':')
                ).encode('utf-8')

                expected_sig = self._sign(payload_bytes)
                if not hmac.compare_digest(expected_sig, entry['sig']):
                    return False

        return True


# Self-test and demonstration
if __name__ == '__main__':
    print("Running PrivacyAuditor self-test...")

    import tempfile

    # Set test secret
    os.environ[AUDIT_SECRET_ENV] = 'test_audit_secret_do_not_use_in_production'

    with tempfile.TemporaryDirectory() as temp_dir:
        auditor = PrivacyAuditor(Path(temp_dir))

        # Record a test entry
        entry = auditor.record_round(
            round_idx=1,
            clients=['client1', 'client2'],
            clients_participation_rate=0.5,
            local_sample_rates={'client1': 0.1, 'client2': 0.2},
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            extra={'test': True}
        )

        print("✓ Audit entry created successfully")

        # Verify the entry
        payload_bytes = json.dumps(
            entry['payload'], sort_keys=True, separators=(',', ':')
        ).encode('utf-8')

        expected_sig = hmac.new(
            os.environ[AUDIT_SECRET_ENV].encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()

        assert hmac.compare_digest(expected_sig, entry['sig'])
        print("✓ Signature verification passed")

        # Verify all entries
        assert auditor.verify_entries()
        print("✓ All entries verified successfully")

    print("PrivacyAuditor self-test completed successfully!")
