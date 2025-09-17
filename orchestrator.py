# -*- coding: utf-8 -*-
"""
Top-level orchestrator for FedGNN_advanced.
Coordinates the entire federated learning process with proper error handling.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from FedGNN_advanced.server.receiver import PersistedStore, Receiver
from FedGNN_advanced.aggregator import Aggregator, AggregationError
from FedGNN_advanced.privacy.auditor import PrivacyAuditor
from FedGNN_advanced.privacy import bonawitz_protocol as bzp
from FedGNN_advanced.privacy.errors import ProtocolAbortError
from FedGNN_advanced.compression import serialize_sparse, deserialize_sparse

logger = logging.getLogger('FedGNN.orchestrator')
logger.setLevel(logging.INFO)


class Orchestrator:
    """
    Main orchestrator for federated learning rounds.

    Args:
        run_dir: Directory for run-specific data
        db_path: Path to database for persistent storage
        expected_shapes: Expected parameter shapes
        auditor: Privacy auditor instance
        report_hook: Optional callback for reporting metrics
    """

    def __init__(
        self,
        run_dir: Path,
        db_path: Path,
        expected_shapes: Dict[str, tuple],
        auditor: PrivacyAuditor,
        report_hook: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.store = PersistedStore(db_path)
        self.receiver = Receiver(self.store)
        self.aggregator = Aggregator(expected_shapes, report_hook=report_hook)
        self.auditor = auditor
        self.report_hook = report_hook

        # State management
        self.current_round = 0
        self.global_state = self._initialize_global_state(expected_shapes)

    def _initialize_global_state(self, expected_shapes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        """Initialize the global model state with zeros."""
        state = {}
        for param_name, shape in expected_shapes.items():
            state[param_name] = np.zeros(shape, dtype=np.float32)
        return state

    def _process_client_message(
        self,
        client_id: str,
        payload_bytes: bytes,
        mac_bytes: bytes,
        nonce: int,
        client_hmac_key: bytes
    ) -> Optional[Dict[str, Any]]:
        """Process a client message and return the parsed update if valid."""
        parsed_data = {}

        def store_parsed_data(data):
            nonlocal parsed_data
            parsed_data = data

        result = self.receiver.handle_client_message(
            client_id, payload_bytes, mac_bytes, nonce, client_hmac_key, store_parsed_data
        )

        if result['status'] != 'accepted':
            logger.warning("Message from client %s rejected: %s", client_id, result['reason'])
            return None

        return parsed_data

    def _process_client_updates(
        self,
        client_messages: List[Dict[str, Any]],
        client_keys: Dict[str, bytes]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Process all client messages and extract valid updates."""
        valid_updates = {}
        local_sample_rates = {}
        valid_clients = []

        for msg in client_messages:
            client_id = msg['client_id']

            if client_id not in client_keys:
                logger.warning("No HMAC key for client %s", client_id)
                continue

            parsed_data = self._process_client_message(
                client_id,
                msg['payload_bytes'],
                msg['mac_bytes'],
                msg['nonce'],
                client_keys[client_id]
            )

            if parsed_data is None:
                continue

            # Extract parameters and metadata
            params = parsed_data.get('params')
            if params is None:
                logger.warning("No parameters in message from client %s", client_id)
                continue

            # Extract sample rate information
            batch_size = parsed_data.get('batch_size', 0)
            dataset_size = parsed_data.get('local_dataset_size', batch_size or 1)
            sample_rate = float(batch_size) / float(dataset_size) if dataset_size > 0 else 0.0

            valid_updates[client_id] = params
            local_sample_rates[client_id] = sample_rate
            valid_clients.append(client_id)

        return valid_updates, local_sample_rates, valid_clients

    def start_round(
        self,
        round_idx: int,
        client_messages: List[Dict[str, Any]],
        client_keys: Dict[str, bytes],
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Execute a full federated learning round.

        Args:
            round_idx: Round index
            client_messages: List of client messages
            client_keys: HMAC keys for clients
            noise_multiplier: DP noise multiplier
            max_grad_norm: DP max gradient norm
            weights: Optional client weights

        Returns:
            Round result dictionary
        """
        logger.info("Starting round %d with %d messages", round_idx, len(client_messages))
        start_time = time.time()

        try:
            # Process client messages
            valid_updates, local_sample_rates, valid_clients = self._process_client_updates(
                client_messages, client_keys
            )

            if not valid_updates:
                logger.warning("No valid updates in round %d", round_idx)
                return {'status': 'no_valid_updates'}

            # Calculate participation rate
            participation_rate = len(valid_clients) / max(1, len(client_messages))

            # Apply privacy-preserving techniques (e.g., secure aggregation)
            # This is where Bonawitz protocol would be integrated
            try:
                # For now, we'll just use the updates directly
                # In a real implementation, this would involve secure aggregation
                processed_updates = valid_updates
            except ProtocolAbortError as e:
                logger.error("Protocol abort in round %d: %s", round_idx, e)
                # Record abort in audit log
                self.auditor.record_round(
                    round_idx, valid_clients, participation_rate,
                    local_sample_rates, noise_multiplier, max_grad_norm,
                    extra={'error': 'protocol_abort', 'details': str(e)}
                )
                return {'status': 'protocol_abort', 'reason': str(e)}

            # Aggregate updates
            try:
                aggregated = self.aggregator.aggregate(processed_updates, weights)
            except AggregationError as e:
                logger.error("Aggregation error in round %d: %s", round_idx, e)
                # Record error in audit log
                self.auditor.record_round(
                    round_idx, valid_clients, participation_rate,
                    local_sample_rates, noise_multiplier, max_grad_norm,
                    extra={'error': 'aggregation_failed', 'details': str(e)}
                )
                return {'status': 'aggregation_failed', 'reason': str(e)}

            # Update global state
            for param_name, update in aggregated.items():
                if param_name in self.global_state:
                    self.global_state[param_name] = update

            # Record successful round in audit log
            self.auditor.record_round(
                round_idx, valid_clients, participation_rate,
                local_sample_rates, noise_multiplier, max_grad_norm,
                extra={'processing_time': time.time() - start_time}
            )

            logger.info("Round %d completed successfully in %.2f seconds",
                        round_idx, time.time() - start_time)

            return {
                'status': 'success',
                'clients_processed': len(valid_clients),
                'aggregated_result': aggregated
            }

        except Exception as e:
            logger.exception("Unexpected error in round %d: %s", round_idx, e)
            return {'status': 'error', 'reason': str(e)}

    def get_global_state(self) -> Dict[str, np.ndarray]:
        """Get the current global model state."""
        return self.global_state

    def set_global_state(self, state: Dict[str, np.ndarray]) -> None:
        """Set the global model state."""
        self.global_state = state


# Self-test and demonstration
if __name__ == '__main__':
    print("Running Orchestrator self-test...")

    import tempfile
    import os
    import numpy as np

    # Set up test environment
    os.environ['AUDIT_SECRET'] = 'test_audit_secret_do_not_use_in_production'

    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir)
        db_path = run_dir / 'test.db'

        # Expected parameter shapes
        expected_shapes = {'weights': (2, 2), 'bias': (2,)}

        # Create auditor
        auditor = PrivacyAuditor(run_dir / 'audit')

        # Create orchestrator
        orchestrator = Orchestrator(run_dir, db_path, expected_shapes, auditor)

        # Test client data
        client_keys = {
            'client1': b'test_key_32_bytes_long_______1',
            'client2': b'test_key_32_bytes_long_______2'
        }

        client_messages = []

        # Create client updates
        for i, client_id in enumerate(['client1', 'client2']):
            # Create mock parameters
            params = {
                'weights': np.ones((2, 2)) * (i + 1),
                'bias': np.ones(2) * (i + 1)
            }

            # Create payload with metadata
            payload = {
                'params': params,
                'batch_size': 10,
                'local_dataset_size': 100
            }

            payload_bytes = json.dumps(payload).encode('utf-8')
            nonce = i + 1

            # Compute HMAC
            from FedGNN_advanced.server.receiver import Receiver, PersistedStore

            receiver = Receiver(PersistedStore(db_path))
            mac = receiver.compute_hmac(client_keys[client_id], payload_bytes, nonce)

            client_messages.append({
                'client_id': client_id,
                'payload_bytes': payload_bytes,
                'mac_bytes': mac,
                'nonce': nonce
            })

        # Run a round
        result = orchestrator.start_round(1, client_messages, client_keys)

        assert result['status'] == 'success'
        assert result['clients_processed'] == 2
        print("✓ Orchestrator round test passed")

        # Check global state was updated
        global_state = orchestrator.get_global_state()
        expected_weights = np.array([[1.5, 1.5], [1.5, 1.5]])  # Average of 1 and 2
        expected_bias = np.array([1.5, 1.5])

        assert np.allclose(global_state['weights'], expected_weights)
        assert np.allclose(global_state['bias'], expected_bias)
        print("✓ Global state update test passed")

    print("All Orchestrator tests passed successfully!")
