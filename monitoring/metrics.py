# -*- coding: utf-8 -*-
"""
Metrics collection for FedGNN_advanced.
Integrated with Prometheus for monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Counters
hmac_failures_total = Counter(
    'fedgnn_hmac_failures_total',
    'Total number of HMAC verification failures',
    ['client_id']
)

replay_attempts_total = Counter(
    'fedgnn_replay_attempts_total',
    'Total number of replay attempts detected',
    ['client_id']
)

shamir_failures_total = Counter(
    'fedgnn_shamir_failures_total',
    'Total number of Shamir reconstruction failures'
)

privacy_audit_entries_total = Counter(
    'fedgnn_privacy_audit_entries_total',
    'Total number of privacy audit entries written'
)

# Gauges
dp_epsilon = Gauge(
    'fedgnn_dp_epsilon',
    'Current DP epsilon value',
    ['round']
)

clients_processed = Gauge(
    'fedgnn_clients_processed',
    'Number of clients processed in current round'
)

# Histograms
round_duration_seconds = Histogram(
    'fedgnn_round_duration_seconds',
    'Time taken to process a round',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

compression_ratio = Histogram(
    'fedgnn_compression_ratio',
    'Compression ratio achieved',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")


# Example usage in other modules:
# from monitoring.metrics import hmac_failures_total
# hmac_failures_total.labels(client_id=client_id).inc()
