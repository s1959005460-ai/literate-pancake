"""
Lightweight monitoring helpers. Exposes log_metrics to collect run-time metrics.
In production this should forward to Prometheus/Influx/MLFlow/TensorBoard etc.
"""

from typing import Dict, Any
from . import logger


def log_metrics(step: int, metrics: Dict[str, Any]) -> None:
    logger.secure_log("info", "metrics", step=step, metrics=metrics)
