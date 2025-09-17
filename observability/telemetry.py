# observability/telemetry.py
"""
Setup OpenTelemetry tracing and Prometheus metrics.
- Exposes a function `setup_telemetry(service_name, prometheus_port)`
that configures OTLP exporter (OTLP_COLLECTOR_URL) and starts Prometheus metrics endpoint.
"""
from __future__ import annotations
import os
import logging
from prometheus_client import start_http_server, Counter, Histogram
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger("telemetry")
logger.setLevel(logging.INFO)

# business metrics
UPLOADS = Counter("app_uploads_total", "Total accepted uploads")
MAC_FAILURES = Counter("app_mac_failures_total", "Total MAC failures")
LATENCY = Histogram("app_upload_latency_seconds", "upload latency seconds")


def setup_telemetry(service_name: str = "fedgnn-service", prometheus_port: int = 8000):
    # Prometheus
    start_http_server(prometheus_port)
    logger.info("Prometheus metrics listening on port %d", prometheus_port)

    # OpenTelemetry
    otlp_collector = os.getenv("OTLP_COLLECTOR_URL")
    if otlp_collector:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        exporter = OTLPSpanExporter(endpoint=otlp_collector, insecure=True if os.getenv("OTLP_INSECURE","true")=="true" else False)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("OpenTelemetry exporter configured to %s", otlp_collector)
    else:
        logger.info("No OTLP_COLLECTOR_URL set; tracing disabled")
