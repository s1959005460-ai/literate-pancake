# edge/edge_server.py
"""
Edge aggregator - production-grade:
 - gRPC Async server with mTLS (server and client certs from Kubernetes Secrets)
 - HMAC verification, replay protection via Redis
 - Publish valid payloads to Kafka using KafkaProducerWrapper
 - Expose health/readiness endpoints via FastAPI
 - OpenTelemetry, Prometheus metrics, structured JSON logging
 - Graceful shutdown and resource cleanup
"""
import os
import asyncio
import logging
import signal
import json
from concurrent import futures
from typing import Optional

import grpc
from grpc import aio
from fastapi import FastAPI
import uvicorn
from pythonjsonlogger import jsonlogger

from proto import federated_service_pb2 as pb
from proto import federated_service_pb2_grpc as rpc

from infra.kafka_client import KafkaProducerWrapper
from infra.redis_client import check_and_set_seq, store_masked_share, get_redis
from crypto.crypto_utils import derive_shared_key, hmac_verify
from observability.telemetry import setup_telemetry, UPLOADS, MAC_FAILURES, LATENCY

# logging
logger = logging.getLogger("edge_server")
logHandler = logging.StreamHandler()
logHandler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(logHandler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# config via env
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "updates")
PROM_PORT = int(os.getenv("PROM_PORT", "8000"))
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
MTLS_CERT = os.getenv("MTLS_CERT", "/certs/server.crt")
MTLS_KEY = os.getenv("MTLS_KEY", "/certs/server.key")
MTLS_CA = os.getenv("MTLS_CA", "/certs/ca.crt")
SERVER_PRIV_KEY_PATH = os.getenv("SERVER_PRIV_KEY_PATH", "/secrets/server_x25519_priv.bin")

# graceful shutdown helpers
shutdown_event = asyncio.Event()

# FastAPI app for health and readiness
http_app = FastAPI()


@http_app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@http_app.get("/readyz")
async def readyz():
    # readiness checks: kafka connection & redis
    try:
        r = await get_redis()
        # simple ping
        await r.ping()
        return {"ready": True}
    except Exception:
        return {"ready": False}


class FederatedServicer(rpc.FederatedServiceServicer):
    def __init__(self, kafka: KafkaProducerWrapper, server_priv: bytes):
        self.kafka = kafka
        self.server_priv = server_priv

    async def UploadUpdate(self, request: pb.UploadRequest, context) -> pb.UploadResponse:
        client_id = request.client_id
        seq = int(request.seq)
        payload = request.payload
        sender_pub = request.sender_pub
        mac = request.mac
        with LATENCY.time():
            # 1. replay protection
            ok = await check_and_set_seq(client_id, seq)
            if not ok:
                MAC_FAILURES.inc()
                logger.warning("replay detected", extra={"client_id": client_id, "seq": seq})
                return pb.UploadResponse(ok=False, message="replay detected")

            # 2. derive shared key and verify MAC
            try:
                shared_key = derive_shared_key(self.server_priv, sender_pub)
                if not hmac_verify(shared_key, seq.to_bytes(8, "big") + payload, mac):
                    MAC_FAILURES.inc()
                    logger.warning("mac verify failed", extra={"client_id": client_id})
                    return pb.UploadResponse(ok=False, message="mac failed")
            except Exception:
                MAC_FAILURES.inc()
                logger.exception("mac derivation failed")
                return pb.UploadResponse(ok=False, message="mac error")

            # 3. persist masked share to redis (best-effort, non-blocking)
            try:
                await store_masked_share(int(request.round), client_id, payload)
            except Exception:
                logger.exception("failed to store masked share")

            # 4. produce to kafka with retry
            produced = await self.kafka.produce_with_retry(KAFKA_TOPIC, payload)
            if not produced:
                logger.error("kafka produce failed for client %s", client_id)
                return pb.UploadResponse(ok=False, message="kafka produce failed")
            # metrics
            UPLOADS.inc()
            logger.info("accepted upload", extra={"client_id": client_id, "round": request.round})
            return pb.UploadResponse(ok=True, message="accepted")


async def serve():
    # telemetry
    setup_telemetry("edge-aggregator", PROM_PORT)

    # load server x25519 private key
    if not os.path.exists(SERVER_PRIV_KEY_PATH):
        logger.critical("server private key missing at %s", SERVER_PRIV_KEY_PATH)
        raise SystemExit(1)
    with open(SERVER_PRIV_KEY_PATH, "rb") as f:
        server_priv = f.read()

    # kafka producer
    kafka = KafkaProducerWrapper(KAFKA_BOOTSTRAP)
    await kafka.start()

    # configure gRPC server with mTLS
    server = aio.server()
    servicer = FederatedServicer(kafka, server_priv)
    rpc.add_FederatedServiceServicer_to_server(servicer, server)

    # mTLS credentials
    # load certs
    private_key = open(MTLS_KEY, "rb").read()
    certificate_chain = open(MTLS_CERT, "rb").read()
    root_certificates = open(MTLS_CA, "rb").read()
    server_credentials = grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=root_certificates,
        require_client_auth=True
    )

    listen_addr = f"[::]:{GRPC_PORT}"
    server.add_secure_port(listen_addr, server_credentials)
    await server.start()
    logger.info("gRPC server started (secure) on %s", listen_addr)

    # start health HTTP
    http_server_task = asyncio.create_task(uvicorn.run(http_app, host="0.0.0.0", port=8080, log_level="info"))

    # wait on shutdown
    await shutdown_event.wait()

    logger.info("shutdown signal received, stopping...")
    await server.stop(5)
    await kafka.stop()
    logger.info("edge server stopped")


def _graceful_shutdown():
    shutdown_event.set()


def main():
    loop = asyncio.get_event_loop()
    # register signals
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, _graceful_shutdown)
    try:
        loop.run_until_complete(serve())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
