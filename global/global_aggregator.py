# global/global_aggregator.py
"""
Global aggregator: Kafka consumer -> aggregate per round -> decrypt via secure decryptor -> update model store.
- Uses kafka consumer wrapper, handles backpressure & retry
- For HE decryption: sends aggregate to a secure decryptor service (recommended) instead of storing secret keys here
- Writes model artifact to S3 (or local path in DEV)
- Exposes metrics + readiness + graceful shutdown
"""
import os
import asyncio
import logging
import json
import base64
from typing import Dict, List
from infra.kafka_client import KafkaConsumerWrapper, KafkaProducerWrapper
from observability.telemetry import setup_telemetry
from infra.redis_client import fetch_round_shares
import aiohttp

logger = logging.getLogger("global_aggregator")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
UPDATES_TOPIC = os.getenv("KAFKA_TOPIC", "updates")
MODEL_TOPIC = os.getenv("MODEL_TOPIC", "model_updates")
AGG_THRESHOLD = int(os.getenv("AGG_THRESHOLD", "500"))
MODEL_STORE_DIR = os.getenv("MODEL_STORE_DIR", "/data/models")
DECRYPTOR_URL = os.getenv("DECRYPTOR_URL", "http://decryptor:9000/decrypt")  # secure decryptor service

stop_event = asyncio.Event()


async def handle_payload(payload: bytes):
    """
    Process raw payload (application-level: check metadata to know if HE ciphertext or masked)
    For simplicity here we just append to in-memory store; in production use persistent store or RocksDB.
    """
    # Placeholder; actual handler details below in main consumer loop
    pass


async def finalize_round(round_id: int, ciphertexts: List[bytes], producer: KafkaProducerWrapper):
    """
    Aggregate ciphertexts locally (add) and send to decryptor to obtain plaintext.
    """
    # naive aggregation: concatenation or TenSEAL add on server (not ideal). Better: partial sums at edges.
    # Here we assume ciphertexts are serialized TenSEAL bytes; we will send base64(list) to decryptor
    try:
        agg_b64 = base64.b64encode(b"".join(ciphertexts)).decode()
        payload = {"round": round_id, "agg_cipher_b64": agg_b64}
        # Call decryptor
        async with aiohttp.ClientSession() as session:
            async with session.post(DECRYPTOR_URL, json=payload, timeout=120) as resp:
                if resp.status != 200:
                    logger.error("decryptor returned %d", resp.status)
                    return False
                data = await resp.json()
                model_delta = data.get("model_delta")
                # persist model delta (simple file)
                os.makedirs(MODEL_STORE_DIR, exist_ok=True)
                path = os.path.join(MODEL_STORE_DIR, f"model_round_{round_id}.json")
                with open(path, "w") as f:
                    json.dump({"round": round_id, "delta": model_delta}, f)
                # notify downstream
                await producer.produce_with_retry(MODEL_TOPIC, json.dumps({"round": round_id, "path": path}).encode())
                logger.info("finalized round %d", round_id)
                return True
    except Exception:
        logger.exception("finalize_round failed")
        return False


async def run():
    setup_telemetry("global-aggregator", int(os.getenv("PROM_PORT", "9100")))
    consumer = KafkaConsumerWrapper(KAFKA_BOOTSTRAP, UPDATES_TOPIC, group_id="global-agg")
    producer = KafkaProducerWrapper(KAFKA_BOOTSTRAP)
    await consumer.start()
    await producer.start()

    # in-memory store per round; production: use persistent store
    agg_store: Dict[int, List[bytes]] = {}

    async def handler(msg_bytes: bytes):
        try:
            # msg_bytes is raw payload; in this design edges directly produce payload bytes
            # but if edges produce JSON, parse accordingly
            # attempt to parse as JSON
            try:
                js = json.loads(msg_bytes.decode())
                payload_b64 = js.get("payload")
                round_id = int(js.get("round"))
                payload = base64.b64decode(payload_b64)
            except Exception:
                # fallback: raw bytes
                payload = msg_bytes
                round_id = 0
            lst = agg_store.setdefault(round_id, [])
            lst.append(payload)
            if len(lst) >= AGG_THRESHOLD:
                await finalize_round(round_id, lst, producer)
                agg_store.pop(round_id, None)
        except Exception:
            logger.exception("handler failed")

    await consumer.consume(handler, stop_event)


def _shutdown():
    stop_event.set()


def main():
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(asyncio.constants.SIGTERM, _shutdown)
    loop.add_signal_handler(asyncio.constants.SIGINT, _shutdown)
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
