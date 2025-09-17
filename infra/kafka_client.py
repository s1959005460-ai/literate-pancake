# infra/kafka_client.py
"""
Kafka helper with aiokafka, retry, backoff, and connection pool management.
- produce_with_retry() with exponential backoff
- consumer creation helper with graceful shutdown
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Callable, Any, Optional
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = logging.getLogger("kafka_client")
logger.setLevel(logging.INFO)


class KafkaProducerWrapper:
    def __init__(self, bootstrap_servers: str, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._loop = loop or asyncio.get_event_loop()
        self._producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers, loop=self._loop)
        self._started = False

    async def start(self):
        try:
            await self._producer.start()
            self._started = True
            logger.info("Kafka producer started")
        except KafkaError:
            logger.exception("Failed to start Kafka producer")
            raise

    async def stop(self):
        if self._started:
            await self._producer.stop()
            self._started = False
            logger.info("Kafka producer stopped")

    async def produce_with_retry(self, topic: str, value: bytes, key: Optional[bytes] = None, max_retries: int = 5):
        attempt = 0
        backoff = 0.5
        while attempt <= max_retries:
            try:
                await self._producer.send_and_wait(topic, value, key=key)
                return True
            except KafkaError:
                logger.exception("kafka send failed, attempt %d", attempt)
                attempt += 1
                await asyncio.sleep(backoff)
                backoff *= 2
        logger.error("kafka send failed after %d attempts", max_retries)
        return False


class KafkaConsumerWrapper:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._loop = loop or asyncio.get_event_loop()
        self._topic = topic
        self._consumer = AIOKafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group_id, loop=self._loop)
        self._stopped = False

    async def start(self):
        await self._consumer.start()
        logger.info("Kafka consumer started for topic %s", self._topic)

    async def stop(self):
        await self._consumer.stop()
        self._stopped = True
        logger.info("Kafka consumer stopped")

    async def consume(self, handler: Callable[[bytes], Any], stop_event: asyncio.Event):
        try:
            while not stop_event.is_set():
                try:
                    msg = await self._consumer.getone()
                    await handler(msg.value)
                except asyncio.TimeoutError:
                    continue
                except KafkaError:
                    logger.exception("kafka consume error")
                    await asyncio.sleep(1)
        finally:
            await self.stop()
