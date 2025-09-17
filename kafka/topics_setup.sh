#!/usr/bin/env bash
# File: kafka/topics_setup.sh
# Usage: KAFKA_BROKER=localhost:9092 ./kafka/topics_setup.sh

BROKER=${KAFKA_BROKER:-localhost:9092}
TOPIC_UPDATES=${KAFKA_TOPIC_UPDATES:-updates}
TOPIC_MODEL=${KAFKA_TOPIC_MODEL:-model_updates}

# Create 'updates' topic
kafka-topics --bootstrap-server ${BROKER} --create --topic ${TOPIC_UPDATES} --partitions 50 --replication-factor 1 || true
kafka-topics --bootstrap-server ${BROKER} --create --topic ${TOPIC_MODEL} --partitions 10 --replication-factor 1 || true

echo "Topics created (or already exist): ${TOPIC_UPDATES}, ${TOPIC_MODEL}"
