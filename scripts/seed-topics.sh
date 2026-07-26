#!/usr/bin/env bash
# Create the Kafka topics with sensible partitioning.
# Works against Redpanda (rpk) or Kafka (kafka-topics.sh).
set -euo pipefail

BROKER="${KAFKA_BOOTSTRAP:-localhost:9092}"
TOPICS=("scada.telemetry:5" "breach.events:3" "features.windowed:3" "remedy.actions:1" "audit.trail:1")

if command -v rpk >/dev/null; then
  for t in "${TOPICS[@]}"; do
    name="${t%%:*}"; parts="${t##*:}"
    rpk topic create "$name" -p "$parts" -r 1 --brokers "$BROKER" 2>/dev/null \
      && echo "created $name ($parts partitions)" || echo "exists  $name"
  done
elif command -v kafka-topics.sh >/dev/null; then
  for t in "${TOPICS[@]}"; do
    name="${t%%:*}"; parts="${t##*:}"
    kafka-topics.sh --create --if-not-exists --topic "$name" \
      --partitions "$parts" --replication-factor 1 \
      --bootstrap-server "$BROKER" && echo "ok $name"
  done
else
  echo "Neither 'rpk' nor 'kafka-topics.sh' found."
  echo "Inside Docker:  docker compose exec redpanda rpk topic create scada.telemetry -p 5"
  exit 1
fi
