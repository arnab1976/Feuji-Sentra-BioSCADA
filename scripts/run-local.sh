#!/usr/bin/env bash
# Run the API (and optionally the simulator) without Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

PORT="${PORT:-8000}"
WITH_SIM="${WITH_SIM:-0}"

cleanup() { [ -n "${SIM_PID:-}" ] && kill "$SIM_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$WITH_SIM" = "1" ]; then
  echo "==> Starting telemetry simulator (needs Kafka at ${KAFKA_BOOTSTRAP:-localhost:9092})"
  ( cd services/simulator/src && $PY producer.py ) &
  SIM_PID=$!
  sleep 2
fi

echo "==> API on http://localhost:${PORT}  (docs at /docs)"
echo "==> Open frontend/public/index.html in a browser"
exec $PY -m uvicorn main:app \
  --app-dir services/api/src --host 0.0.0.0 --port "$PORT"
