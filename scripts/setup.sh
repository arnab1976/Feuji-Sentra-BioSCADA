#!/usr/bin/env bash
# BioSCADA AI — local setup (no Docker required).
# Installs dependencies, trains models, builds the knowledge base.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
PIP_FLAGS="${PIP_FLAGS:-}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$1"; }

say "1/4  Installing Python dependencies"
$PY -m pip install $PIP_FLAGS -q \
  -r services/ml/requirements.txt \
  -r services/rag/requirements.txt \
  -r services/api/requirements.txt \
  || warn "some optional packages failed — the pipeline degrades gracefully"

say "2/4  Training Phase-2 models"
if [ -f services/ml/models/metrics.json ] && [ "${FORCE_TRAIN:-0}" != "1" ]; then
  warn "models already present (set FORCE_TRAIN=1 to retrain)"
else
  $PY services/ml/src/train.py --generate "${SAMPLES:-30000}"
fi

say "3/4  Building the RAG knowledge base"
$PY services/rag/src/knowledge_base.py --no-qdrant

say "4/4  Running the test suite"
$PY -m pytest tests/ -q || warn "some tests failed — see output above"

cat <<'DONE'

Setup complete.

  Start the API:      ./scripts/run-local.sh
  Open the portal:    frontend/public/index.html
  Or use Docker:      cd infra && docker compose up -d

DONE
