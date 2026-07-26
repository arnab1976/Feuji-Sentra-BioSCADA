# BioSCADA AI — common tasks.
# Everything here uses free / open-source tooling only.

PY      ?= python3
PORT    ?= 8000
SAMPLES ?= 30000
PARAM   ?= ph

.DEFAULT_GOAL := help
.PHONY: help install train kb test api demo docker-up docker-down docker-full \
        flink-submit topics lint clean reset

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install:  ## Install all Python dependencies
	$(PY) -m pip install -q \
	  -r services/ml/requirements.txt \
	  -r services/rag/requirements.txt \
	  -r services/api/requirements.txt \
	  -r requirements-dev.txt

train:  ## Phase 2 — train the five per-parameter models
	$(PY) services/ml/src/train.py --generate $(SAMPLES)

kb:  ## Build the RAG knowledge base (structured + unstructured)
	$(PY) services/rag/src/knowledge_base.py --no-qdrant

test:  ## Run the full test suite
	$(PY) -m pytest tests/ -v

api:  ## Run the API locally on $(PORT)
	$(PY) -m uvicorn main:app --app-dir services/api/src \
	  --host 0.0.0.0 --port $(PORT) --reload

demo:  ## Drive every phase through the running API (PARAM=ph|temp|press|cond|hum)
	./scripts/demo.sh $(PARAM)

setup: install train kb test  ## Full local setup from scratch
	@echo "\nSetup complete. Run 'make api', then open frontend/public/index.html"

docker-up:  ## Start the core stack (broker, simulator, api, frontend)
	cd infra && docker compose up -d redpanda simulator api frontend
	@echo "Portal: http://localhost:3000   API: http://localhost:8000/docs"

docker-full:  ## Start everything (adds Keycloak, Kong, OPA, MinIO, MLflow, Grafana, Ollama)
	cd infra && docker compose --profile full up -d

docker-down:  ## Stop and remove the stack
	cd infra && docker compose --profile full down -v

topics:  ## Create Kafka topics with correct partitioning
	./scripts/seed-topics.sh

flink-submit:  ## Submit the Flink breach-detection job
	cd infra && docker compose exec flink-jobmanager \
	  flink run -py /opt/flink-jobs/src/breach_detection_job.py

lint:  ## Static checks
	-$(PY) -m ruff check services/ tests/
	-$(PY) -m compileall -q services/ tests/

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

reset: clean  ## Also drop trained models and the built index
	rm -rf services/ml/models/*.joblib services/ml/models/metrics.json
	rm -rf services/rag/index
