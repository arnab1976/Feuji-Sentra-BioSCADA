# BioSCADA AI

**Automating SCADA threshold monitoring with an agentic AI ecosystem.**
Transforms a predictive-maintenance model into a RAG-based multi-agent
architecture: each critical SCADA parameter becomes an autonomous agent that
diagnoses a breach, retrieves the approved remedy from a governed knowledge
base, and executes or escalates it under zero-trust, GxP-compliant control.

Every component in this repository is **free / open source**.

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
  PHASE 0                 │  Sensors / PLC / OPC UA  (simulated here)   │
  Ingestion               └───────────────────┬─────────────────────────┘
                                              │  JSON, keyed by parameter
                                  ┌───────────▼───────────┐
                                  │  Kafka / Redpanda     │  scada.telemetry
                                  │  5 partitions         │
                                  └───────────┬───────────┘
                                              │
  PHASE 1                 ┌───────────────────▼─────────────────────────┐
  Stream processing       │  Apache Flink + Flink SQL                   │
  (RELATIONAL-PLAN        │  ① Source ② Window/Watermark                │
   DECOMPOSITION)         │  ③ Feature aggregation (independent vars)   │
                          │  ④ PDM_SCORE UDF  ⑤ Filter  ⑥ Sink          │
                          └───────────────────┬─────────────────────────┘
                                              │  breach.events
                                              │  "breach + why" (STRUCTURED)
  ══════════════════════ DECOMPOSITION BOUNDARY ══════════════════════════
                                              │
  PHASE 2                 ┌───────────────────▼─────────────────────────┐
  Predictive modelling    │  Per-parameter models (GBM/RF/SVM/ANN)      │
                          │  scikit-learn · XGBoost · SHAP · MLflow     │
                          └───────────────────┬─────────────────────────┘
                                              │
  PHASE 3                 ┌───────────────────▼─────────────────────────┐
  Agentic RAG             │  Orchestrator ──RBAC──> ParameterAgent ×5   │
  (SEMANTIC               │    ① plan sub-questions                     │
   DECOMPOSITION)         │    ② multi-retrieval fan-out                │
                          │    ③ RRF fusion + MMR re-rank               │
                          │    ④ cited synthesis (local LLM)            │
                          └───────────────────┬─────────────────────────┘
                                              │  RemedyCard
  PHASE 4                 ┌───────────────────▼─────────────────────────┐
  Governance              │  Keycloak (JWT) → Kong → OPA (PDP)          │
                          │  → Human Escalation Agent → e-signature     │
                          │  → hash-chained WORM audit (Postgres)       │
                          └─────────────────────────────────────────────┘
```

**The boundary matters.** Flink decomposes the *computation* into an operator
graph and emits a structured event. The LLM then decomposes the *question*
into sub-queries. Embedding, retrieval and reasoning never live in the stream
engine.

---

## Repository layout

```
bioscada-ai/
├── data/                          # knowledge corpus (structured + unstructured)
│   ├── sop/                       #   SOPs               (unstructured, .md)
│   ├── oem/                       #   OEM manuals        (unstructured, .md)
│   ├── capa/                      #   CAPA + maintenance (structured, .csv)
│   └── batch/                     #   batch records      (structured, .json)
│
├── services/
│   ├── simulator/                 # PHASE 0 — SCADA telemetry → Kafka
│   │   └── src/
│   │       ├── parameters.py      #   canonical top-5 definitions (shared)
│   │       └── producer.py        #   physics sim + breach injection API
│   │
│   ├── flink-jobs/                # PHASE 1 — stream decomposition
│   │   ├── sql/01_breach_detection.sql
│   │   └── src/breach_detection_job.py   # PyFlink + PDM_SCORE / TOP_DRIVER UDFs
│   │
│   ├── ml/                        # PHASE 2 — predictive modelling
│   │   ├── src/train.py           #   per-parameter models + SHAP + MLflow
│   │   └── models/                #   trained artifacts (.joblib, metrics.json)
│   │
│   ├── rag/                       # knowledge base
│   │   └── src/knowledge_base.py  #   ingest, verbalize, embed, hybrid search
│   │
│   ├── agents/                    # PHASE 3 — multi-agent layer
│   │   └── src/agents.py          #   Orchestrator, ParameterAgent, Escalation
│   │
│   └── api/                       # integration tier
│       └── src/main.py            #   FastAPI — exposes every phase
│
├── frontend/public/index.html     # intelligence portal (single file, no build)
│
├── infra/
│   ├── docker-compose.yml         # the whole free stack
│   ├── docker/                    # Dockerfiles
│   └── config/
│       ├── opa/bioscada.rego      #   PDP policy + tests
│       ├── postgres-init.sql      #   hash-chained WORM audit schema
│       ├── kong.yml, nginx.conf, prometheus/
│
├── tests/test_pipeline.py         # 52 tests across all phases
└── scripts/                       # quickstart helpers
```

---

## Quickstart

### Option A — everything in Docker

```bash
cd infra
docker compose up -d redpanda simulator api frontend
# open http://localhost:3000
```

Add the optional tier (Keycloak, Kong, MinIO, MLflow, Grafana, Ollama):

```bash
docker compose --profile full up -d
docker compose exec ollama ollama pull llama3.1:8b   # local LLM, no data egress
```

### Option B — run locally without Docker

```bash
# 1. train the models (synthesizes data if you have no historian export)
pip install -r services/ml/requirements.txt
python services/ml/src/train.py --generate 30000

# 2. build the knowledge base
pip install -r services/rag/requirements.txt
python services/rag/src/knowledge_base.py --no-qdrant

# 3. start the API
pip install -r services/api/requirements.txt
uvicorn main:app --app-dir services/api/src --port 8000

# 4. open the portal
open frontend/public/index.html      # talks to http://localhost:8000
```

The portal degrades gracefully: without Kafka it still serves prediction,
RAG and governance; without an LLM it produces grounded template remedies
lifted verbatim from the retrieved SOPs (still fully cited).

### Submit the Flink job

```bash
docker compose exec flink-jobmanager \
  flink run -py /opt/flink-jobs/src/breach_detection_job.py
# or, pure SQL:
docker compose exec flink-jobmanager \
  ./bin/sql-client.sh -f /opt/flink-jobs/sql/01_breach_detection.sql
```

---

## The five parameters

| Parameter | Asset | Control band | Model | Independent variables (PdM features) |
|---|---|---|---|---|
| Reactor Temperature | BR-12 | 36.5–37.5 °C | GBM | coolant flow, jacket temp, steam valve, HX ΔP, agitator torque |
| pH | BR-12 | 6.8–7.2 | Random Forest | dose rate, CO₂ accumulation, agitator speed, probe drift, DO |
| Differential Pressure | FIL-07 | 100–110 kPa | GBM | filter ΔP, exhaust flow, pump speed, valve position, seal index |
| Conductivity (WFI) | WFI-02 | 700–900 µS/cm | SVM | water flow, resin ΔP, regen cycles, TOC, feed composition |
| Humidity (Cleanroom) | CR-A1 | 40–55 % | ANN | HVAC fan, coil temp, HEPA ΔP, outdoor RH, door count |

Plus **one Human Escalation Agent** (QA / Production) that receives every
action the policy flags as high-risk.

**Threshold breach** = the value leaves its *control* band. Three nested
zones: control (normal) → alarm (agent acts) → trip (QA e-signature required).

---

## Governance rules (encoded in OPA + tested)

| Condition | Decision |
|---|---|
| Agent lacks `agent.execute.<param>` role | **DENY** |
| Parameter is pH (SOP-PH-009) | **REQUIRE_ESIGN** |
| Zone is `trip` | **REQUIRE_ESIGN** |
| P(breach) ≥ 0.80 | **REQUIRE_ESIGN** |
| Otherwise | **ALLOW** (+ operator signature for traceability) |

Every decision, signature and control action is written to an append-only,
hash-chained audit table. `UPDATE` and `DELETE` are blocked by trigger.

---

## Tech stack — all free

| Layer | Choice | Licence |
|---|---|---|
| Broker | Redpanda / Apache Kafka | BSL / Apache-2.0 |
| Stream processing | Apache Flink + Flink SQL | Apache-2.0 |
| Storage | Postgres, MinIO, Apache Iceberg/Hudi | PostgreSQL / AGPL / Apache-2.0 |
| Modelling | scikit-learn, XGBoost, SHAP, MLflow | BSD / Apache-2.0 |
| Embeddings | sentence-transformers (BGE) | Apache-2.0 |
| Vector store | Qdrant | Apache-2.0 |
| Lexical search | rank-bm25 | Apache-2.0 |
| Agents | plain Python (Haystack adapter included) | Apache-2.0 |
| LLM | Llama 3.1 / Mistral / Qwen via Ollama | community / Apache-2.0 |
| Identity | Keycloak | Apache-2.0 |
| Gateway | Kong OSS | Apache-2.0 |
| Policy | Open Policy Agent | Apache-2.0 |
| Observability | Prometheus + Grafana | Apache-2.0 / AGPL-3.0 |
| API | FastAPI + Uvicorn | MIT / BSD |

No LangChain or LangGraph required. The orchestration is plain Python;
a Haystack adapter scaffold is provided in `services/agents/src/agents.py`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

52 tests covering parameter bands, simulator physics and feature
correlation, Flink UDF scoring (bounded, monotonic, volatility-sensitive),
knowledge-base ingestion of both data families, agent decomposition and
citation, the GxP decision table, and audit hash-chain tamper detection.

---

## Two honest caveats

**"Free" is not "no cost."** Self-hosting Kafka, Flink, Qdrant and a
GPU-served LLM carries real infrastructure and operations cost. Licensing is
zero; hardware and people are not.

**GxP validation is the hidden line item.** In a regulated environment the
expensive part is computer-system validation (IQ/OQ/PQ), change control and
audit readiness. Open-source components are perfectly acceptable to
regulators — but you own the validation evidence a commercial vendor might
otherwise supply. The hash-chained audit schema and the tested policy table
here are inputs to that evidence, not a substitute for it.
