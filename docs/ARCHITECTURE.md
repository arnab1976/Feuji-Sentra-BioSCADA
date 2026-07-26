# BioSCADA AI — Architecture

This document explains *why* the system is built the way it is. The README
covers what exists and how to run it; this covers the design decisions a
reviewer or auditor is likely to question.

---

## 1. The problem in one paragraph

Predictive-maintenance models tell you a parameter *will* breach. They do not
tell the operator *what to do now*. The corrective knowledge — SOPs, CAPA
history, OEM manuals, maintenance logs — is scattered across systems, so
response is slow, varies by whoever is on shift, and is hard to defend in an
audit. BioSCADA AI closes that loop: it fuses live telemetry with a governed
knowledge base and lets a per-parameter agent produce a cited, policy-checked
remedy that a human signs before anything moves.

---

## 2. The two decompositions

This is the single most important idea in the architecture, and the one most
often conflated.

### 2.1 Relational-plan decomposition (Apache Flink)

Flink SQL takes **one declarative statement** and decomposes it into a
distributed graph of streaming operators:

```
Source → Watermark/Window → Aggregate → Scalar UDF → Filter → Sink
```

This is a **compute** decomposition. It is deterministic, typed, and has
nothing to do with language. Its output is a *structured* record:

```json
{
  "event_id": "evt-temp-1719...",
  "param": "temp", "asset": "BR-12",
  "v_avg": 38.21, "v_std": 0.09, "v_delta": 0.43,
  "zone": "alarm", "p_breach": 0.87,
  "top_driver": "coolant_flow_rate"
}
```

### 2.2 Semantic decomposition (the LLM)

The agent takes that structured record and decomposes the **question**:

```
"What do I do about this?"
    ├── What is the approved SOP for temperature above band on BR-12?
    ├── What prior CAPA records exist, and which actions were effective?
    └── What do the OEM manual and maintenance history say about coolant flow?
```

Each sub-question is embedded and retrieved separately, then fused and
re-ranked. This is a **meaning** decomposition.

### 2.3 Why the boundary is enforced

Flink never embeds text, never calls a vector store, never invokes an LLM.
Three reasons:

1. **Latency and backpressure.** An LLM call inside a windowed aggregation
   would stall the operator and build unbounded state.
2. **Determinism.** Stream processing must be replayable. LLM output is not.
   Keeping generation out of the stream means a replayed window produces
   byte-identical results.
3. **Validation.** In GxP terms, the deterministic path (telemetry → features
   → score → event) can be validated conventionally. The probabilistic path
   (retrieval → generation) is validated differently — as a decision-support
   tool whose output a human signs. Mixing them makes both harder to validate.

If someone tells you Flink SQL is "processing the RAG query," that is an
overstatement. Flink assembles the structured payload; the agent decomposes
the question.

---

## 3. Phase 0 — Ingestion

**`services/simulator/src/producer.py`**

The simulator stands in for a historian/OPC-UA feed. It is not decorative: it
generates physically-plausible data with **correlated independent variables**,
which is what makes the Phase-2 model learnable rather than fitting noise.

Each record carries both the dependent variable and its five independent
variables:

```json
{"param":"temp","value":38.21,"zone":"alarm",
 "features":{"coolant_flow_rate":103.1,"heat_exchanger_dp":31.9, ...}}
```

The correlation is signed and physical — as temperature rises, coolant flow
*falls* and heat-exchanger ΔP *rises*. A test asserts this
(`test_features_correlate_with_deviation`), because if the correlation were
accidentally inverted the model would learn the wrong physics and nobody
would notice from the AUC alone.

**Topic keying.** Records are keyed by `param`, so each parameter lands in a
stable partition. This gives per-parameter ordering, which the windowed
aggregation depends on.

**Replacing the simulator.** Point a Kafka Connect source, an OPC-UA bridge,
or a historian export at `scada.telemetry` with the same schema. Nothing
downstream changes.

---

## 4. Phase 1 — Stream processing

**`services/flink-jobs/sql/01_breach_detection.sql`**
**`services/flink-jobs/src/breach_detection_job.py`**

### 4.1 Windowing

10-second tumbling windows on **event time**, with a 5-second watermark for
out-of-orderness. Event time (not processing time) matters because a network
hiccup must not shift a reading into the wrong window and corrupt the
aggregate.

### 4.2 The features are computed here, not in the model

`v_avg`, `v_std`, `v_delta` are computed by Flink. The offline training code
in `services/ml/src/train.py` computes **the same three aggregates the same
way** (`build_windowed_features`). This is deliberate: it is the standard
guard against training/serving skew. If you change the aggregation in Flink,
you must change it in `train.py`, or the model sees a different feature
distribution in production than it was trained on.

### 4.3 In-stream scoring

`PDM_SCORE` is a Python UDF that loads the trained `.joblib` model and calls
`predict_proba`. If no model artifact is present it falls back to a
deterministic analytic score:

```
deviation  = |v_avg − control_centre| / half_alarm_width
volatility = min(v_std / half_alarm_width, 0.5)
score      = clamp((deviation − 0.30) / 0.85 + volatility × 0.35, 0, 1)
```

The fallback is not a toy — it is monotonic, bounded, volatility-sensitive
and explainable, which makes it a useful sanity baseline *and* keeps the
pipeline runnable before any model exists. Tests assert all three properties.

### 4.4 The HAVING clause is the filter

```sql
HAVING PDM_SCORE(...) > 0.30 OR MAX(zone) <> 'control'
```

In-band, low-risk windows are dropped here. Only breach candidates reach the
agent layer. This is what keeps the expensive downstream path (retrieval +
generation) from running on every window.

---

## 5. Phase 2 — Predictive modelling

**`services/ml/src/train.py`**

### 5.1 Label definition

The label is **forward-looking**:

> 1 if the parameter leaves its control band at any point in the next
> `horizon` samples, else 0.

The `shift(-1)` before the rolling max is important — it prevents the current
row from seeing its own breach, which would leak the answer and produce an
AUC near 1.0 that collapses in production.

### 5.2 Model per parameter

| Parameter | Model | Rationale |
|---|---|---|
| temp, press | Gradient boosting | Non-linear thresholds, tabular, strong default |
| ph | Random Forest | Robust to the probe-drift noise that produces false positives |
| cond | SVM (RBF, calibrated) | Smooth decision boundary on a slow-moving signal |
| hum | MLP | Interaction between HVAC variables and outdoor conditions |

Measured on synthetic-but-correlated data: AUC 0.83–0.94, positive rate
19–32%. The pH model is the weakest (0.83), which is honest — pH has the
noisiest driver set, and that is exactly why pH is also the parameter where
policy forbids autonomous action.

### 5.3 Explainability

SHAP where available, native `feature_importances_` otherwise, permutation
importance as a last resort. The top driver becomes the "why" carried into
the RAG prompt, so retrieval is grounded in the actual cause rather than a
generic query.

---

## 6. Knowledge base — consolidating two data families

**`services/rag/src/knowledge_base.py`**

### 6.1 The verbalization decision

Structured rows embed *badly*. A raw CSV row —

```
CAPA-2231,temp,BR-12,2024-04-18,Reactor temperature exceeded...
```

— has almost no linguistic signal for a sentence embedder. So every
structured row is **verbalized** into prose before embedding:

> "CAPA record CAPA-2231 for temp on asset BR-12, opened 2024-04-18.
> Problem: Reactor temperature exceeded 38.0 degC for 22 minutes...
> Corrective action taken: Cleaned heat-exchanger plate pack..."

The original row is retained in `chunk.extra["row"]` so the UI can display
the exact record. One hybrid retriever then serves both families.

### 6.2 Hybrid retrieval

- **Dense**: `BAAI/bge-small-en-v1.5` via sentence-transformers
- **Lexical**: BM25 (`rank_bm25`) — catches exact identifiers like
  `SOP-THM-014` and `CW-3` that embeddings blur
- **Fusion**: `alpha × dense + (1 − alpha) × lexical`, default alpha 0.6
- **MMR**: diversity re-rank so you don't get five near-identical chunks from
  the same SOP

### 6.3 Provenance is mandatory

Every chunk carries `source_id`, `source_type`, `param`, `asset` and
`effective_date`. Without provenance the output is not citable, and an
uncitable remedy is useless in a GxP context.

### 6.4 A bug worth documenting

`maintenance_log.csv` lives in the `capa/` directory. The original type
inference scanned the whole path, matched `"capa"` from the *directory*, and
applied the CAPA verbalizer to maintenance rows — silently producing
`None`-filled chunks. Fixed by checking the **filename before the
directory**, with a regression test
(`test_maintenance_not_mislabelled_as_capa`).

This is the class of bug that does not raise an exception and does not fail a
smoke test. It just quietly degrades retrieval quality.

---

## 7. Phase 3 — The agent layer

**`services/agents/src/agents.py`**

### 7.1 Why plain Python

The orchestration is ~40 lines of dispatch logic. It does not need a graph
framework. Consequences:

- No LangChain / LangGraph dependency
- No framework version churn
- The RBAC check is a visible `if`, not a decorator you have to trust
- Easy to validate — an auditor can read the whole control path

A Haystack adapter scaffold is included for teams standardising on it
(`build_haystack_pipeline`), wrapping each `ParameterAgent` as a
`ComponentTool` under a coordinator `Agent`.

### 7.2 The agent cycle

```
plan()       → 3 sub-questions (SOP / CAPA / OEM+maintenance)
retrieve()   → per-sub-question search, filtered by param and source_type
               → Reciprocal Rank Fusion across sub-questions
synthesize() → LLM composes cited steps, or grounded template fallback
```

**RRF** (`1 / (60 + rank)`) is used rather than score averaging because the
three sub-questions return scores on different scales; rank is comparable,
raw score is not.

### 7.3 The fallback is grounded, not generic

When no LLM is reachable, `_template_steps` extracts literal `Step N.`
sentences from the retrieved SOP text. The output is less fluent but **more
defensible** — it is verbatim approved procedure with a citation, not a
paraphrase. For a regulated context that is arguably the safer default.

A dedup guard (`seen` set on the first 60 characters) prevents overlapping
chunks from re-emitting the same step — another silent bug, caught by
`test_no_duplicate_steps`.

### 7.4 The human escalation agent

One agent, not five. All parameter agents route to it. It:

- refuses a signature with a name under 3 characters
- refuses a signature with a reason under 4 characters
- refuses to sign an unknown event

These are not cosmetic validations. Under 21 CFR Part 11 a signature must
carry the signer's identity and the *meaning* of the signature.

---

## 8. Phase 4 — Governance

### 8.1 Decision table

Encoded in `infra/config/opa/bioscada.rego`, mirrored in the API and in
`ParameterAgent.esign_policy`, and tested in three places:

| Condition | Decision |
|---|---|
| Missing `agent.execute.<param>` role | DENY |
| Parameter is pH | REQUIRE_ESIGN |
| Zone is trip | REQUIRE_ESIGN |
| P(breach) ≥ 0.80 | REQUIRE_ESIGN |
| Otherwise | ALLOW |

The pH rule is absolute and comes from SOP-PH-009 — pH setpoint changes
always require QA signature, regardless of how confident the model is. This
is the case where the policy deliberately overrides the model.

**Boundary tested explicitly**: P = 0.79 → ALLOW, P = 0.80 → REQUIRE_ESIGN.
Off-by-one errors in a policy threshold are exactly the kind of defect an
auditor looks for.

### 8.2 Tamper-evident audit

`infra/config/postgres-init.sql` implements hash chaining:

```
row_hash = sha256(prev_hash || ts || action || event_id || detail)
```

Plus a trigger that raises on `UPDATE` or `DELETE`. Modifying any historical
row invalidates every subsequent hash — verified by
`test_tamper_breaks_chain`.

This is a supporting control, not a certified WORM appliance. For production,
mirror to MinIO with object-lock enabled.

---

## 9. Integration tier

**`services/api/src/main.py`**

One FastAPI app exposing every phase. Design choices:

- **Graceful degradation.** No Kafka → prediction, RAG and governance still
  work. No trained models → analytic fallback. No LLM → grounded template.
  The portal never shows a blank screen because one service is down.
- **Feature names on inference.** Models are fitted on named columns, so
  inference passes a DataFrame rather than a bare array — otherwise sklearn
  emits a warning on every call and, worse, silently accepts mis-ordered
  columns.
- **In-memory state.** Deliberate, for a demo. Production swaps `State` for
  Postgres — the audit schema is already written.

---

## 10. Frontend

**`frontend/public/index.html`**

Single file, no build step, no npm dependency tree. It talks to the real API;
nothing is mocked. API base resolution handles three cases: served by nginx
(same-origin `/api`), opened as a local file (`http://localhost:8000/api`),
or overridden via `?api=`.

The `file://` case was a genuine bug — `location.port` is empty for a file
URL, which matched the same-origin branch and produced `file:///api/health`.

---

## 11. What this does not do

Stated plainly, because a panel will ask:

- **No real plant connection.** The simulator is a stand-in. An OPC-UA / PI /
  historian bridge is required for a real deployment.
- **No CSV/PDF ingestion at scale.** The loader handles the sample corpus.
  Production needs `unstructured` or Tika, plus incremental re-indexing.
- **No model monitoring.** Drift detection and scheduled retraining are not
  implemented. MLflow is wired for tracking, not for continuous evaluation.
- **No RAG evaluation harness.** RAGAS is listed in the stack but not
  integrated. Retrieval quality is asserted by a handful of tests, not
  measured systematically.
- **No multi-tenancy or realm import.** Keycloak runs in dev mode with an
  empty realm; the JWT is minted by the API for demonstration rather than
  validated against a real IdP.

Each of these is a known gap with an obvious next step, not a hidden
limitation.

---

## 12. Validation posture (GxP)

The honest framing for a regulated environment:

| Path | Character | Validation approach |
|---|---|---|
| Telemetry → features → score → event | Deterministic, replayable | Conventional CSV — IQ/OQ/PQ, replay tests |
| Retrieval → generation → remedy card | Probabilistic | Decision-support: validate the *process* (retrieval provenance, citation integrity, policy gate), not the model's output text |
| Policy decision → e-signature → audit | Deterministic, enforced | Test the decision table exhaustively; prove audit immutability |

The system is designed so the probabilistic component **never actuates
anything on its own**. It produces a recommendation; a deterministic policy
gate and a human signature stand between it and the plant. That separation is
what makes the architecture defensible.
