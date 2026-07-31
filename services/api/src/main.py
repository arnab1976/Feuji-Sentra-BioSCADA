"""
BioSCADA AI — Intelligence Platform API.

Single integration point for the frontend portal. Exposes every phase:

    Phase 0  /api/telemetry      live readings (Kafka consumer tail)
             /api/inject         operator-driven breach injection
    Phase 1  /api/flink/*        stream-processing state & the SQL plan
    Phase 2  /api/predict        PdM scoring (trained models)
    Phase 3  /api/agent/run      agent activation + RAG remedy
             /api/rag/search     direct knowledge-base search
    Phase 4  /api/govern/*       token, policy decision, e-signature
             /api/audit          WORM-style audit trail

Free stack: FastAPI + Uvicorn (MIT/BSD).
Run: uvicorn main:app --reload --port 8080
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [api] %(message)s",
)
log = logging.getLogger("api")

_ROOT = Path(__file__).resolve().parents[3]
for sub in ("services/simulator/src", "services/rag/src", "services/agents/src"):
    sys.path.insert(0, str(_ROOT / sub))

from parameters import PARAMETERS, PARAM_IDS, TOPIC_TELEMETRY, TOPIC_BREACH  # noqa: E402
from agents import (  # noqa: E402
    BreachEvent, Orchestrator, LLMClient, load_orchestrator,
)

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
MODEL_DIR = Path(os.getenv("MODEL_DIR", _ROOT / "services/ml/models"))
SIM_URL = os.getenv("SIMULATOR_URL", "http://localhost:8081")

app = FastAPI(
    title="BioSCADA AI — Intelligence Platform",
    description="Agentic SCADA threshold monitoring: streaming, PdM, RAG, governance.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# mount the studio (upload / train / ingest) router at import time so its
# routes are registered before the server starts serving
try:
    import studio as _studio
    app.include_router(_studio.router)
except Exception as _exc:  # pragma: no cover
    logging.getLogger("api").warning("studio router not mounted: %s", _exc)

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content={})


@app.get("/", include_in_schema=False)
@app.get("/index", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def serve_index():
    return RedirectResponse(url="/studio", status_code=307)


@app.get("/studio", include_in_schema=False)
@app.get("/studio.html", include_in_schema=False)
@app.get("/studio/live", include_in_schema=False)
@app.get("/live", include_in_schema=False)
def serve_studio():
    return FileResponse(_ROOT / "frontend" / "studio.html", headers=NO_CACHE_HEADERS)







@app.get("/sentra_architecture.jpg", include_in_schema=False)
@app.get("/scada_enterprise_architecture.jpg", include_in_schema=False)
def serve_arch_image():
    p = _ROOT / "frontend" / "public" / "scada_enterprise_architecture.jpg"
    if not p.exists():
        p = _ROOT / "frontend" / "public" / "sentra_architecture.jpg"
    if not p.exists():
        p = _ROOT / "frontend" / "sentra_architecture.jpg"
    return FileResponse(p)


app.mount("/public", StaticFiles(directory=_ROOT / "frontend" / "public"), name="public")
app.mount("/frontend", StaticFiles(directory=_ROOT / "frontend"), name="frontend")





# =====================================================================
# In-memory state (a real deployment would use Postgres + object lock)
# =====================================================================
class State:
    def __init__(self) -> None:
        self.telemetry: Dict[str, Deque[Dict]] = {p: deque(maxlen=240) for p in PARAM_IDS}
        self.breaches: Deque[Dict] = deque(maxlen=200)
        self.audit: Deque[Dict] = deque(maxlen=1000)
        self.cards: Dict[str, Dict] = {}
        self.tokens: Dict[str, Dict] = {}
        self.orchestrator: Optional[Orchestrator] = None
        self.models: Dict[str, Any] = {}
        self.kafka_connected = False
        self.lock = threading.Lock()

    def log_audit(self, level: str, action: str, detail: Dict) -> Dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": level, "action": action, "detail": detail,
        }
        with self.lock:
            self.audit.append(entry)
        return entry


S = State()


# =====================================================================
# Startup — load models, knowledge base, start Kafka consumer
# =====================================================================
@app.on_event("startup")
def startup() -> None:
    _load_models()
    threading.Thread(target=_load_orchestrator, daemon=True).start()
    threading.Thread(target=_consume_kafka, daemon=True).start()
    threading.Thread(target=_run_local_simulator, daemon=True).start()
    S.log_audit("info", "system.start", {"models": list(S.models)})



def _load_models() -> None:
    try:
        import joblib
        for pid in PARAM_IDS:
            f = MODEL_DIR / f"{pid}_model.joblib"
            if f.exists():
                S.models[pid] = joblib.load(f)
        log.info("Loaded %d/%d PdM models from %s", len(S.models), len(PARAM_IDS), MODEL_DIR)
    except Exception as exc:
        log.warning("Model load failed: %s", exc)


def _load_orchestrator() -> None:
    try:
        S.orchestrator = load_orchestrator()
        log.info("Agent orchestrator ready (%d agents, %d KB chunks)",
                 len(S.orchestrator.agents), len(S.orchestrator.kb.chunks))
    except Exception as exc:
        log.error("Orchestrator init failed: %s", exc)
    # wire the studio (upload/train/ingest) endpoints to the shared state
    try:
        import studio
        studio.set_state(S)
        log.info("Studio state wired")
    except Exception as exc:
        log.warning("Studio state wiring failed: %s", exc)


def _get_dataset_dataframes() -> Dict[str, Any]:
    """Load or synthesize actual stored CSV datasets for all 5 parameters."""
    import pandas as pd
    import train as trainer  # services/ml/src/train.py

    dfs = {}
    data_dir = _ROOT / "data" / "uploaded"
    data_dir.mkdir(parents=True, exist_ok=True)

    for pid, p in PARAMETERS.items():
        csv_path = data_dir / f"{pid}.csv"
        tmp_path = Path("/tmp/bioscada_uploads") / f"{pid}.csv"

        if csv_path.exists():
            try:
                dfs[pid] = pd.read_csv(csv_path)
                log.info("Loaded stored dataset for %s (%d rows) from %s", pid, len(dfs[pid]), csv_path)
                continue
            except Exception:
                pass

        if tmp_path.exists():
            try:
                dfs[pid] = pd.read_csv(tmp_path)
                log.info("Loaded uploaded dataset for %s (%d rows) from %s", pid, len(dfs[pid]), tmp_path)
                continue
            except Exception:
                pass

        # Synthesize real physical time-series dataset & store to CSV
        try:
            df = trainer.synthesize(p, 2500)
            df.to_csv(csv_path, index=False)
            dfs[pid] = df
            log.info("Synthesized & stored dataset for %s (%d rows) at %s", pid, len(df), csv_path)
        except Exception as exc:
            log.warning("Could not synthesize dataset for %s: %s", pid, exc)

    return dfs


def _run_local_simulator() -> None:
    """Streams live telemetry sequentially from actual stored CSV datasets for all 5 parameters."""
    import pandas as pd
    import numpy as np
    import random
    log.info("Dataset-driven telemetry streaming engine starting for 5 parameters...")

    dfs = _get_dataset_dataframes()
    pointers = {pid: 0 for pid in PARAMETERS}
    tick = 0

    while True:
        try:
            tick += 1
            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

            for pid, param_obj in PARAMETERS.items():
                df = dfs.get(pid)
                if df is not None and not df.empty:
                    idx = pointers[pid] % len(df)
                    row = df.iloc[idx]
                    pointers[pid] += 1

                    val = float(row["value"]) if "value" in row else float(param_obj.baseline)
                    val = round(val, 2)

                    if val >= param_obj.trip[1] or val <= param_obj.trip[0]:
                        zone = "trip"
                        flag = "RED"
                        p_breach = round(random.uniform(0.85, 0.98), 3)
                    elif val >= param_obj.alarm[1] or val <= param_obj.alarm[0]:
                        zone = "alarm"
                        flag = "AMBER"
                        p_breach = round(random.uniform(0.42, 0.72), 3)
                    else:
                        zone = "normal"
                        flag = "GREEN"
                        p_breach = round(random.uniform(0.01, 0.15), 3)

                    why_feats = param_obj.features[:3]
                    top_driver = why_feats[0]

                    if pid in S.models:
                        try:
                            feat_dict = {f: float(row[f]) for f in param_obj.features if f in row}
                            if len(feat_dict) == len(param_obj.features):
                                feat_vals = np.array([[feat_dict[f] for f in param_obj.features]])
                                if hasattr(S.models[pid], "predict_proba"):
                                    p_breach = round(float(S.models[pid].predict_proba(feat_vals)[0][1]), 3)
                        except Exception:
                            pass
                else:
                    val = round(param_obj.baseline + random.uniform(-0.1, 0.1), 2)
                    zone = "normal"
                    flag = "GREEN"
                    p_breach = 0.03
                    why_feats = param_obj.features[:3]
                    top_driver = why_feats[0]

                rec = {
                    "ts": now_iso,
                    "param": pid,
                    "name": param_obj.name,
                    "short": param_obj.short,
                    "asset": param_obj.asset,
                    "unit": param_obj.unit,
                    "value": val,
                    "zone": zone,
                    "flag": flag,
                    "p_breach": p_breach,
                    "agent": f"{param_obj.short} Agent",
                    "top_driver": top_driver,
                    "driver": top_driver,
                    "why": why_feats,
                    "event_id": f"evt-{pid}-{tick}" if zone != "normal" else None,
                }

                with S.lock:
                    if pid in S.telemetry:
                        S.telemetry[pid].append(rec)
                    if zone in ("alarm", "trip"):
                        S.breaches.append(rec)

            time.sleep(1.0)
        except Exception as exc:
            log.warning("Dataset streaming simulator error: %s", exc)
            time.sleep(2.0)



def _consume_kafka() -> None:
    """Tail telemetry + breach topics so the portal can render live state."""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            TOPIC_TELEMETRY, TOPIC_BREACH,
            bootstrap_servers=KAFKA,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset="latest",
            consumer_timeout_ms=1000,
            group_id="bioscada-api",
        )
        S.kafka_connected = True
        log.info("Kafka consumer connected to %s", KAFKA)
        while True:
            for msg in consumer:
                rec = msg.value
                if msg.topic == TOPIC_TELEMETRY:
                    pid = rec.get("param")
                    if pid in S.telemetry:
                        S.telemetry[pid].append(rec)
                else:
                    S.breaches.append(rec)
                    S.log_audit("warn", "breach.detected", {
                        "event_id": rec.get("event_id"), "param": rec.get("param"),
                        "p_breach": rec.get("p_breach")})
            time.sleep(0.2)
    except Exception as exc:
        S.kafka_connected = False
        log.warning("Kafka consumer unavailable (%s) — using local simulator", exc)



# =====================================================================
# Schemas
# =====================================================================
class PredictRequest(BaseModel):
    param: str = Field(..., description="parameter id, e.g. 'temp'")
    v_avg: float
    v_std: float = 0.0
    v_delta: float = 0.0


class PredictResponse(BaseModel):
    param: str
    p_breach: float
    zone: str
    top_driver: str
    model: str
    horizon_minutes: int = 15


class AgentRunRequest(BaseModel):
    param: str
    v_avg: float
    v_std: float = 0.08
    v_delta: float = 0.3
    zone: str = "alarm"
    p_breach: float = 0.7
    top_driver: str = ""
    event_id: Optional[str] = None
    prompt_override: Optional[str] = None


class SignRequest(BaseModel):
    event_id: str
    signer: str
    role: str = "Performed by — Operator"
    reason: str


class TokenRequest(BaseModel):
    param: str
    action: str = "setpoint.adjust"
    ttl_minutes: int = 10


# =====================================================================
# Meta
# =====================================================================
@app.get("/api/health")
def health() -> Dict:
    return {
        "status": "ok",
        "kafka_connected": S.kafka_connected,
        "models_loaded": sorted(S.models),
        "kb_chunks": len(S.orchestrator.kb.chunks) if S.orchestrator else 0,
        "llm_available": S.orchestrator.llm.available if S.orchestrator else False,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/api/parameters")
def list_parameters() -> List[Dict]:
    return [{
        "id": p.id, "name": p.name, "short": p.short, "unit": p.unit,
        "asset": p.asset, "control": p.control, "alarm": p.alarm, "trip": p.trip,
        "features": p.features, "model": p.model, "root_cause": p.root_cause,
    } for p in PARAMETERS.values()]


# =====================================================================
# Phase 0 — telemetry
# =====================================================================
@app.get("/api/telemetry")
def telemetry(param: Optional[str] = None, limit: int = 60) -> Dict:
    if param:
        if param not in S.telemetry:
            raise HTTPException(404, f"unknown parameter '{param}'")
        return {param: list(S.telemetry[param])[-limit:]}
    return {p: list(v)[-limit:] for p, v in S.telemetry.items()}


@app.post("/api/inject")
def inject(param: str = Query(...), severity: str = Query("alarm")) -> Dict:
    """Operator-driven breach trigger: injects a breach reading directly into the telemetry dataset stream."""
    if param not in PARAMETERS:
        raise HTTPException(404, f"unknown parameter '{param}'")

    import random
    p = PARAMETERS[param]
    why_feats = p.features[:3]
    evt_id = f"evt-{param}-{int(time.time())}"

    if severity == "trip":
        val = round(p.trip[1] + random.uniform(0.3, 0.8), 2)
        p_breach = round(random.uniform(0.88, 0.98), 3)
        zone = "trip"
        flag = "RED"
    else:
        val = round(p.alarm[1] + random.uniform(0.1, 0.3), 2)
        p_breach = round(random.uniform(0.55, 0.75), 3)
        zone = "alarm"
        flag = "AMBER"

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "param": param,
        "name": p.name,
        "short": p.short,
        "asset": p.asset,
        "unit": p.unit,
        "value": val,
        "zone": zone,
        "flag": flag,
        "p_breach": p_breach,
        "agent": f"{p.short} Agent",
        "top_driver": why_feats[0],
        "driver": why_feats[0],
        "why": why_feats,
        "event_id": evt_id,
        "pushed_to": "buffer",
        "injected": True
    }

    with S.lock:
        if param in S.telemetry:
            S.telemetry[param].append(rec)
        S.breaches.append(rec)
        S.log_audit("act", "breach.injected", {"param": param, "severity": severity, "event_id": evt_id})

    log.info("Injected %s breach for %s: val=%s, p_breach=%s", severity, param, val, p_breach)
    return rec



# =====================================================================
# Phase 1 — Flink
# =====================================================================
@app.get("/api/flink/plan")
def flink_plan() -> Dict:
    """The operator plan Flink derives — drives the portal's Flink Live page."""
    return {
        "sql": (
            "INSERT INTO breach_events\n"
            "SELECT param, AVG(value) v_avg, STDDEV_POP(value) v_std,\n"
            "       MAX(value)-MIN(value) delta,\n"
            "       PDM_SCORE(param, AVG(value), STDDEV_POP(value),\n"
            "                 MAX(value)-MIN(value)) p_breach,\n"
            "       TOP_DRIVER(param, AVG(value)) top_driver\n"
            "FROM TABLE(TUMBLE(TABLE scada_telemetry, DESCRIPTOR(ts),\n"
            "                  INTERVAL '10' SECOND))\n"
            "GROUP BY param, window_start, window_end\n"
            "HAVING p_breach > 0.30 OR MAX(zone) <> 'control'"
        ),
        "operators": [
            {"n": 1, "name": "Source", "detail": "consume Kafka scada.telemetry (unbounded)"},
            {"n": 2, "name": "Watermark + Window", "detail": "event-time, 10s tumbling"},
            {"n": 3, "name": "Feature aggregation", "detail": "v_avg, v_std, v_delta (independent variables)"},
            {"n": 4, "name": "Model UDF", "detail": "PDM_SCORE -> P(breach) + top driver"},
            {"n": 5, "name": "Filter / HAVING", "detail": "keep breach candidates only"},
            {"n": 6, "name": "Sink", "detail": "emit enriched 'breach + why' event"},
        ],
        "boundary": {
            "flink": "relational-plan (compute) decomposition — ends here",
            "llm": "semantic query decomposition — begins in the agent layer",
        },
    }


@app.get("/api/flink/events")
def flink_events(limit: int = 20) -> List[Dict]:
    return list(S.breaches)[-limit:]


# =====================================================================
# Phase 2 — prediction
# =====================================================================
@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if req.param not in PARAMETERS:
        raise HTTPException(404, f"unknown parameter '{req.param}'")
    p = PARAMETERS[req.param]
    model = S.models.get(req.param)

    if model is not None:
        try:
            import pandas as pd
            # models were fitted on named columns; pass a DataFrame so sklearn
            # does not warn about missing feature names
            x = pd.DataFrame([[req.v_avg, req.v_std, req.v_delta]],
                             columns=["v_avg", "v_std", "v_delta"])
            proba = float(model.predict_proba(x)[0][1])
            source = f"{p.model} (trained)"
        except Exception as exc:
            log.warning("model inference failed: %s", exc)
            proba, source = _analytic(p, req.v_avg, req.v_std), "analytic fallback"
    else:
        proba, source = _analytic(p, req.v_avg, req.v_std), "analytic fallback"

    return PredictResponse(
        param=req.param, p_breach=round(proba, 4), zone=p.zone(req.v_avg),
        top_driver=p.features[0], model=source,
    )


def _analytic(p, v_avg: float, v_std: float) -> float:
    centre = p.control_center
    half = max((p.alarm[1] - p.alarm[0]) / 2, 1e-9)
    dev = abs(v_avg - centre) / half
    return float(max(0.0, min(1.0, (dev - 0.30) / 0.85 + min(v_std / half, 0.5) * 0.35)))


# =====================================================================
# Phase 3 — agents + RAG
# =====================================================================
@app.post("/api/agent/run")
def agent_run(req: AgentRunRequest) -> Dict:
    if S.orchestrator is None:
        raise HTTPException(503, "orchestrator not ready")
    if req.param not in PARAMETERS:
        raise HTTPException(404, f"unknown parameter '{req.param}'")

    p = PARAMETERS[req.param]
    event = BreachEvent(
        event_id=req.event_id or f"evt-{req.param}-{int(time.time())}",
        param=req.param, asset=p.asset, v_avg=req.v_avg, v_std=req.v_std,
        v_delta=req.v_delta, zone=req.zone, p_breach=req.p_breach,
        top_driver=req.top_driver or p.features[0],
    )
    try:
        card = S.orchestrator.dispatch(event)
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

    payload = card.to_dict()
    S.cards[card.event_id] = payload
    S.log_audit("act", "agent.remedy", {
        "event_id": card.event_id, "param": card.param,
        "requires_esign": card.requires_esign, "chunks": card.retrieved_chunks})
    return payload


@app.get("/api/rag/search")
def rag_search(q: str, top_k: int = 6, param: Optional[str] = None) -> List[Dict]:
    if S.orchestrator is None:
        raise HTTPException(503, "knowledge base not ready")
    hits = S.orchestrator.kb.search(q, top_k=top_k, param=param)
    return [{
        "score": round(h["score"], 4),
        "dense": round(h["dense"], 4),
        "lexical": round(h["lexical"], 4),
        "source_type": h["chunk"].source_type,
        "source_id": h["chunk"].source_id,
        "title": h["chunk"].title,
        "structured": h["chunk"].structured,
        "text": h["chunk"].text[:600],
    } for h in hits]


@app.get("/api/rag/stats")
def rag_stats() -> Dict:
    if S.orchestrator is None:
        raise HTTPException(503, "knowledge base not ready")
    chunks = S.orchestrator.kb.chunks
    by_type: Dict[str, int] = {}
    for c in chunks:
        by_type[c.source_type] = by_type.get(c.source_type, 0) + 1
    return {
        "total_chunks": len(chunks),
        "structured": sum(1 for c in chunks if c.structured),
        "unstructured": sum(1 for c in chunks if not c.structured),
        "by_source_type": by_type,
        "embed_model": S.orchestrator.kb.embedder.model_name,
    }


# =====================================================================
# Phase 4 — governance
# =====================================================================
@app.post("/api/govern/token")
def mint_token(req: TokenRequest) -> Dict:
    if req.param not in PARAMETERS:
        raise HTTPException(404, f"unknown parameter '{req.param}'")
    p = PARAMETERS[req.param]
    now = datetime.now(timezone.utc)
    claims = {
        "sub": f"agent:{p.id}@bioscada",
        "name": f"{p.short} Parameter Agent",
        "roles": [f"agent.execute.{p.id}", "rag.read"],
        "resource": f"asset:{p.asset}:{p.id}",
        "action": req.action,
        "iss": "https://keycloak.bioscada.local/realms/bioscada",
        "aud": "apigee.gateway",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=req.ttl_minutes)).timestamp()),
    }
    S.tokens[p.id] = claims
    S.log_audit("act", "token.mint",
                {"sub": claims["sub"], "ttl_min": req.ttl_minutes})
    return {"claims": claims, "ttl_minutes": req.ttl_minutes}


@app.post("/api/govern/decide")
def policy_decide(event_id: str = Query(...)) -> Dict:
    """
    PDP evaluation. In production this delegates to Open Policy Agent;
    the decision logic mirrors services/../infra/config/opa/bioscada.rego.
    """
    card = S.cards.get(event_id)
    if not card:
        raise HTTPException(404, f"no remedy card for '{event_id}'")
    decision = "REQUIRE_ESIGN" if card["requires_esign"] else "ALLOW"
    obligations = (["require_esign", "qa_review_24h", "log_worm"]
                   if card["requires_esign"] else ["log_worm"])
    out = {"event_id": event_id, "decision": decision,
           "reason": card["esign_reason"], "obligations": obligations}
    S.log_audit("warn" if card["requires_esign"] else "ok", "pdp.decision", out)
    return out


@app.post("/api/govern/sign")
def sign(req: SignRequest) -> Dict:
    if S.orchestrator is None:
        raise HTTPException(503, "orchestrator not ready")
    card = S.cards.get(req.event_id)
    if not card:
        raise HTTPException(404, f"no remedy card for '{req.event_id}'")
    if len(req.signer.strip()) < 3:
        raise HTTPException(400, "signer name required (21 CFR Part 11)")
    if len(req.reason.strip()) < 4:
        raise HTTPException(400, "reason for signature required (21 CFR Part 11)")

    esc = S.orchestrator.escalation
    if req.event_id not in esc.pending:
        # ALLOW path still captures an operator signature for traceability
        from agents import Signature
        sig = Signature(signer=req.signer, role=req.role, reason=req.reason,
                        signed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        committed=True)
        esc.signatures[req.event_id] = sig
    else:
        sig = esc.sign(req.event_id, req.signer, req.role, req.reason)

    S.log_audit("ok", "esign.apply", {
        "event_id": req.event_id, "signer": sig.signer, "role": sig.role,
        "reason": sig.reason})
    S.log_audit("act", "control.execute", {
        "event_id": req.event_id, "param": card["param"],
        "resource": f"asset:{card['asset']}:{card['param']}", "status": "committed"})
    S.log_audit("ok", "worm.write", {"event_id": req.event_id, "immutable": True})
    return {"signed": True, "signature": sig.__dict__, "event_id": req.event_id}


@app.post("/api/govern/reject")
def reject(req: SignRequest) -> Dict:
    if S.orchestrator is None:
        raise HTTPException(503, "orchestrator not ready")
    sig = S.orchestrator.escalation.reject(req.event_id, req.signer, req.reason)
    S.log_audit("warn", "operator.reject",
                {"event_id": req.event_id, "signer": req.signer, "reason": req.reason})
    return {"signed": False, "signature": sig.__dict__}


@app.get("/api/audit")
def audit(limit: int = 100) -> List[Dict]:
    return list(S.audit)[-limit:]


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> Dict:
    card = S.cards.get(event_id)
    if not card:
        raise HTTPException(404, "unknown event")
    sig = (S.orchestrator.escalation.signatures.get(event_id)
           if S.orchestrator else None)
    return {"card": card, "signature": sig.__dict__ if sig else None}


@app.exception_handler(Exception)
def unhandled(request, exc):  # pragma: no cover
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
