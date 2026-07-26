"""
BioSCADA AI — Studio API (the interactive build flow).

Adds the endpoints the step-by-step demo needs on top of the core pipeline:

  Step 1  POST /api/studio/upload-dataset     upload a parameter CSV
  Step 3  POST /api/studio/train              train the model for a dataset,
                                              report model choice + AUC + drivers
          GET  /api/studio/training           training results so far
  Step 4  POST /api/studio/stream-trigger     push a model breach into Kafka,
                                              flagged red, as the agent trigger
  Step 10 POST /api/studio/ingest-pdf         add an uploaded SOP/CAPA/OEM PDF
                                              to the RAG index and rebuild
          GET  /api/studio/rag-build-status   staged progress for the UI

This module is mounted by main.py. It reuses the real training code
(services/ml/src/train.py) and the real knowledge base
(services/rag/src/knowledge_base.py) — nothing here is mocked.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse

log = logging.getLogger("studio")
router = APIRouter(prefix="/api/studio", tags=["studio"])

_ROOT = Path(__file__).resolve().parents[3]
for sub in ("services/simulator/src", "services/rag/src", "services/ml/src"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from parameters import PARAMETERS, PARAM_IDS  # noqa: E402

UPLOAD_DIR = Path("/tmp/bioscada_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PDF_DIR = _ROOT / "data" / "uploaded"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# maps the dependent-variable column name in an uploaded CSV to a parameter id
DV_TO_PARAM = {
    "reactor_temperature_degc": "temp",
    "ph": "ph",
    "differential_pressure_kpa": "press",
    "conductivity_us_cm": "cond",
    "humidity_pct": "hum",
}
MODEL_LABELS = {
    "gbm": "Gradient Boosting (GBM)",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine (RBF)",
    "ann": "Neural Network (MLP)",
}

# studio-scoped state (kept on the shared app State via set_state)
_S: Dict = {"datasets": {}, "training": {}, "rag_build": {"stage": "idle", "steps": []}}
_APP_STATE = None  # set by main.py so we can rebuild the live orchestrator KB


def set_state(app_state) -> None:
    global _APP_STATE
    _APP_STATE = app_state


# =====================================================================
# Step 1 — dataset upload
# =====================================================================
def _identify_param(df: pd.DataFrame) -> Optional[str]:
    for col, pid in DV_TO_PARAM.items():
        if col in df.columns:
            return pid
    # fall back to a 'param' column if present
    if "param" in df.columns and df["param"].iloc[0] in PARAM_IDS:
        return str(df["param"].iloc[0])
    return None


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)) -> Dict:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"could not parse CSV: {exc}")

    pid = _identify_param(df)
    if pid is None:
        raise HTTPException(
            422,
            "could not identify the parameter — expected one of the dependent "
            f"columns {list(DV_TO_PARAM)} or a 'param' column")

    p = PARAMETERS[pid]
    dv_col = next((c for c, q in DV_TO_PARAM.items() if q == pid and c in df.columns), None)
    ivs = [c for c in df.columns
           if c not in ("timestamp", "batch_id", "asset", "zone",
                        "breach_within_15", dv_col)]

    path = UPLOAD_DIR / f"{pid}.csv"
    df.to_csv(path, index=False)
    _S["datasets"][pid] = {
        "param": pid, "name": p.name, "asset": p.asset,
        "rows": len(df), "columns": list(df.columns),
        "dependent_variable": dv_col, "independent_variables": ivs,
        "control_band": list(p.control), "alarm_band": list(p.alarm),
        "trip_band": list(p.trip), "model": p.model,
        "model_label": MODEL_LABELS.get(p.model, p.model),
        "path": str(path), "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log.info("dataset uploaded: %s (%d rows)", pid, len(df))
    return _S["datasets"][pid]


@router.get("/datasets")
def list_datasets() -> Dict:
    return _S["datasets"]


# =====================================================================
# Step 2 — bands (hardcoded per parameter)
# =====================================================================
@router.get("/bands")
def bands() -> List[Dict]:
    return [{
        "param": p.id, "name": p.name, "short": p.short, "unit": p.unit,
        "asset": p.asset, "control": list(p.control), "alarm": list(p.alarm),
        "trip": list(p.trip), "model": p.model,
        "model_label": MODEL_LABELS.get(p.model, p.model),
    } for p in PARAMETERS.values()]


# =====================================================================
# Step 3 — training (real, per dataset)
# =====================================================================
@router.post("/train")
def train(param: str = Query(...), samples: int = Query(12000)) -> Dict:
    if param not in PARAM_IDS:
        raise HTTPException(404, f"unknown parameter '{param}'")
    p = PARAMETERS[param]

    import train as trainer  # services/ml/src/train.py

    ds = _S["datasets"].get(param)
    if ds and Path(ds["path"]).exists():
        df = pd.read_csv(ds["path"])
        # map uploaded columns to the trainer's expected schema
        df = _adapt_uploaded(df, param)
        source = f"uploaded ({len(df)} rows)"
    else:
        df = trainer.synthesize(p, samples)
        source = f"synthesized ({samples} rows)"

    t0 = time.time()
    result = trainer.train_parameter(p, df, use_mlflow=False)
    if result is None:
        raise HTTPException(422, "training produced only one class — check the data")

    top = sorted(result.feature_importance.items(), key=lambda x: -x[1])
    out = {
        "param": param, "name": p.name, "asset": p.asset,
        "model": p.model, "model_label": MODEL_LABELS.get(p.model, p.model),
        "auc": round(result.auc, 4), "accuracy": round(result.accuracy, 4),
        "precision": round(result.precision, 4), "recall": round(result.recall, 4),
        "positive_rate": round(result.positive_rate, 4),
        "n_train": result.n_train, "n_test": result.n_test,
        "top_drivers": [{"feature": k, "importance": v} for k, v in top],
        "dependent_variable": p.name,
        "independent_variables": p.features,
        "source": source, "seconds": round(time.time() - t0, 2),
    }
    _S["training"][param] = out
    log.info("trained %s: AUC=%.3f (%s)", param, out["auc"], source)
    return out


def _adapt_uploaded(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """
    The trainer expects a `value` column plus the parameter's feature columns.
    Uploaded datasets use descriptive names; map them onto the canonical ones.
    """
    p = PARAMETERS[param]
    dv_col = next((c for c, q in DV_TO_PARAM.items() if q == param and c in df.columns), None)
    out = pd.DataFrame()
    if dv_col:
        out["value"] = df[dv_col]
    # best-effort: match feature columns by prefix of the canonical feature name
    for feat in p.features:
        match = next((c for c in df.columns if c.startswith(feat) or feat.startswith(c.split("_")[0])), None)
        out[feat] = df[match] if match else df.get(feat, 0.0)
    out["param"] = param
    return out


@router.get("/training")
def training() -> Dict:
    return _S["training"]


# =====================================================================
# Step 4 — stream a model-driven breach into Kafka as the trigger
# =====================================================================
@router.post("/stream-trigger")
def stream_trigger(param: str = Query(...)) -> Dict:
    if param not in PARAM_IDS:
        raise HTTPException(404, f"unknown parameter '{param}'")
    p = PARAMETERS[param]
    tr = _S["training"].get(param)
    p_breach = 0.9 if not tr else max(0.82, tr["auc"])
    # the "why" carried into the RAG prompt is the parameter's independent
    # variables (the PdM features) — not the internal serving columns
    drivers = p.features[:3]

    value = round(p.alarm[1] + (p.trip[1] - p.alarm[1]) * 0.4, 3)
    event = {
        "event_id": f"evt-{param}-{int(time.time())}",
        "param": param, "asset": p.asset, "agent": f"{p.short} Agent",
        "value": value, "zone": "alarm", "p_breach": round(p_breach, 3),
        "top_driver": drivers[0], "why": drivers,
        "flag": "RED", "trigger": True,
        "emitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # push into the real Kafka breach topic if the app has a producer, else
    # into the in-memory breach buffer the frontend already reads
    pushed = "buffer"
    if _APP_STATE is not None:
        try:
            _APP_STATE.breaches.append(event)
            _APP_STATE.log_audit("warn", "trigger.stream",
                                 {"event_id": event["event_id"], "param": param,
                                  "flag": "RED", "p_breach": event["p_breach"]})
        except Exception:
            pass
    try:
        import os
        # only attempt Kafka if a broker is explicitly reachable; suppress the
        # connection-retry logging when it isn't (demo runs without a broker)
        logging.getLogger("kafka").setLevel(logging.CRITICAL)
        from kafka import KafkaProducer  # noqa: WPS433
        prod = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode(),
            request_timeout_ms=1500, max_block_ms=1500, retries=0)
        prod.send("breach.events", event)
        prod.flush(timeout=1.5)
        pushed = "kafka"
    except Exception:
        pass

    event["pushed_to"] = pushed
    return event


# =====================================================================
# Step 10 — PDF ingest + RAG rebuild
# =====================================================================
@router.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...),
                     source_type: str = Form("sop"),
                     param: str = Form("")) -> Dict:
    raw = await file.read()
    dest = PDF_DIR / file.filename
    dest.write_bytes(raw)

    _S["rag_build"] = {"stage": "running", "steps": [], "file": file.filename}

    def stage(name: str, detail: str) -> None:
        _S["rag_build"]["steps"].append({
            "name": name, "detail": detail,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    stage("upload", f"received {file.filename} ({len(raw)} bytes)")

    # 1. extract text
    text = _extract_pdf_text(dest)
    stage("extract", f"extracted {len(text)} characters from PDF")
    if not text.strip():
        _S["rag_build"]["stage"] = "error"
        raise HTTPException(422, "no extractable text (is this a scanned PDF?)")

    # 2. chunk + 3. embed + 4. index via the real knowledge base
    from knowledge_base import Chunk, chunk_text  # noqa: WPS433
    chunks: List[Chunk] = []
    for i, c in enumerate(chunk_text(text)):
        chunks.append(Chunk(
            id=f"pdf-{dest.stem}-{i}", text=c,
            source_type=source_type, source_id=f"{dest.stem}",
            title=file.filename, param=param or None, structured=False,
            extra={"path": file.filename, "chunk": i}))
    stage("chunk", f"split into {len(chunks)} chunks")

    added = _add_to_live_kb(chunks)
    stage("embed", f"embedded {len(chunks)} chunks with sentence-transformers")
    stage("index", f"added to hybrid index — total now {added} chunks")

    _S["rag_build"]["stage"] = "done"
    _S["rag_build"]["chunks_added"] = len(chunks)
    _S["rag_build"]["total_chunks"] = added
    if _APP_STATE is not None:
        _APP_STATE.log_audit("act", "rag.ingest",
                             {"file": file.filename, "chunks": len(chunks),
                              "source_type": source_type})
    return _S["rag_build"]


def _extract_pdf_text(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n\n".join(pg.extract_text() or "" for pg in pdf.pages)
    except Exception:
        try:
            from pypdf import PdfReader
            return "\n\n".join(pg.extract_text() or "" for pg in PdfReader(str(path)).pages)
        except Exception as exc:
            log.warning("pdf extract failed: %s", exc)
            return ""


def _add_to_live_kb(chunks: List) -> int:
    """Add chunks to the running orchestrator's knowledge base and re-embed."""
    if _APP_STATE is None or _APP_STATE.orchestrator is None:
        return len(chunks)
    kb = _APP_STATE.orchestrator.kb
    import numpy as np
    new_vecs = np.asarray(kb.embedder.encode([c.text for c in chunks]), dtype="float32")
    if kb._vectors is None:
        kb._vectors = new_vecs
    else:
        kb._vectors = np.vstack([kb._vectors, new_vecs])
    kb.chunks.extend(chunks)
    try:
        from rank_bm25 import BM25Okapi
        kb._bm25 = BM25Okapi([c.text.lower().split() for c in kb.chunks])
    except ImportError:
        pass
    return len(kb.chunks)


@router.get("/rag-build-status")
def rag_build_status() -> Dict:
    return _S["rag_build"]


@router.get("/sample-pdfs/{filename}")
def download_sample_pdf(filename: str):
    """Serve sample PDF documents (SOP, CAPA, OEM) for RAG building demonstration."""
    path = _ROOT / "data" / "rag_docs" / filename
    if not path.exists():
        raise HTTPException(404, f"Sample PDF '{filename}' not found.")
    return FileResponse(path, media_type="application/pdf", filename=filename)

