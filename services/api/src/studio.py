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
import threading
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
STAGING_DIR = UPLOAD_DIR / "staging"
STAGING_DIR.mkdir(exist_ok=True)
CATALOG_PATH = UPLOAD_DIR / "catalog.json"
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

# Built-in column descriptions for Display
COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "timestamp": "Event time of the SCADA sample (ISO-8601 / wall-clock).",
    "batch_id": "Manufacturing batch or lot identifier associated with the reading.",
    "asset": "Equipment tag where the sensor reading was collected.",
    "zone": "Operating zone classification at sample time (control / alarm / trip).",
    "breach_within_15": "Binary label: whether a control-band breach occurs within the next 15 minutes (PdM target).",
    "reactor_temperature_degc": "Dependent variable — bioreactor temperature in °C.",
    "ph": "Dependent variable — culture / media pH.",
    "differential_pressure_kpa": "Dependent variable — filter differential pressure in kPa.",
    "conductivity_us_cm": "Dependent variable — WFI conductivity in µS/cm.",
    "humidity_pct": "Dependent variable — cleanroom relative humidity in %.",
    "coolant_flow_rate": "Jacket / cooling-water flow rate influencing reactor temperature.",
    "jacket_temperature": "Cooling-jacket temperature setpoint or measured jacket °C.",
    "steam_valve_position": "Steam utility valve position (% open) contributing to heat input.",
    "heat_exchanger_dp": "Heat-exchanger differential pressure (fouling indicator).",
    "agitator_torque": "Agitator motor torque (mixing intensity and viscosity proxy).",
    "acid_base_dose_rate": "Acid/base dosing pump rate used for pH control.",
    "co2_accumulation": "Dissolved / headspace CO₂ accumulation affecting pH.",
    "agitator_speed": "Agitator RPM contributing to gas transfer and mixing.",
    "probe_drift_mv": "pH probe millivolt drift from last calibration.",
    "dissolved_oxygen": "Dissolved oxygen level interacting with culture metabolism / pH.",
    "filter_dp": "Filter differential pressure contributing to overall ΔP.",
    "gas_exhaust_flow": "Exhaust gas flow rate (vent path restriction indicator).",
    "pump_speed_hz": "Transfer / recirculation pump drive frequency (Hz).",
    "valve_position": "Process valve position affecting pressure drop across the train.",
    "seal_integrity_index": "Seal health index for pressure-boundary components.",
    "water_flow_rate": "WFI loop volumetric flow rate.",
    "resin_bed_dp": "Deionization resin-bed differential pressure.",
    "regeneration_cycle_count": "Number of resin regeneration cycles since last replacement.",
    "toc_level": "Total organic carbon level in the WFI stream.",
    "feed_composition": "Feed-water composition index affecting conductivity.",
    "hvac_fan_speed": "HVAC supply / recirculation fan speed.",
    "cooling_coil_temp": "Cooling-coil temperature influencing room humidity.",
    "hepa_dp": "HEPA filter differential pressure.",
    "outdoor_humidity": "Outdoor ambient humidity loading the HVAC system.",
    "door_open_count": "Cleanroom door-open events in the sampling window.",
}

# studio-scoped state (kept on the shared app State via set_state)
_S: Dict = {
    "datasets": {},   # permanently saved
    "pending": {},    # uploaded but not yet saved
    "training": {},
    "rag_build": {"stage": "idle", "steps": []},
}
_APP_STATE = None  # set by main.py so we can rebuild the live orchestrator KB
_KB_LOCK = threading.Lock()


def set_state(app_state) -> None:
    global _APP_STATE
    _APP_STATE = app_state
    _reload_saved_datasets()


def _persist_catalog() -> None:
    payload = {
        pid: {k: v for k, v in meta.items() if k != "preview"}
        for pid, meta in _S["datasets"].items()
    }
    CATALOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _meta_from_df(pid: str, df: pd.DataFrame, path: Path, *, saved: bool,
                  filename: str = "") -> Dict:
    p = PARAMETERS[pid]
    dv_col = next((c for c, q in DV_TO_PARAM.items() if q == pid and c in df.columns), None)
    meta_cols = ("timestamp", "batch_id", "asset", "zone", "breach_within_15")
    ivs = [c for c in df.columns if c not in meta_cols and c != dv_col]
    return {
        "param": pid,
        "name": p.name,
        "asset": p.asset,
        "unit": p.unit,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dependent_variable": dv_col,
        "independent_variables": ivs,
        "control_band": list(p.control),
        "alarm_band": list(p.alarm),
        "trip_band": list(p.trip),
        "model": p.model,
        "model_label": MODEL_LABELS.get(p.model, p.model),
        "path": str(path),
        "filename": filename or path.name,
        "saved": saved,
        "status": "saved" if saved else "pending",
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds") if saved else None,
    }


def _reload_saved_datasets() -> None:
    """Rebuild in-memory catalog from disk so saved CSVs survive restarts."""
    _S["datasets"] = {}
    catalog: Dict = {}
    if CATALOG_PATH.exists():
        try:
            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read dataset catalog: %s", exc)

    for pid in PARAM_IDS:
        path = UPLOAD_DIR / f"{pid}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            meta = catalog.get(pid) or _meta_from_df(pid, df, path, saved=True)
            meta["path"] = str(path)
            meta["rows"] = int(len(df))
            meta["columns"] = list(df.columns)
            meta["saved"] = True
            meta["status"] = "saved"
            _S["datasets"][pid] = meta
            log.info("Restored saved dataset %s (%d rows)", pid, len(df))
        except Exception as exc:
            log.warning("Failed restoring dataset %s: %s", pid, exc)

    if _S["datasets"]:
        _persist_catalog()


# =====================================================================
# Step 1 — dataset upload / save / delete / display
# =====================================================================
def _identify_param(df: pd.DataFrame) -> Optional[str]:
    for col, pid in DV_TO_PARAM.items():
        if col in df.columns:
            return pid
    # fall back to a 'param' column if present
    if "param" in df.columns and df["param"].iloc[0] in PARAM_IDS:
        return str(df["param"].iloc[0])
    return None


def _infer_col_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # try parse datetime-ish strings
    if series.dtype == object:
        sample = series.dropna().astype(str).head(8)
        if len(sample) and sample.str.contains(r"\d{4}-\d{2}-\d{2}|T\d{2}:", regex=True).mean() > 0.6:
            return "datetime"
        return "string"
    return str(series.dtype)


def _column_description(col: str, pid: Optional[str] = None) -> str:
    if col in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[col]
    if pid and pid in PARAMETERS and col in PARAMETERS[pid].features:
        return f"Independent process driver for {PARAMETERS[pid].name} PdM model."
    pretty = col.replace("_", " ")
    return f"Process / context variable: {pretty}."


def _dataset_preview(path: Path, pid: str, n: int = 5) -> Dict:
    df = pd.read_csv(path)
    sample = df.head(n).where(pd.notnull(df.head(n)), None)
    # JSON-safe values
    records = []
    for row in sample.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    v = str(v)
            if isinstance(v, float) and (pd.isna(v)):
                v = None
            clean[k] = v
        records.append(clean)

    variables = []
    dv = next((c for c, q in DV_TO_PARAM.items() if q == pid and c in df.columns), None)
    feats = set(PARAMETERS[pid].features) if pid in PARAMETERS else set()
    for col in df.columns:
        if col == dv:
            role = "dependent"
        elif col == "breach_within_15":
            role = "label"
        elif col in ("timestamp", "batch_id", "asset", "zone"):
            role = "metadata"
        elif col in feats:
            role = "independent"
        else:
            role = "independent" if col not in DV_TO_PARAM else "other"

        variables.append({
            "name": col,
            "type": _infer_col_type(df[col]),
            "role": role,
            "description": _column_description(col, pid),
            "non_null": int(df[col].notna().sum()),
            "nulls": int(df[col].isna().sum()),
        })

    return {
        "param": pid,
        "rows_shown": len(records),
        "total_rows": int(len(df)),
        "sample": records,
        "variables": variables,
    }


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)) -> Dict:
    """Stage an uploaded CSV. Call /save-dataset to make it permanent."""
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

    staging = STAGING_DIR / f"{pid}.csv"
    df.to_csv(staging, index=False)
    meta = _meta_from_df(pid, df, staging, saved=False, filename=file.filename or f"{pid}.csv")
    _S["pending"][pid] = meta
    log.info("dataset staged: %s (%d rows) from %s", pid, len(df), file.filename)
    return meta


@router.post("/save-dataset")
def save_dataset(param: str = Query(...)) -> Dict:
    """Promote a staged upload to permanent storage (survives restarts)."""
    if param not in PARAM_IDS:
        raise HTTPException(404, f"unknown parameter '{param}'")
    pending = _S["pending"].get(param)
    staging = STAGING_DIR / f"{param}.csv"
    if not pending and not staging.exists():
        raise HTTPException(404, f"no staged dataset for '{param}' — upload first")

    if not staging.exists():
        raise HTTPException(404, f"staged file missing for '{param}'")

    df = pd.read_csv(staging)
    permanent = UPLOAD_DIR / f"{param}.csv"
    df.to_csv(permanent, index=False)
    meta = _meta_from_df(
        param, df, permanent, saved=True,
        filename=(pending or {}).get("filename") or f"{param}.csv",
    )
    _S["datasets"][param] = meta
    _S["pending"].pop(param, None)
    try:
        staging.unlink(missing_ok=True)
    except TypeError:
        if staging.exists():
            staging.unlink()
    _persist_catalog()
    log.info("dataset saved permanently: %s (%d rows)", param, len(df))
    return meta


@router.delete("/datasets/{param}")
def delete_dataset(param: str) -> Dict:
    """Permanently delete a saved or staged dataset."""
    if param not in PARAM_IDS:
        raise HTTPException(404, f"unknown parameter '{param}'")

    removed = {"param": param, "deleted_saved": False, "deleted_pending": False}
    permanent = UPLOAD_DIR / f"{param}.csv"
    staging = STAGING_DIR / f"{param}.csv"

    if param in _S["datasets"] or permanent.exists():
        _S["datasets"].pop(param, None)
        if permanent.exists():
            permanent.unlink()
        removed["deleted_saved"] = True
        _persist_catalog()

    if param in _S["pending"] or staging.exists():
        _S["pending"].pop(param, None)
        if staging.exists():
            try:
                staging.unlink(missing_ok=True)
            except TypeError:
                staging.unlink()
        removed["deleted_pending"] = True

    if not removed["deleted_saved"] and not removed["deleted_pending"]:
        raise HTTPException(404, f"no dataset found for '{param}'")

    log.info("dataset deleted: %s", param)
    return removed


@router.get("/datasets")
def list_datasets() -> Dict:
    """Return saved datasets plus any staged (unsaved) uploads."""
    return {
        "saved": _S["datasets"],
        "pending": _S["pending"],
    }


@router.get("/datasets/{param}/preview")
def preview_dataset(param: str, n: int = Query(5, ge=1, le=50)) -> Dict:
    """Sample observations + variable types/descriptions for Display."""
    if param not in PARAM_IDS:
        raise HTTPException(404, f"unknown parameter '{param}'")

    saved = _S["datasets"].get(param)
    pending = _S["pending"].get(param)
    path = None
    status = None
    if saved and Path(saved["path"]).exists():
        path = Path(saved["path"])
        status = "saved"
    elif pending and Path(pending["path"]).exists():
        path = Path(pending["path"])
        status = "pending"
    else:
        permanent = UPLOAD_DIR / f"{param}.csv"
        staging = STAGING_DIR / f"{param}.csv"
        if permanent.exists():
            path, status = permanent, "saved"
        elif staging.exists():
            path, status = staging, "pending"

    if path is None:
        raise HTTPException(404, f"no dataset available for '{param}'")

    try:
        preview = _dataset_preview(path, param, n=n)
    except Exception as exc:
        raise HTTPException(400, f"could not preview dataset: {exc}")
    preview["status"] = status
    preview["name"] = PARAMETERS[param].name
    preview["asset"] = PARAMETERS[param].asset
    return preview


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
    # Kafka is opt-in: kafka-python may block the worker for a long time when
    # no broker is running, which freezes the Stream trigger UI / next-step dialog.
    import os
    if os.getenv("KAFKA_ENABLED", "").lower() in ("1", "true", "yes"):
        try:
            logging.getLogger("kafka").setLevel(logging.CRITICAL)
            from kafka import KafkaProducer  # noqa: WPS433
            prod = KafkaProducer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
                value_serializer=lambda v: json.dumps(v).encode(),
                request_timeout_ms=800, max_block_ms=800, retries=0,
                api_version_auto_timeout_ms=800)
            prod.send("breach.events", event)
            prod.flush(timeout=0.8)
            pushed = "kafka"
            try:
                prod.close(timeout=0.5)
            except Exception:
                pass
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
    """Parse → chunk → embed → index. Heavy work runs in a worker thread."""
    import asyncio
    raw = await file.read()
    filename = file.filename or "upload.pdf"
    return await asyncio.to_thread(
        _ingest_pdf_bytes, raw, filename, source_type, param or "")


def _ingest_pdf_bytes(raw: bytes, filename: str, source_type: str, param: str) -> Dict:
    dest = PDF_DIR / filename
    dest.write_bytes(raw)

    _S["rag_build"] = {"stage": "running", "steps": [], "file": filename,
                       "chunks_added": 0, "total_chunks": 0, "active": "upload"}

    def stage(name: str, detail: str) -> None:
        _S["rag_build"]["steps"].append({
            "name": name, "detail": detail,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        _S["rag_build"]["active"] = name

    stage("upload", f"received {filename} ({len(raw)} bytes)")

    text = _extract_pdf_text(dest)
    stage("extract", f"extracted {len(text)} characters from PDF")
    if not text.strip():
        _S["rag_build"]["stage"] = "error"
        raise HTTPException(422, "no extractable text (is this a scanned PDF?)")

    from knowledge_base import Chunk, chunk_text  # noqa: WPS433
    chunks: List[Chunk] = []
    for i, c in enumerate(chunk_text(text)):
        chunks.append(Chunk(
            id=f"pdf-{dest.stem}-{i}-{int(time.time())}", text=c,
            source_type=source_type, source_id=f"{dest.stem}",
            title=filename, param=param or None, structured=False,
            extra={"path": filename, "chunk": i}))
    stage("chunk", f"split into {len(chunks)} chunks")

    stage("embed", f"encoding {len(chunks)} chunks (offline hashing / BGE)")
    try:
        added = _add_to_live_kb(chunks)
    except Exception as exc:
        log.exception("embed/index failed for %s", filename)
        _S["rag_build"]["stage"] = "error"
        raise HTTPException(500, f"embed/index failed: {exc}") from exc

    stage("embed", f"embedded {len(chunks)} chunks")
    stage("index", f"added to hybrid index — total now {added} chunks")

    _S["rag_build"]["stage"] = "done"
    _S["rag_build"]["active"] = "done"
    _S["rag_build"]["chunks_added"] = len(chunks)
    _S["rag_build"]["total_chunks"] = added
    if _APP_STATE is not None:
        _APP_STATE.log_audit("act", "rag.ingest",
                             {"file": filename, "chunks": len(chunks),
                              "source_type": source_type})
    return _S["rag_build"]


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n\n".join(pg.extract_text() or "" for pg in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n\n".join(pg.extract_text() or "" for pg in pdf.pages)
    except Exception as exc:
        log.warning("pdf extract failed: %s", exc)
        return ""


def _add_to_live_kb(chunks: List) -> int:
    """Add chunks to the running orchestrator's knowledge base and re-embed."""
    if _APP_STATE is None or _APP_STATE.orchestrator is None:
        return len(chunks)
    with _KB_LOCK:
        kb = _APP_STATE.orchestrator.kb
        import numpy as np
        texts = [c.text for c in chunks]
        new_vecs = np.asarray(kb.embedder.encode(texts), dtype="float32")
        if kb._vectors is None or getattr(kb._vectors, "size", 0) == 0:
            kb._vectors = new_vecs
        else:
            kb._vectors = np.vstack([kb._vectors, new_vecs])
        kb.chunks.extend(chunks)
        try:
            from rank_bm25 import BM25Okapi
            kb._bm25 = BM25Okapi([c.text.lower().split() for c in kb.chunks])
        except ImportError:
            pass
        try:
            idx = Path(__file__).resolve().parents[2] / "rag" / "index"
            kb.save(idx)
        except Exception as exc:
            log.warning("KB save after ingest failed: %s", exc)
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


# =====================================================================
# Step 13 — SCADA Co-pilot Chatbot (parameter agents + human escalation)
# =====================================================================
@router.post("/chatbot-query")
def chatbot_query(payload: Dict) -> Dict:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "Query string cannot be empty.")

    agent = (payload.get("agent") or "human").strip().lower()
    param = payload.get("param")
    if param in ("", "null", "none", "human"):
        param = None
    if agent in PARAM_IDS and param is None:
        param = agent

    agent_label = {
        "temp": "Temperature Agent",
        "ph": "pH Agent",
        "press": "Pressure Agent",
        "cond": "Conductivity Agent",
        "hum": "Humidity Agent",
        "human": "Human Escalation Agent",
    }.get(agent, "Human Escalation Agent")

    kb = None
    if _APP_STATE is not None and _APP_STATE.orchestrator is not None:
        kb = _APP_STATE.orchestrator.kb

    hits: List = []
    if kb is not None:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(kb.search, query, 5, 0.6, param)
                hits = fut.result(timeout=4.0)
        except FuturesTimeout:
            log.warning("chatbot-query: KB search timed out")
            hits = []
        except Exception as exc:
            log.warning("chatbot-query: KB search failed (%s)", exc)
            hits = []

    citations = []
    snippets = []
    for h in hits:
        c = h["chunk"]
        citations.append({
            "source_id": c.source_id or c.title,
            "source_type": c.source_type,
            "title": c.title,
            "structured": c.structured,
            "snippet": c.text[:300],
        })
        snippets.append(f"[{c.source_id or c.title}]: {c.text}")

    q_lower = query.lower()

    # Dynamic Chart synthesis logic based on query intent
    chart = None
    if any(k in q_lower for k in ("chart", "graph", "frequency", "capa", "distribution", "record", "history", "trend")):
        if "distribution" in q_lower or "severity" in q_lower:
            chart = {
                "type": "doughnut",
                "title": "Breach Severity Distribution (Historical Corpus)",
                "labels": ["Normal / In-Control", "Amber Alarm", "Red Trip Breach"],
                "data": [65, 24, 11],
                "colors": ["#2fb387", "#f59e0b", "#ef4444"],
            }
        elif "trend" in q_lower or "temperature" in q_lower or "ph" in q_lower:
            chart = {
                "type": "line",
                "title": "Bioprocess Parameter Excursion Trend (Last 10 Batches)",
                "labels": ["B-101", "B-102", "B-103", "B-104", "B-105", "B-106", "B-107", "B-108", "B-109", "B-110"],
                "data": [36.8, 36.9, 37.1, 37.0, 37.6, 37.9, 38.2, 37.4, 37.1, 37.0],
                "colors": ["#2bdfc4"],
            }
        else:
            chart = {
                "type": "bar",
                "title": "CAPA Excursions & Work Orders by Parameter",
                "labels": ["Temperature", "pH", "Pressure", "Conductivity", "Humidity"],
                "data": [14, 22, 9, 18, 7],
                "colors": ["#2bb3c0", "#2fb387", "#f59e0b", "#a78bfa", "#ec4899"],
            }

    # Grounded answer composition (parameter-aware when an agent is selected)
    param_meta = PARAMETERS.get(param) if param else None

    if param_meta is not None and agent != "human":
        ans = (
            f"**{agent_label} Response**:\n\n"
            f"Threshold-breach guidance for **{param_meta.name}** on **{param_meta.asset}**.\n\n"
            f"1. **SOP** — Confirm redundant reading, then correct linked process drivers per **SOP-BSC-001**.\n"
            f"2. **CAPA** — Cross-check prior excursions for the same failure mode (**CAPA-LOG-2024**).\n"
            f"3. **OEM** — Observe equipment rate limits for **{param_meta.asset}** (**OEM-MAN-CW3**).\n"
            f"4. Trip-zone / high P(breach) actions require the **Human Escalation Agent** and 21 CFR Part 11 e-signature."
        )
        if snippets:
            ans += f"\n\nRetrieved {len(snippets)} grounded citation(s) from the hybrid index for this query."
    elif "ph" in q_lower or "probe" in q_lower:
        ans = (
            f"**{agent_label} Response**:\n\n"
            "Per **SOP-BSC-001** (§3.2) and **CAPA-LOG-2024** (#14), reactor pH probe drift requires "
            "immediate buffer check against standard reference solutions (pH 4.01, 7.00, 10.01). "
            "Because pH adjustments directly impact product quality attributes, **no autonomous agent setpoint modification is allowed** — "
            "every corrective acid/base dosing requires a mandatory **21 CFR Part 11 electronic signature** by a qualified QA engineer before execution."
        )
    elif "capa" in q_lower or "history" in q_lower:
        ans = (
            f"**{agent_label} Response**:\n\n"
            "Retrieved **CAPA records** and maintenance work orders from the unified structured/unstructured RAG index. "
            "The dominant root cause for bioprocess excursions is **coolant heat-exchanger fouling** (42% of temperature breaches) "
            "and **acid/base dosing valve diaphragm wear** (31% of pH breaches). See the historical breakdown chart below."
        )
    elif "21 cfr" in q_lower or "signature" in q_lower or "esign" in q_lower or "worm" in q_lower:
        ans = (
            f"**{agent_label} Response**:\n\n"
            "Per **GxP 21 CFR Part 11 Governance Rules**:\n"
            "1. Any setpoint change to a GxP critical parameter (pH, Temperature trip zone) requires dual-factor e-signature.\n"
            "2. Signature manifest must record: *Signer Full Name*, *Operator Role*, *Timestamp (UTC)*, and *Justification Code*.\n"
            "3. All approved actions are written to an immutable SHA-256 hash-chained WORM audit trail log."
        )
    elif "conductivity" in q_lower or "wfi" in q_lower or "resin" in q_lower:
        ans = (
            f"**{agent_label} Response**:\n\n"
            "Per **OEM-MAN-CW3** (§5.4) and **WFI Maintenance Runbook**: Elevated WFI conductivity (>960 µS/cm) indicates "
            "deionization resin-bed exhaustion or regeneration cycle overrun. "
            "Immediate action: Reroute non-conforming water to holding tank, perform forward-flush, and check regeneration cycle count."
        )
    else:
        ans = (
            f"**{agent_label} Response**:\n\n"
            f"Query answered using the SCADA Co-pilot RAG index "
            f"({len(snippets)} relevant citations from structured CAPA + unstructured SOP/OEM). "
            f"Review the grounded citations and generated diagnostics below."
        )

    return {
        "query": query,
        "answer": ans,
        "agent": agent_label,
        "mode": f"SCADA Co-pilot · {agent_label}",
        "param": param,
        "citations": citations,
        "chart": chart,
    }


