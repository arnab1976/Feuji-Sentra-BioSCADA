"""
BioSCADA AI — Phase 1: PyFlink streaming job.

Registers the Python UDFs used by the continuous query and submits the
Kafka -> window -> features -> score -> filter -> Kafka pipeline.

This is the *executable* counterpart to sql/01_breach_detection.sql.

Free stack: Apache Flink (Apache-2.0) + PyFlink.

Run locally:
    python breach_detection_job.py
Submit to a cluster:
    flink run -py breach_detection_job.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from pyflink.table import EnvironmentSettings, TableEnvironment, DataTypes
from pyflink.table.udf import udf

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [flink-job] %(message)s",
)
log = logging.getLogger("flink-job")

KAFKA = os.getenv("KAFKA_BOOTSTRAP_INTERNAL", "kafka:29092")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/models"))
WINDOW = os.getenv("FLINK_WINDOW", "10")
BREACH_THRESHOLD = float(os.getenv("BREACH_THRESHOLD", "0.30"))

# --------------------------------------------------------------------------
# Parameter metadata — kept inline so the UDF has no import-time dependency
# on the simulator package when shipped to a Flink cluster.
# --------------------------------------------------------------------------
BANDS: Dict[str, Dict[str, tuple]] = {
    "temp":  {"control": (36.5, 37.5), "alarm": (36.0, 38.0)},
    "ph":    {"control": (6.8, 7.2),   "alarm": (6.6, 7.4)},
    "press": {"control": (100.0, 110.0), "alarm": (96.0, 114.0)},
    "cond":  {"control": (700.0, 900.0), "alarm": (640.0, 960.0)},
    "hum":   {"control": (40.0, 55.0), "alarm": (36.0, 59.0)},
}
PRIMARY_DRIVER = {
    "temp":  "coolant_flow_rate",
    "ph":    "acid_base_dose_rate",
    "press": "filter_dp",
    "cond":  "feed_composition",
    "hum":   "hvac_fan_speed",
}

# Lazily-loaded trained models (Phase 2 artifacts). If absent, we fall back
# to a deterministic analytic score so the pipeline still runs end-to-end.
_MODELS: Dict[str, object] = {}
_MODELS_TRIED = False


def _load_models() -> None:
    global _MODELS_TRIED
    if _MODELS_TRIED:
        return
    _MODELS_TRIED = True
    try:
        import joblib  # noqa: WPS433
        for pid in BANDS:
            f = MODEL_DIR / f"{pid}_model.joblib"
            if f.exists():
                _MODELS[pid] = joblib.load(f)
                log.info("Loaded trained model for '%s'", pid)
    except Exception as exc:  # pragma: no cover - optional path
        log.warning("Model loading skipped (%s); using analytic fallback", exc)


def _analytic_score(param: str, v_avg: float, v_std: float, v_delta: float) -> float:
    """
    Deterministic fallback score in [0,1].

    Rationale: normalized distance from the centre of the control band,
    inflated by within-window volatility. Monotonic and explainable — good
    enough to drive the pipeline before a model is trained, and useful as a
    sanity baseline afterwards.
    """
    band = BANDS.get(param)
    if not band or v_avg is None:
        return 0.0
    lo, hi = band["control"]
    a_lo, a_hi = band["alarm"]
    centre = (lo + hi) / 2.0
    half_alarm = max((a_hi - a_lo) / 2.0, 1e-9)
    deviation = abs(v_avg - centre) / half_alarm          # ~0 in band, ~1 at alarm edge
    volatility = min((v_std or 0.0) / half_alarm, 0.5)
    raw = (deviation - 0.30) / 0.85 + volatility * 0.35
    return float(max(0.0, min(1.0, raw)))


@udf(result_type=DataTypes.DOUBLE())
def pdm_score(param: str, v_avg: float, v_std: float, v_delta: float) -> float:
    """
    In-stream PdM inference — Operator 5 of the plan.

    Uses the trained Phase-2 model when available, else the analytic
    fallback. Returns P(breach within the forecast horizon).
    """
    _load_models()
    model = _MODELS.get(param)
    if model is not None:
        try:
            import numpy as np  # noqa: WPS433
            x = np.array([[v_avg or 0.0, v_std or 0.0, v_delta or 0.0]])
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(x)[0][1])
            return float(max(0.0, min(1.0, model.predict(x)[0])))
        except Exception as exc:  # pragma: no cover
            log.warning("Model inference failed for %s (%s); falling back", param, exc)
    return _analytic_score(param, v_avg, v_std, v_delta)


@udf(result_type=DataTypes.STRING())
def top_driver(param: str, v_avg: float) -> str:
    """
    The "why" carried into the RAG prompt — Operator 5b.

    Returns the dominant independent variable for this parameter. With a
    trained model present we could rank by SHAP; the primary driver mapping
    is the documented default.
    """
    return PRIMARY_DRIVER.get(param, "unknown")


def build_environment() -> TableEnvironment:
    settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(settings)
    cfg = t_env.get_config().get_configuration()
    cfg.set_string("pipeline.name", "bioscada-breach-detection")
    cfg.set_string("parallelism.default", os.getenv("FLINK_PARALLELISM", "2"))
    cfg.set_string("table.exec.source.idle-timeout", "10s")

    # Kafka connector jar (mounted into the image / provided by the cluster)
    jar = os.getenv("FLINK_KAFKA_JAR")
    if jar and Path(jar).exists():
        cfg.set_string("pipeline.jars", f"file://{jar}")
        log.info("Using Kafka connector jar: %s", jar)

    t_env.create_temporary_function("PDM_SCORE", pdm_score)
    t_env.create_temporary_function("TOP_DRIVER", top_driver)
    log.info("Registered UDFs: PDM_SCORE, TOP_DRIVER")
    return t_env


DDL_SOURCE = f"""
CREATE TABLE scada_telemetry (
    `param`    STRING,
    `asset`    STRING,
    `value`    DOUBLE,
    `unit`     STRING,
    `zone`     STRING,
    `batch_id` STRING,
    `molecule` STRING,
    `ts`       TIMESTAMP_LTZ(3),
    WATERMARK FOR `ts` AS `ts` - INTERVAL '5' SECOND
) WITH (
    'connector'                      = 'kafka',
    'topic'                          = 'scada.telemetry',
    'properties.bootstrap.servers'   = '{KAFKA}',
    'properties.group.id'            = 'bioscada-flink',
    'scan.startup.mode'              = 'latest-offset',
    'format'                         = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors'       = 'true'
)
"""

DDL_SINK = f"""
CREATE TABLE breach_events (
    `event_id`     STRING,
    `param`        STRING,
    `asset`        STRING,
    `batch_id`     STRING,
    `window_start` TIMESTAMP_LTZ(3),
    `window_end`   TIMESTAMP_LTZ(3),
    `v_avg`        DOUBLE,
    `v_std`        DOUBLE,
    `v_delta`      DOUBLE,
    `n_rows`       BIGINT,
    `zone`         STRING,
    `p_breach`     DOUBLE,
    `top_driver`   STRING
) WITH (
    'connector'                    = 'kafka',
    'topic'                        = 'breach.events',
    'properties.bootstrap.servers' = '{KAFKA}',
    'format'                       = 'json'
)
"""

CONTINUOUS_QUERY = f"""
INSERT INTO breach_events
SELECT
    CONCAT('evt-', `param`, '-', CAST(UNIX_TIMESTAMP(CAST(window_start AS STRING)) AS STRING)),
    `param`,
    MAX(`asset`),
    MAX(`batch_id`),
    window_start,
    window_end,
    ROUND(AVG(`value`), 4),
    ROUND(STDDEV_POP(`value`), 4),
    ROUND(MAX(`value`) - MIN(`value`), 4),
    COUNT(*),
    MAX(`zone`),
    ROUND(PDM_SCORE(`param`, AVG(`value`), STDDEV_POP(`value`), MAX(`value`) - MIN(`value`)), 4),
    TOP_DRIVER(`param`, AVG(`value`))
FROM TABLE(
    TUMBLE(TABLE scada_telemetry, DESCRIPTOR(`ts`), INTERVAL '{WINDOW}' SECOND)
)
GROUP BY `param`, window_start, window_end
HAVING PDM_SCORE(`param`, AVG(`value`), STDDEV_POP(`value`), MAX(`value`) - MIN(`value`)) > {BREACH_THRESHOLD}
    OR MAX(`zone`) <> 'control'
"""


def main() -> None:
    log.info("Starting BioSCADA breach-detection job (Kafka=%s, window=%ss)", KAFKA, WINDOW)
    t_env = build_environment()
    t_env.execute_sql(DDL_SOURCE)
    t_env.execute_sql(DDL_SINK)
    log.info("Submitting continuous query ...")
    result = t_env.execute_sql(CONTINUOUS_QUERY)
    log.info("Job submitted: %s", result.get_job_client().get_job_id() if result.get_job_client() else "local")
    result.wait()


if __name__ == "__main__":
    main()
