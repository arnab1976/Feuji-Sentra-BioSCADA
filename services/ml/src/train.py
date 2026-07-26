"""
BioSCADA AI — Phase 2: Predictive modelling.

Trains one model per SCADA parameter. Each parameter is the DEPENDENT
variable; its PdM features are the INDEPENDENT variables.

Target definition
-----------------
We predict P(breach within the forecast horizon): a binary label that is 1
when the parameter leaves its control band at any point within the next
`horizon` samples. This is *pre-breach* forecasting, not breach detection —
the label looks forward, the features do not.

Models (all free / OSS):
    temp, press -> Gradient Boosting     (sklearn / xgboost)
    ph          -> Random Forest
    cond        -> SVM (RBF, calibrated)
    hum         -> ANN (MLP)

Artifacts: joblib model + metrics JSON + SHAP feature importance,
optionally tracked in MLflow.

Usage:
    python train.py --generate 40000       # synthesize + train
    python train.py --data data/telemetry.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [train] %(message)s",
)
log = logging.getLogger("train")

# make shared parameter defs importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services/simulator/src"))
from parameters import PARAMETERS, PARAM_IDS, Parameter  # noqa: E402

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
HORIZON = int(os.getenv("FORECAST_HORIZON", "15"))
RANDOM_STATE = 42


# =====================================================================
# Data generation (stand-in for historian export)
# =====================================================================
def synthesize(param: Parameter, n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a physically-plausible run for one parameter, including
    excursions, with independent variables correlated to the deviation.

    In production, replace this with a historian / Hudi / BigQuery read of
    3-5 real batches. The downstream code is identical.
    """
    rng = np.random.default_rng(seed)
    centre = param.control_center
    half_alarm = (param.alarm[1] - param.alarm[0]) / 2

    values = np.empty(n)
    v = param.baseline
    excursion_left = 0
    target = param.baseline

    for i in range(n):
        if excursion_left <= 0 and rng.random() < 0.004:
            # start an excursion
            direction = 1 if rng.random() > 0.5 else -1
            magnitude = rng.uniform(0.6, 1.6) * half_alarm
            target = centre + direction * magnitude
            excursion_left = int(rng.integers(25, 70))
        if excursion_left > 0:
            excursion_left -= 1
        else:
            target = param.baseline + np.sin(i / 120.0) * param.noise * 1.5

        v += (target - v) * 0.15 + rng.normal(0, param.noise)
        values[i] = v

    deviation = (values - centre) / half_alarm  # normalized, ~[-1.5, 1.5]

    # independent variables correlated with deviation (+ their own noise)
    feats: Dict[str, np.ndarray] = {}
    weights = [-1.0, 0.9, 0.75, 0.85, 0.45]     # signed influence per feature
    scales = [18.0, 4.0, 9.0, 7.0, 6.0]
    bases = [120.0, 32.0, 28.0, 24.0, 48.0]
    for j, fname in enumerate(param.features):
        w, s, b = weights[j % 5], scales[j % 5], bases[j % 5]
        feats[fname] = b + deviation * w * s + rng.normal(0, s * 0.18, n)

    df = pd.DataFrame(feats)
    df["value"] = values
    df["param"] = param.id
    return df


# =====================================================================
# Feature engineering — must mirror what Flink computes in-stream
# =====================================================================
def build_windowed_features(df: pd.DataFrame, param: Parameter,
                            window: int = 10, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Reproduce Flink's windowed aggregates (v_avg / v_std / v_delta) plus the
    raw independent variables, and attach the forward-looking breach label.

    Keeping this identical to the Flink SQL aggregation is what prevents
    training/serving skew.
    """
    out = pd.DataFrame(index=df.index)
    roll = df["value"].rolling(window, min_periods=window)
    out["v_avg"] = roll.mean()
    out["v_std"] = roll.std(ddof=0)
    out["v_delta"] = roll.max() - roll.min()

    for f in param.features:
        out[f] = df[f].rolling(window, min_periods=window).mean()

    # forward-looking label: does it leave the control band in the next `horizon`?
    lo, hi = param.control
    breach_now = (df["value"] < lo) | (df["value"] > hi)
    # shift(-1) so the current row never sees its own breach
    out["label"] = (
        breach_now.shift(-1).rolling(horizon, min_periods=1).max()
        .shift(-(horizon - 1)).fillna(0).astype(int)
    )
    return out.dropna()


# =====================================================================
# Model zoo
# =====================================================================
def make_model(kind: str):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "gbm":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=220, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        except ImportError:
            log.info("xgboost unavailable -> sklearn GradientBoosting")
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.08,
                random_state=RANDOM_STATE,
            )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        )
    if kind == "svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(C=2.0, gamma="scale", probability=True,
                        random_state=RANDOM_STATE, class_weight="balanced")),
        ])
    if kind == "ann":
        return Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                                  early_stopping=True, random_state=RANDOM_STATE)),
        ])
    raise ValueError(f"unknown model kind: {kind}")


@dataclass
class TrainResult:
    param_id: str
    model_kind: str
    auc: float
    accuracy: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    positive_rate: float
    feature_importance: Dict[str, float]
    model_path: str


def train_parameter(param: Parameter, df: pd.DataFrame,
                    use_mlflow: bool = False) -> Optional[TrainResult]:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                 precision_score, recall_score)
    import joblib

    feats = build_windowed_features(df, param)
    if feats["label"].nunique() < 2:
        log.warning("[%s] only one class present — skipping", param.id)
        return None

    # Flink scores on (v_avg, v_std, v_delta); we train the deployable model on
    # exactly those so the in-stream UDF and the offline model agree.
    serving_cols = ["v_avg", "v_std", "v_delta"]
    X = feats[serving_cols]
    y = feats["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = make_model(param.model)
    model.fit(X_tr, y_tr)

    proba = (model.predict_proba(X_te)[:, 1]
             if hasattr(model, "predict_proba") else model.decision_function(X_te))
    pred = (proba >= 0.5).astype(int)

    res = TrainResult(
        param_id=param.id,
        model_kind=param.model,
        auc=float(roc_auc_score(y_te, proba)),
        accuracy=float(accuracy_score(y_te, pred)),
        precision=float(precision_score(y_te, pred, zero_division=0)),
        recall=float(recall_score(y_te, pred, zero_division=0)),
        n_train=len(X_tr),
        n_test=len(X_te),
        positive_rate=float(y.mean()),
        feature_importance=explain(model, X_te, serving_cols, param),
        model_path=str(MODEL_DIR / f"{param.id}_model.joblib"),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, res.model_path)
    log.info("[%s] %s  AUC=%.3f  acc=%.3f  prec=%.3f  rec=%.3f  (pos rate %.1f%%)",
             param.id, param.model, res.auc, res.accuracy, res.precision,
             res.recall, res.positive_rate * 100)

    if use_mlflow:
        _log_mlflow(param, model, res)
    return res


def explain(model, X_test: pd.DataFrame, cols: List[str],
            param: Parameter) -> Dict[str, float]:
    """
    Feature importance for the 'why'. SHAP when available (tree models),
    else the model's native importances, else permutation importance.
    """
    try:
        import shap
        expl = shap.Explainer(model, X_test)
        vals = expl(X_test[:200]).values
        if vals.ndim == 3:            # (n, features, classes)
            vals = vals[:, :, -1]
        imp = np.abs(vals).mean(axis=0)
    except Exception:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        else:
            try:
                from sklearn.inspection import permutation_importance
                r = permutation_importance(model, X_test[:400],
                                           model.predict(X_test[:400]),
                                           n_repeats=4, random_state=RANDOM_STATE)
                imp = r.importances_mean
            except Exception:
                imp = np.ones(len(cols))
    imp = np.asarray(imp, dtype=float)
    total = imp.sum() or 1.0
    return {c: round(float(v / total), 4) for c, v in zip(cols, imp)}


def _log_mlflow(param: Parameter, model, res: TrainResult) -> None:
    try:
        import mlflow, mlflow.sklearn
        mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))
        mlflow.set_experiment("bioscada-pdm")
        with mlflow.start_run(run_name=f"{param.id}-{param.model}"):
            mlflow.log_params({"parameter": param.id, "model": param.model,
                               "horizon": HORIZON})
            mlflow.log_metrics({"auc": res.auc, "accuracy": res.accuracy,
                                "precision": res.precision, "recall": res.recall})
            mlflow.sklearn.log_model(model, "model")
        log.info("[%s] logged to MLflow", param.id)
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    ap = argparse.ArgumentParser(description="BioSCADA Phase-2 training")
    ap.add_argument("--generate", type=int, default=30000,
                    help="synthesize N samples per parameter")
    ap.add_argument("--data", type=str, default=None,
                    help="parquet/csv of real historian data")
    ap.add_argument("--mlflow", action="store_true", help="log runs to MLflow")
    ap.add_argument("--params", nargs="*", default=PARAM_IDS)
    args = ap.parse_args()

    results: List[TrainResult] = []
    for pid in args.params:
        param = PARAMETERS[pid]
        if args.data:
            full = (pd.read_parquet(args.data) if args.data.endswith(".parquet")
                    else pd.read_csv(args.data))
            df = full[full["param"] == pid].reset_index(drop=True)
            log.info("[%s] loaded %d rows from %s", pid, len(df), args.data)
        else:
            df = synthesize(param, args.generate)
            log.info("[%s] synthesized %d rows", pid, len(df))
        r = train_parameter(param, df, use_mlflow=args.mlflow)
        if r:
            results.append(r)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(
        {r.param_id: r.__dict__ for r in results}, indent=2))

    print("\n" + "=" * 74)
    print(f"{'PARAMETER':<16}{'MODEL':<16}{'AUC':>7}{'ACC':>8}{'PREC':>8}{'REC':>8}{'POS%':>8}")
    print("-" * 74)
    for r in results:
        print(f"{r.param_id:<16}{r.model_kind:<16}{r.auc:>7.3f}{r.accuracy:>8.3f}"
              f"{r.precision:>8.3f}{r.recall:>8.3f}{r.positive_rate*100:>7.1f}%")
    print("=" * 74)
    print(f"Artifacts -> {MODEL_DIR}\nMetrics   -> {metrics_path}\n")


if __name__ == "__main__":
    main()
