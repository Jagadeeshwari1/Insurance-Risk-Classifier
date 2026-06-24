"""Train the insurance risk model end to end.

Pipeline: load -> validate -> split -> preprocess -> fit calibrated classifier
-> evaluate (discrimination + calibration + fairness) -> explain -> persist a
self-contained model bundle, a metrics.json, and a MODEL_CARD.md. Logs to
MLflow when installed; otherwise writes everything locally.

Run:  python -m src.models.train
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import CONFIG
from src.data.validation import validate
from src.features.build_features import build_preprocessor, get_feature_lists
from src.models.evaluate import evaluate_all
from src.models.explain import global_importance
from src.models.model_factory import build_model
from src.utils.logging import get_logger

logger = get_logger("train")
_HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


def load_data() -> pd.DataFrame:
    path = CONFIG.path("raw_data")
    if not path.exists():
        raise FileNotFoundError(
            f"No data at {path}. Generate it: `python -m src.data.make_dataset`."
        )
    return pd.read_csv(path)


def train() -> dict:
    df = validate(load_data())
    numeric, categorical = get_feature_lists()
    feature_cols = numeric + categorical
    target = CONFIG.target
    protected_cols = list(CONFIG.features.protected)

    X = df[feature_cols + protected_cols].copy()
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG.model.test_size,
        stratify=y,
        random_state=CONFIG.random_seed,
    )
    logger.info(
        "Train=%d  Test=%d  claim_rate(train)=%.1f%%",
        len(X_train), len(X_test), 100 * y_train.mean(),
    )

    # Class imbalance ratio for libraries that use scale_pos_weight (xgboost).
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)

    base_model, backend = build_model(pos_weight=pos_weight)
    pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", base_model)]
    )

    # Calibrate so the output is a trustworthy probability, not just a ranking.
    calibrated = CalibratedClassifierCV(
        pipeline,
        method=CONFIG.model.calibration_method,
        cv=CONFIG.model.calibration_cv,
    )
    logger.info("Fitting calibrated model (%s)...", backend)
    # Fit on the model's own feature columns only (protected attrs excluded).
    calibrated.fit(X_train[feature_cols], y_train)

    # ---- Evaluate on held-out test set ----
    y_prob = calibrated.predict_proba(X_test[feature_cols])[:, 1]
    metrics = evaluate_all(y_test, y_prob, protected=X_test[protected_cols])

    importances = global_importance(
        calibrated, X_test[feature_cols], y_test
    )

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    background = X_train[feature_cols].sample(
        min(200, len(X_train)), random_state=CONFIG.random_seed
    ).reset_index(drop=True)

    bundle = {
        "pipeline": calibrated,
        "backend": backend,
        "version": version,
        "feature_columns": feature_cols,
        "protected_columns": protected_cols,
        "risk_tiers": dict(CONFIG.risk_tiers),
        "decision_threshold": CONFIG.decision_threshold,
        "background": background,
        "trained_at": version,
    }

    # ---- Persist artifacts ----
    CONFIG.path("model_dir").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CONFIG.path("model_file"))
    logger.info("Saved model -> %s", CONFIG.path("model_file"))

    report = {
        "version": version,
        "backend": backend,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics,
        "global_importance": importances,
    }
    with open(CONFIG.path("metrics_file"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved metrics -> %s", CONFIG.path("metrics_file"))

    _write_model_card(report)
    _log_mlflow(report)
    return report


def _write_model_card(report: dict) -> None:
    d = report["metrics"]["discrimination"]
    c = report["metrics"]["calibration"]
    fair = report["metrics"].get("fairness", {})
    top = ", ".join(f["feature"] for f in report["global_importance"][:5])

    lines = [
        "# Model Card — Insurance Risk Classifier",
        "",
        f"- **Version:** {report['version']}",
        f"- **Backend:** {report['backend']}",
        f"- **Trained on:** {report['n_train']} rows (held-out test: {report['n_test']})",
        "",
        "## Intended use",
        "Estimates the probability that a policy/applicant produces a significant "
        "claim or loss in the coverage period. The calibrated probability is mapped "
        "to low/medium/high risk tiers for underwriting support. Decisions must "
        "remain human-reviewed; this model is a decision aid, not an autopilot.",
        "",
        "## Performance",
        f"- ROC-AUC: {d['roc_auc']:.3f}",
        f"- PR-AUC: {d['pr_auc']:.3f}",
        f"- Brier score (calibration): {c['brier_score']:.4f}",
        f"- At threshold {d['threshold']}: precision {d['precision']:.3f}, "
        f"recall {d['recall']:.3f}",
        "",
        "## Top drivers",
        f"{top}",
        "",
        "## Fairness",
    ]
    for attr, fm in fair.items():
        di = fm.get("disparate_impact_ratio")
        flag = "⚠️ REVIEW" if fm.get("adverse_impact_flag") else "ok"
        di_str = f"{di:.2f}" if di is not None else "n/a"
        lines.append(f"- `{attr}`: disparate-impact ratio {di_str} ({flag})")
    lines += [
        "",
        "## Limitations",
        "- Trained on synthetic data for demonstration; retrain on governed, "
        "representative production data before any real use.",
        "- Protected attributes are excluded from inputs but monitored for "
        "disparate impact; proxy bias via correlated features is still possible.",
        "- Calibration and fairness must be re-checked on each retrain and "
        "monitored for drift in production.",
    ]
    with open(CONFIG.path("model_card"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Saved model card -> %s", CONFIG.path("model_card"))


def _log_mlflow(report: dict) -> None:
    if not _HAS_MLFLOW:
        logger.info("MLflow not installed — skipping experiment logging.")
        return
    import mlflow  # pragma: no cover

    mlflow.set_experiment("insurance-risk-classifier")
    with mlflow.start_run(run_name=report["version"]):
        d = report["metrics"]["discrimination"]
        c = report["metrics"]["calibration"]
        mlflow.log_params({"backend": report["backend"], **CONFIG.model.params})
        mlflow.log_metrics(
            {
                "roc_auc": d["roc_auc"],
                "pr_auc": d["pr_auc"],
                "brier_score": c["brier_score"],
                "precision": d["precision"],
                "recall": d["recall"],
            }
        )
        mlflow.log_artifact(str(CONFIG.path("model_file")))
        mlflow.log_artifact(str(CONFIG.path("metrics_file")))


if __name__ == "__main__":
    train()
