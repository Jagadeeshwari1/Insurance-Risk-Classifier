## Insurance Risk Classifier

A full-stack ML pipeline that scores insurance applicants/policies for claim
risk. It outputs a **calibrated probability**, maps it to a **low/medium/high
risk tier**, returns a **per-decision explanation**, checks **fairness**, and
monitors **input drift** — the pieces an insurance ML system needs beyond a
plain classifier.

> **Status:** Phase 1 (data → features → training → serving API) is complete and
> runnable. Phase 2 (React frontend + DB persistence) is next.

---

## Architecture

```
                ┌──────────────┐
  applicant ──▶ │ React UI     │  (Phase 2)
                └──────┬───────┘
                       │ HTTP (JSON)
                ┌──────▼───────┐
                │ FastAPI      │  validate → infer → explain → audit
                │  /predict    │
                └──────┬───────┘
                       │ loads
                ┌──────▼───────────────────────────────────┐
                │ Model bundle (joblib)                     │
                │  preprocess → calibrated gradient boost   │
                │  + background sample for explanations     │
                └──────▲───────────────────────────────────┘
                       │ produced by
   data ──▶ validate ──▶ features ──▶ train ──▶ evaluate (AUC, calibration,
                                                 fairness) + SHAP + model card
                       │
                ┌──────▼───────┐
                │ Drift monitor │  PSI on production inputs (Phase: ongoing)
                └──────────────┘
```

## Project structure

```
insurance-risk-classifier/
├── config/config.yaml          # features, target, tiers, model + fairness params
├── src/
│   ├── config.py               # config loader (yaml + safe defaults)
│   ├── data/
│   │   ├── make_dataset.py      # synthetic insurance data generator
│   │   └── validation.py        # schema + range checks
│   ├── features/build_features.py   # shared ColumnTransformer (no train/serve skew)
│   ├── models/
│   │   ├── model_factory.py     # xgboost/lightgbm/catboost or sklearn fallback
│   │   ├── train.py             # orchestration: fit → evaluate → persist bundle
│   │   ├── evaluate.py          # discrimination + calibration + fairness
│   │   ├── explain.py           # SHAP (or permutation/proxy fallback)
│   │   └── predict.py           # inference: probability → tier → factors
│   ├── monitoring/drift.py      # Population Stability Index
│   └── utils/logging.py
├── api/
│   ├── main.py                  # FastAPI app (+ CORS for the frontend)
│   ├── schemas.py               # Pydantic request/response models
│   ├── dependencies.py
│   └── routes/{health,predict}.py
├── tests/test_pipeline.py
├── docker/{Dockerfile.api,docker-compose.yml}
├── .github/workflows/ci.yml
├── Makefile
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt   # or: make install

make data      # generate synthetic dataset  -> data/raw/applications.csv
make train     # train + evaluate            -> models/risk_model.joblib, metrics.json, MODEL_CARD.md
make serve     # run API                     -> http://localhost:8000/docs
make test      # run tests
```

The pipeline runs on **pandas + scikit-learn alone**. Installing XGBoost, SHAP,
and MLflow upgrades the model backend, explanations, and experiment tracking
automatically — no code changes needed.

### Example request

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 62, "annual_income": 48000, "bmi": 34.0, "num_dependents": 2,
  "prior_claims_count": 4, "years_as_customer": 1.0, "credit_score": 580,
  "chronic_conditions": 3, "smoker": "yes", "region": "southeast",
  "occupation_risk": "high", "exercise_frequency": "never"
}'
```

```json
{
  "probability": 0.9167,
  "risk_tier": "high",
  "flagged_high_risk": true,
  "decision_threshold": 0.4,
  "model_version": "20260623-221144",
  "top_factors": [
    {"feature": "smoker", "value": "yes", "direction": "increases_risk", "contribution": 0.068},
    {"feature": "age", "value": 62, "direction": "increases_risk", "contribution": 0.063}
  ]
}
```

### API endpoints

| Method | Path             | Purpose                              |
|--------|------------------|--------------------------------------|
| GET    | `/health`        | Liveness + model-loaded flag         |
| GET    | `/model-info`    | Version, backend, schema, thresholds |
| POST   | `/predict`       | Score one applicant (with reasons)   |
| POST   | `/predict/batch` | Score many applicants                |

## Why this isn't a generic classifier (insurance-specific design)

- **Calibration.** Underwriting prices off a *true* probability, so the model is
  wrapped in `CalibratedClassifierCV` and judged on Brier score, not only AUC.
- **Explainability.** Every prediction returns the top factors (SHAP in
  production) — needed for adverse-action notices and regulatory review.
- **Fairness.** Protected attributes (e.g. `sex`) are excluded from inputs but
  monitored for disparate impact via the four-fifths ratio; training fails loud
  if impact is flagged.
- **Governance.** Each run writes a versioned model bundle, a `metrics.json`,
  and a `MODEL_CARD.md` (intended use, performance, fairness, limitations).
- **Class imbalance.** Handled per-backend (`scale_pos_weight` / balanced
  class weights) since high-risk cases are rare.
- **Drift monitoring.** PSI on production inputs to trigger retraining.

## Roadmap

- **Phase 2 — Frontend:** React dashboard (applicant form, risk result with the
  factor breakdown, model-performance + drift views), wired to the API; add the
  `frontend` service to `docker-compose.yml`.
- **Persistence:** store applications + decisions in Postgres for the audit
  trail (schema hook already in `_audit`).
- **Auth:** JWT on the API before any real deployment.
- **MLOps:** schedule retraining + drift reports (Airflow/Prefect); register
  models in the MLflow Model Registry.

> Trained on **synthetic** data for demonstration. Retrain on governed,
> representative data and complete fairness/compliance review before real use.
