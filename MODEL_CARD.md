# Model Card — Insurance Risk Classifier

- **Version:** 20260623-221144
- **Backend:** sklearn-histgbm
- **Trained on:** 16000 rows (held-out test: 4000)

## Intended use
Estimates the probability that a policy/applicant produces a significant claim or loss in the coverage period. The calibrated probability is mapped to low/medium/high risk tiers for underwriting support. Decisions must remain human-reviewed; this model is a decision aid, not an autopilot.

## Performance
- ROC-AUC: 0.768
- PR-AUC: 0.427
- Brier score (calibration): 0.1117
- At threshold 0.4: precision 0.590, recall 0.246

## Top drivers
smoker, age, chronic_conditions, occupation_risk, bmi

## Fairness
- `sex`: disparate-impact ratio 0.98 (ok)

## Limitations
- Trained on synthetic data for demonstration; retrain on governed, representative production data before any real use.
- Protected attributes are excluded from inputs but monitored for disparate impact; proxy bias via correlated features is still possible.
- Calibration and fairness must be re-checked on each retrain and monitored for drift in production.
