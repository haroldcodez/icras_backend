# ICRAS Backend — FastAPI ML Inference Service

Lightweight Python microservice that serves the trained Gradient Boosting model for credit risk prediction.

## Stack

- **FastAPI** — API framework
- **scikit-learn + joblib** — model loading and inference
- **Uvicorn** — ASGI server

## Structure

```
icras_backend/
├── main.py                        # FastAPI app
├── model/
│   └── advanced_model.pkl         # Trained GradientBoostingClassifier (production)
├── final_models/                  # Model training pipeline
│   ├── 02_train_baseline_model.py # Train baseline (traditional features only)
│   ├── 03_train_advanced_model.py # Train advanced model (traditional + alternative)
│   ├── 04_evaluate_models.py      # Compare models at 5% default rate threshold
│   ├── run_all.py                 # Run full training pipeline
│   ├── engineered_features.csv    # Pre-engineered feature dataset for model training
│   ├── feature_categories.csv     # Feature category labels (traditional/alternative)
│   └── README.md                  # Training pipeline documentation
├── FIELD_GUIDE.md                 # Field definitions (dataset features)
├── requirements.txt
├── .env
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
```

## Run (Start the server)

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Endpoints

### GET `/health`

```json
{ "status": "ok", "model_loaded": true, "model_version": "v1.0" }
```

### POST `/api/predict`

**Sample Request:**

```json
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8080/api/predict" -Method POST -ContentType "application/json" -Body '{
  "AMT_ANNUITY": 24750,
  "goods_price_credit_ratio": 0.92,
  "bureau_credit_types": 2,
  "AMT_CREDIT": 526500,
  "bureau_days_credit_mean": -1097,
  "credit_income_ratio": 3.29,
  "AMT_GOODS_PRICE": 463500,
  "annuity_income_ratio": 0.157,
  "bureau_closed_count": 2,
  "bureau_active_count": 1,
  "cc_limit_mean": 0,
  "cc_utilization": 0.0,
  "prev_approved_ratio": 0.82,
  "FLAG_MOBIL": 0,
  "FLAG_EMP_PHONE": 0,
  "geo_stability": 0,
  "inst_payment_ratio": 1.0,
  "FLAG_OWN_REALTY": 0,
  "inst_dpd_mean": -9.9,
  "FLAG_WORK_PHONE": 0
}' | Select-Object -ExpandProperty Content
```

**Sample Response:**

```json
{
    "model_version": "v1.0",
    "default_probability": 0.0271,
    "recommendation": "APPROVE",
    "risk_level": "Low"
}
```

### Risk Levels

| Probability | Risk Level | Recommendation |
| ----------- | ---------- | -------------- |
| < 0.10      | Low        | APPROVE        |
| 0.10 – 0.20 | Medium     | REJECT         |
| > 0.20      | High       | REJECT         |

> Approval threshold is 0.085 (calibrated for ~5% portfolio default rate).
