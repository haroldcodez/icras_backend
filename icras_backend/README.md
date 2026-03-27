# ICRAS Backend — FastAPI ML Inference Service

Lightweight Python microservice that serves the trained Gradient Boosting model for credit risk prediction.

## Stack

- **FastAPI** — API framework
- **scikit-learn + joblib** — model loading and inference
- **Uvicorn** — ASGI server

## Structure

```
icras_backend/
├── main.py                  # FastAPI app
├── model/
│   └── advanced_model.pkl   # Trained GradientBoostingClassifier
├── requirements.txt
├── .env
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Endpoints

### GET `/health`

```json
{ "status": "ok", "model_loaded": true, "model_version": "v1.0" }
```

### POST `/api/predict`

**Request:**

```json
{
    "AMT_ANNUITY": 9000,
    "goods_price_credit_ratio": 0.82,
    "bureau_credit_types": 4,
    "AMT_CREDIT": 200000,
    "bureau_days_credit_mean": 720,
    "credit_income_ratio": 3.7,
    "AMT_GOODS_PRICE": 164000,
    "annuity_income_ratio": 0.17,
    "bureau_closed_count": 3,
    "bureau_active_count": 2,
    "cc_limit_mean": 8000,
    "cc_utilization": 0.32,
    "prev_approved_ratio": 0.8,
    "FLAG_MOBIL": 1,
    "FLAG_EMP_PHONE": 1,
    "geo_stability": 3,
    "inst_payment_ratio": 0.95,
    "FLAG_OWN_REALTY": 0,
    "inst_dpd_mean": 0,
    "FLAG_WORK_PHONE": 1
}
```

**Response:**

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

## Interactive Docs

With the server running, visit:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc
