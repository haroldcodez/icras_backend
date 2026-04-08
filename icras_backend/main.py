from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ICRAS Prediction API",
    description="Intelligent Credit Risk Assessment System - ML Inference Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "model/advanced_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    model = None

# Feature order must match training order exactly
FEATURE_ORDER = [
    "AMT_ANNUITY",
    "goods_price_credit_ratio",
    "bureau_credit_types",
    "prev_approved_ratio",
    "AMT_CREDIT",
    "FLAG_MOBIL",
    "bureau_days_credit_mean",
    "FLAG_EMP_PHONE",
    "credit_income_ratio",
    "geo_stability",
    "AMT_GOODS_PRICE",
    "annuity_income_ratio",
    "bureau_closed_count",
    "inst_payment_ratio",
    "FLAG_OWN_REALTY",
    "bureau_active_count",
    "cc_limit_mean",
    "inst_dpd_mean",
    "cc_utilization",
    "FLAG_WORK_PHONE",
]


class PredictRequest(BaseModel):
    AMT_ANNUITY: float
    goods_price_credit_ratio: float
    bureau_credit_types: int
    AMT_CREDIT: float
    bureau_days_credit_mean: float
    credit_income_ratio: float
    AMT_GOODS_PRICE: float
    annuity_income_ratio: float
    bureau_closed_count: int
    bureau_active_count: int
    cc_limit_mean: float
    cc_utilization: float
    prev_approved_ratio: float
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    geo_stability: float
    inst_payment_ratio: float
    FLAG_OWN_REALTY: int
    inst_dpd_mean: float
    FLAG_WORK_PHONE: int


class PredictResponse(BaseModel):
    model_version: str
    default_probability: float
    recommendation: str
    risk_level: str


@app.get("/")
def root():
    return {
        "service": "ICRAS Prediction API",
        "version": MODEL_VERSION,
        "status": "ready" if model is not None else "model not loaded",
        "endpoint": "POST /api/predict"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # These features are stored as negative values in training data (Home Credit convention)
    # We Negate them so the model receives the expected sign for accuratre inference
    req.bureau_days_credit_mean = -abs(req.bureau_days_credit_mean)
    req.inst_dpd_mean = -abs(req.inst_dpd_mean)

    # Build feature vector in correct order
    features = np.array([[getattr(req, f) for f in FEATURE_ORDER]])

    # Get default probability (class 1)
    prob = float(model.predict_proba(features)[0][1])

    # Risk bands
    if prob < 0.10:
        risk_level = "Low"
    elif prob < 0.20:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Recommendation threshold calibrated for ~5% portfolio default rate
    recommendation = "APPROVE" if prob < 0.085 else "REJECT"

    return PredictResponse(
        model_version=MODEL_VERSION,
        default_probability=round(prob, 4),
        recommendation=recommendation,
        risk_level=risk_level,
    )
