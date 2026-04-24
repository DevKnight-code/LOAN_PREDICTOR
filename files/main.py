"""
FastAPI Loan Defaulter Prediction App — Enhanced
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
import json
import os

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model_meta.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        "model.pkl / scaler.pkl not found.\n"
        "Please run:  python train_model.py  first."
    )

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

model_meta = {"model_type": "Ensemble", "accuracy": "N/A", "roc_auc": "N/A"}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        model_meta = json.load(f)

# ── FastAPI setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Loan Defaulter Predictor")

static_path = os.path.join(BASE_DIR, "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates_path = os.path.join(BASE_DIR, "templates")
os.makedirs(templates_path, exist_ok=True)
templates = Jinja2Templates(directory=templates_path)

# ── Encoding maps (must match train_model.py) ──────────────────────────────────
EDU_MAP = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
HOUSING_MAP = {
    "Mortgage": (0, 0),
    "Own":      (1, 0),
    "Rent":     (0, 1),
}

FEATURE_COLS = [
    "Age", "Income", "Loan_Amount", "Credit_Score",
    "Employment_Years", "Education_Level",
    "Housing_Status_Own", "Housing_Status_Rent",
]

def _risk_level(prob: float) -> str:
    if prob < 25:   return "Low"
    if prob < 50:   return "Moderate"
    if prob < 75:   return "High"
    return "Critical"

def _risk_color(prob: float) -> str:
    if prob < 25:   return "#22c55e"
    if prob < 50:   return "#f59e0b"
    if prob < 75:   return "#f97316"
    return "#ef4444"

def _build_features(age, income, loan_amount, credit_score,
                    employment_years, education_level, housing_status):
    edu_encoded = EDU_MAP.get(education_level, 1)
    housing_own, housing_rent = HOUSING_MAP.get(housing_status, (0, 0))
    return pd.DataFrame([[
        age, income, loan_amount, credit_score,
        employment_years, edu_encoded, housing_own, housing_rent
    ]], columns=FEATURE_COLS)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_meta": model_meta,
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float               = Form(...),
    income: float            = Form(...),
    loan_amount: float       = Form(...),
    credit_score: float      = Form(...),
    employment_years: float  = Form(...),
    education_level: str     = Form(...),
    housing_status: str      = Form(...),
):
    features_df     = _build_features(age, income, loan_amount, credit_score,
                                      employment_years, education_level, housing_status)
    features_scaled = scaler.transform(features_df)
    prediction      = model.predict(features_scaled)[0]
    proba           = model.predict_proba(features_scaled)[0]

    default_prob = round(float(proba[1]) * 100, 1)
    safe_prob    = round(float(proba[0]) * 100, 1)

    # Debt-to-income ratio insight
    dti = round((loan_amount / income) * 100, 1) if income > 0 else 0

    result = {
        "is_defaulter"  : bool(prediction),
        "default_prob"  : default_prob,
        "safe_prob"     : safe_prob,
        "label"         : "LIKELY DEFAULTER" if prediction else "CREDITWORTHY",
        "risk_level"    : _risk_level(default_prob),
        "risk_color"    : _risk_color(default_prob),
        "dti"           : dti,
    }

    form_data = {
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "employment_years": employment_years,
        "education_level": education_level, "housing_status": housing_status,
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "form_data": form_data,
        "model_meta": model_meta,
    })

# ── REST API ───────────────────────────────────────────────────────────────────
from pydantic import BaseModel

class CustomerData(BaseModel):
    age: float
    income: float
    loan_amount: float
    credit_score: float
    employment_years: float
    education_level: str
    housing_status: str

@app.post("/api/predict")
async def api_predict(data: CustomerData):
    features_df     = _build_features(data.age, data.income, data.loan_amount,
                                      data.credit_score, data.employment_years,
                                      data.education_level, data.housing_status)
    features_scaled = scaler.transform(features_df)
    prediction      = int(model.predict(features_scaled)[0])
    proba           = model.predict_proba(features_scaled)[0]
    default_prob    = round(float(proba[1]) * 100, 1)

    return {
        "prediction"          : prediction,
        "is_defaulter"        : bool(prediction),
        "default_probability" : default_prob,
        "safe_probability"    : round(float(proba[0]) * 100, 1),
        "risk_level"          : _risk_level(default_prob),
        "model"               : model_meta.get("model_type", "Ensemble"),
    }

@app.get("/api/model-info")
async def model_info():
    return model_meta
