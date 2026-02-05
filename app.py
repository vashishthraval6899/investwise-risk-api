import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import shap
import pandas as pd

from logic.hybrid_engine import (
    hybrid_risk_engine,
    preprocess_user_input,
    ml_risk_raw_prediction
)
from recommender.fund_mapper import recommend_funds

app = FastAPI(title="Mutual Fund Risk Profiling API")

# Enable CORS for Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_MODEL_PATH = os.path.join(BASE_DIR, "models", "user_risk_model.pkl")
NEW_MODEL_PATH = os.path.join(BASE_DIR, "models", "new_user_risk_lgbm.pkl")

old_model = joblib.load(OLD_MODEL_PATH)
risk_model = joblib.load(NEW_MODEL_PATH)

@app.post("/predict-risk")
def predict_risk(user: dict):

    X = pd.DataFrame([{
        "age": user["age"],
        "risk_appetite": user["risk_appetite"],
        "investment_duration": user["investment_duration"],
        "liquidity_needs": user["liquidity_needs"],
        "expected_returns": user["expected_returns"]
    }])

    risk_score = float(risk_model.predict(X)[0])

    funds, clusters = recommend_funds(risk_score)

    return {
        "risk_score": round(risk_score, 3),
        "recommended_clusters": clusters,
        "recommended_funds": funds
    }


@app.post("/explain-risk")
def explain_risk(user: dict):

    X = pd.DataFrame([{
        "age": user["age"],
        "risk_appetite": user["risk_appetite"],
        "investment_duration": user["investment_duration"],
        "liquidity_needs": user["liquidity_needs"],
        "expected_returns": user["expected_returns"]
    }])

    explainer = shap.TreeExplainer(risk_model)
    shap_values = explainer.shap_values(X)[0]

    explanation = []
    for col, val, shap_val in zip(X.columns, X.iloc[0], shap_values):
        explanation.append({
            "feature": col,
            "value": float(val),
            "impact": round(float(shap_val), 4),
            "effect": "increase" if shap_val > 0 else "decrease"
        })

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return {
        "risk_score": round(float(risk_model.predict(X)[0]), 3),
        "top_factors": explanation[:3]
    }


@app.get("/")
def health():
    return {"status": "API running"}
