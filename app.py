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
MODEL_PATH = os.path.join(BASE_DIR, "models", "user_risk_model.pkl")

model = joblib.load(MODEL_PATH)

@app.post("/predict-risk")
def predict_risk(user: dict):

    risk_result = hybrid_risk_engine(user, model)
    funds, clusters = recommend_funds(risk_result["final_risk"])

    return {
        **risk_result,
        "recommended_clusters": clusters,
        "recommended_funds": funds
    }

@app.post("/explain-risk")
def explain_risk(user: dict):

    try:
        # 1️⃣ Get ML-ready input
        X, probs = ml_risk_raw_prediction(model, user)

        # 2️⃣ FORCE numeric dtype (CRITICAL FIX ✅)
        X = X.astype(float)

        # 3️⃣ Create SHAP explainer dynamically (sklearn-safe)
        explainer = shap.Explainer(model, X)

        # 4️⃣ SHAP values
        shap_values = explainer(X)

        # 5️⃣ Predicted class index
        pred_class = int(np.argmax(probs))

        # 6️⃣ SHAP values for predicted class
        class_shap_vals = shap_values.values[0][pred_class]

        explanation = []
        for col, val, shap_val in zip(X.columns, X.iloc[0], class_shap_vals):
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
        )[:5]

        return {
            "ml_predicted_class": pred_class,
            "top_factors": explanation
        }

    except Exception as e:
        return {
            "error": "Explainability failed",
            "details": str(e)
        }


@app.get("/")
def health():
    return {"status": "API running"}
