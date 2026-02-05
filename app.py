import os
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
try:
    explainer = shap.TreeExplainer(model)
except Exception as e:
    explainer = None
    print("SHAP explainer init failed:", e)


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

    # Safety check
    if explainer is None:
        return {
            "error": "Explainability not available",
            "message": "SHAP explainer failed to initialize"
        }

    try:
        # ML-only preprocessing
        X, probs = ml_risk_raw_prediction(model, user)

        # Predict class index
        pred_class = int(np.argmax(probs))

        # SHAP explanation
        shap_values = explainer.shap_values(X)

        # For multiclass, select predicted class
        shap_vals = shap_values[pred_class][0]

        explanation = []
        for col, val, shap_val in zip(X.columns, X.iloc[0], shap_vals):
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
