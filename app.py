import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

from logic.hybrid_engine import hybrid_risk_engine
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


@app.get("/")
def health():
    return {"status": "API running"}
