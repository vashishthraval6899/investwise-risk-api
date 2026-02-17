# 📈 InvestWiseInvestWise: Supervised Learning & SHAP API
**Focus:** Risk Prediction, Explainable AI, Production API.

# 🧠 InvestWise: Intelligent Risk Engine (API)

![FastAPI](https://img.shields.io/badge/FastAPI-High_Performance-009688)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![SHAP](https://img.shields.io/badge/XAI-Explainable_AI-ff69b4)
![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E)

> **The "Brain" of InvestWise. A Hybrid AI API that predicts Investor Risk and explains the 'Why'.**

## 🚀 System Architecture
This is a production-grade Microservice deployed on **Railway**.

1.  **Input:** User Demographics (Age, Income, Horizon, Goals).
2.  **Model (LightGBM):** Predicts a continuous **Risk Score (0.0 - 1.0)**.
3.  **Explainability (SHAP):** Calculates feature contribution (e.g., *"Age=25 increased risk score by +0.1"*).
4.  **Hybrid Mapping:** Maps the predicted Risk Score to the Fund Clusters (from Repo 3).

## 🔮 Why LightGBM + SHAP?
- **LightGBM:** Chosen over Neural Networks for its superior performance on tabular financial data and ability to capture non-linear relationships.
- **SHAP (TreeExplainer):** Provides "Glass Box" transparency, building trust with retail investors by explaining every recommendation.

## 🔌 API Endpoints

### `POST /predict-risk`
Accepts user profile and returns:
- Calculated Risk Score.
- Recommended Fund Cluster.
- Top 5 Specific Mutual Funds.

### `POST /explain-risk`
Returns SHAP values for visualization:
```json
{
  "feature": "investment_horizon",
  "impact": "+0.42",
  "description": "Long-term horizon increases risk capacity."
}
