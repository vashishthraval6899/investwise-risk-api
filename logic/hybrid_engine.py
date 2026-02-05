import pandas as pd
import numpy as np
from logic.rule_engine import rule_based_risk_score

def build_model_features(user: dict) -> pd.DataFrame:
    """
    Maps frontend user inputs to model-trained feature space.
    This is the single source of truth for ML + SHAP.
    """

    # ---------- MAPPINGS ----------
    # investment_duration → horizon
    horizon = user["investment_duration"]

    # risk_appetite (numeric) → risk_tolerance (categorical)
    risk_tolerance_map = {0: "low", 1: "medium", 2: "high"}
    risk_tolerance = risk_tolerance_map.get(user["risk_appetite"], "medium")

    # liquidity_needs → emergency_fund
    emergency_fund = "no" if user["liquidity_needs"] == 1 else "yes"

    # expected_returns → market_exp (proxy mapping)
    if user["expected_returns"] >= 15:
        market_exp = "advanced"
    elif user["expected_returns"] >= 10:
        market_exp = "intermediate"
    else:
        market_exp = "beginner"

    # job_stability (frontend doesn’t collect it yet)
    job_stability = "stable"   # sensible default

    # income (frontend doesn’t collect it yet)
    income = "medium"          # sensible default

    # ---------- FINAL FEATURE DICT ----------
    features = {
        "age": user["age"],
        "horizon": horizon,
        "risk_tolerance": risk_tolerance,
        "emergency_fund": emergency_fund,
        "market_exp": market_exp,
        "job_stability": job_stability,
        "income": income,
    }

    return pd.DataFrame([features])


def preprocess_user_input(user: dict) -> pd.DataFrame:
    """
    Safe preprocessing wrapper for ML + SHAP.
    Does NOT change existing behavior.
    """
    return pd.DataFrame([user])

RISK_TO_NUM = {"low": 0, "medium": 1, "high": 2}
NUM_TO_RISK = {v: k for k, v in RISK_TO_NUM.items()}

CONF_THRESHOLD = 0.65


def rule_risk_label(user):
    score = rule_based_risk_score(user)

    if score >= 75:
        return "high", score
    elif score >= 45:
        return "medium", score
    else:
        return "low", score


def ml_risk_label(model, user):
    df = build_model_features(user)
    probs = model.predict_proba(df)[0]
    pred = np.argmax(probs)
    return NUM_TO_RISK[pred], float(np.max(probs))


def final_decision(rule_label, rule_score, ml_label, ml_conf):

    r, m = RISK_TO_NUM[rule_label], RISK_TO_NUM[ml_label]

    if r == m:
        return rule_label, "rule=ml"

    if ml_conf < CONF_THRESHOLD:
        return rule_label, "ml_low_confidence"

    if rule_label == "low" and ml_label == "high":
        return "medium", "safety_override"

    if abs(r - m) == 1:
        return ml_label, "ml_refined"

    return rule_label, "rule_fallback"


def hybrid_risk_engine(user, model):
    rule_label, rule_score = rule_risk_label(user)
    ml_label, ml_conf = ml_risk_label(model, user)

    final_label, decision_source = final_decision(
        rule_label, rule_score, ml_label, ml_conf
    )

    return {
        "final_risk": final_label,
        "risk_score": rule_score,
        "rule_label": rule_label,
        "ml_label": ml_label,
        "ml_confidence": round(ml_conf, 3),
        "decision_source": decision_source,
    }

def ml_risk_raw_prediction(model, user: dict):
    """
    Used ONLY for explainability.
    Returns:
    - DataFrame input
    - Raw probability output
    """
    df = build_model_features(user)
    probs = model.predict_proba(df)[0]
    return df, probs
