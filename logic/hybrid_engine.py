import pandas as pd
import numpy as np
from logic.rule_engine import rule_based_risk_score

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
    df = pd.DataFrame([user])
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
