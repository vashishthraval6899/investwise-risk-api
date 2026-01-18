def rule_based_risk_score(user):
    score = 0

    # Age
    if user["age"] < 30:
        score += 25
    elif user["age"] <= 45:
        score += 20
    elif user["age"] <= 60:
        score += 10
    else:
        score += 5

    # Investment Horizon
    if user["horizon"] > 10:
        score += 25
    elif user["horizon"] >= 5:
        score += 20
    elif user["horizon"] >= 3:
        score += 10
    else:
        score += 5

    score += {"high": 20, "medium": 12, "low": 5}[user["risk_tolerance"]]
    score += 10 if user["emergency_fund"] == "yes" else 0
    score += {"advanced": 10, "intermediate": 6, "beginner": 2}[user["market_exp"]]

    if user["job_stability"] == "unstable":
        score -= 5

    return score
