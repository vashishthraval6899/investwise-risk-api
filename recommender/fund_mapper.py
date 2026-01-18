RISK_CLUSTER_MAP = {
    "low": [1],
    "medium": [0, 1],
    "high": [2, 0]
}

FUNDS_BY_CLUSTER = {
    0: ["Fund A", "Fund B", "Fund C"],
    1: ["Fund D", "Fund E"],
    2: ["Fund F", "Fund G", "Fund H"]
}


def recommend_funds(risk_level):
    clusters = RISK_CLUSTER_MAP[risk_level]
    funds = []

    for c in clusters:
        funds.extend(FUNDS_BY_CLUSTER.get(c, []))

    return list(set(funds)), clusters
