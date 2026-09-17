from src.agent.campaigns import evaluate

BASE = {
    "Contract": "Month-to-month", "tenure": 2, "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "TechSupport": "No", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
}
DRIVERS = [
    {"feature": "Contract", "value": "Month-to-month", "impact": 0.58},
    {"feature": "PaymentMethod", "value": "Electronic check", "impact": 0.21},
    {"feature": "OnlineSecurity", "value": "No", "impact": 0.21},
    {"feature": "InternetService", "value": "Fiber optic", "impact": 0.22},
]


def codes(entries):
    return [e["code"] for e in entries]


def test_numeric_eligibility_rules():
    res = evaluate(BASE, DRIVERS)
    not_eligible = codes(res["not_eligible"])
    assert "RET-LOCK24" in not_eligible        # tenure 2 < 3
    assert "RET-FIBERVALUE" in not_eligible    # $70.7 is not above $80

    res = evaluate({**BASE, "tenure": 5, "MonthlyCharges": 95.0}, DRIVERS)
    eligible = codes(res["recommended"] + res["other_eligible"])
    assert {"RET-LOCK24", "RET-FIBERVALUE"} <= set(eligible)


def test_stacking_rules():
    res = evaluate({**BASE, "tenure": 5}, DRIVERS)
    rec = codes(res["recommended"])
    assert len(rec) <= 2
    assert not {"RET-LOCK24", "RET-LOCK12"} <= set(rec)


def test_ranking_follows_drivers():
    rec = evaluate(BASE, DRIVERS)["recommended"]
    assert rec[0]["code"] == "RET-LOCK12"      # contract is the strongest driver
    assert "Contract" in rec[0]["addresses_drivers"]


def test_no_offer_when_nothing_drives_risk():
    loyal = {**BASE, "Contract": "Two year", "tenure": 60, "OnlineSecurity": "Yes",
             "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)"}
    negative = [{"feature": "Contract", "value": "Two year", "impact": -1.2}]
    assert evaluate(loyal, negative)["recommended"] == []
