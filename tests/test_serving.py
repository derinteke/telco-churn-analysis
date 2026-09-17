from fastapi.testclient import TestClient

from src.serving.predictor import _original_feature
from tests.conftest import requires_model


def test_original_feature_mapping():
    assert _original_feature("cat__Contract_Month-to-month") == "Contract"
    assert _original_feature("cat__StreamingTV_No internet service") == "StreamingTV"
    assert _original_feature("num__tenure") == "tenure"
    assert _original_feature("cat__tenure_group_0-1yr") == "tenure_group"


@requires_model
def test_predictor_contributions_are_consistent():
    """SHAP contributions + bias must reproduce the model's log-odds."""
    import numpy as np
    from src.features import engineer
    from src.serving.predictor import get_predictor, get_store

    p, row = get_predictor(), get_store().get("7590-VHVEG")
    pred = p.predict_frame(row, "7590-VHVEG", top_k=100)
    X = engineer(row).drop(columns=["Churn", "customerID"])
    total = sum(r.impact for r in pred.top_reasons)
    bias = p.clf.get_booster().predict(
        __import__("xgboost").DMatrix(p.prep.transform(X)), pred_contribs=True
    )[0, -1]
    assert np.isclose(1 / (1 + np.exp(-(total + bias))), pred.churn_probability, atol=1e-3)
    assert pred.risk_level in {"low", "medium", "high"}


@requires_model
def test_api_predict_endpoints():
    from api.main import app

    client = TestClient(app)
    r = client.get("/predict/7590-VHVEG")
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1
    assert client.get("/predict/unknown").status_code == 404

    risky = client.post("/predict", json={"Contract": "Month-to-month", "tenure": 1}).json()
    safe = client.post("/predict", json={
        "Contract": "Two year", "tenure": 70, "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 50, "TotalCharges": 3500,
        "OnlineSecurity": "Yes", "TechSupport": "Yes",
    }).json()
    assert risky["churn_probability"] > safe["churn_probability"]
