"""Score a single customer and explain the score with per-feature contributions.

Contributions come from the booster's native SHAP implementation
(``pred_contribs`` in XGBoost, ``pred_contrib`` in LightGBM), so no extra
explainer object is needed at serving time. One-hot columns are folded back
into their original feature ("Contract_Month-to-month" -> "Contract") so the
LLM receives human-readable reasons.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .. import config
from ..data import load_clean
from ..features import engineer


@dataclass
class Reason:
    feature: str
    value: str
    impact: float  # log-odds contribution; > 0 pushes towards churn

    def to_dict(self) -> dict:
        return {"feature": self.feature, "value": self.value, "impact": round(self.impact, 3)}


@dataclass
class Prediction:
    customer_id: str
    churn_probability: float
    risk_level: str
    threshold: float
    top_reasons: list[Reason] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "churn_probability": round(self.churn_probability, 4),
            "risk_level": self.risk_level,
            "threshold": self.threshold,
            "top_reasons": [r.to_dict() for r in self.top_reasons],
        }


def _original_feature(encoded_name: str) -> str:
    """Map a ColumnTransformer output name back to its source column."""
    name = encoded_name.split("__", 1)[-1]
    if name in config.ALL_NUMERIC:
        return name
    # Longest match first so e.g. "StreamingTV" isn't swallowed by a shorter prefix
    for col in sorted(config.ALL_CATEGORICAL, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


class ChurnPredictor:
    def __init__(self, model_path=None, metrics_path=None):
        model_path = model_path or config.MODELS_DIR / "best_model.joblib"
        metrics_path = metrics_path or config.MODELS_DIR / "metrics.json"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found. Run `python train.py` first.")
        self.pipeline = joblib.load(model_path)
        self.prep = self.pipeline.named_steps["prep"]
        self.clf = self.pipeline.named_steps["clf"]
        self.encoded_names = list(self.prep.get_feature_names_out())

        self.threshold = 0.5
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.threshold = json.load(f).get("_best_f1_threshold", 0.5)

    # --- risk ----------------------------------------------------------------
    def risk_level(self, proba: float) -> str:
        if proba >= self.threshold:
            return "high"
        if proba >= self.threshold * 0.6:
            return "medium"
        return "low"

    # --- contributions -------------------------------------------------------
    def _contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-encoded-feature log-odds contributions (bias column dropped)."""
        X_enc = self.prep.transform(X)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()

        if hasattr(self.clf, "get_booster"):  # XGBoost
            import xgboost as xgb
            contribs = self.clf.get_booster().predict(
                xgb.DMatrix(X_enc, feature_names=None), pred_contribs=True
            )
        elif hasattr(self.clf, "booster_"):  # LightGBM
            contribs = self.clf.predict(X_enc, pred_contrib=True)
        elif hasattr(self.clf, "coef_"):  # linear baseline
            return X_enc * self.clf.coef_[0]
        else:
            raise TypeError(f"Unsupported classifier: {type(self.clf).__name__}")
        return np.asarray(contribs)[:, :-1]

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[Reason]:
        row = X.iloc[0]
        contribs = self._contributions(X.iloc[[0]])[0]

        grouped: dict[str, float] = {}
        for name, value in zip(self.encoded_names, contribs):
            feat = _original_feature(name)
            grouped[feat] = grouped.get(feat, 0.0) + float(value)

        ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        def fmt(v):
            return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)

        return [Reason(feat, fmt(row.get(feat, "")), impact) for feat, impact in ranked]

    # --- public --------------------------------------------------------------
    def predict_frame(self, X: pd.DataFrame, customer_id: str = "adhoc", top_k: int = 5) -> Prediction:
        X = engineer(X) if "num_services" not in X.columns else X
        X = X.drop(columns=[c for c in (config.TARGET, config.ID_COL) if c in X.columns])
        proba = float(self.pipeline.predict_proba(X.iloc[[0]])[:, 1][0])
        return Prediction(
            customer_id=customer_id,
            churn_probability=proba,
            risk_level=self.risk_level(proba),
            threshold=self.threshold,
            top_reasons=self.explain(X, top_k=top_k),
        )


class CustomerStore:
    """In-memory lookup over the Telco dataset (stands in for a CRM/database)."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = (df if df is not None else load_clean()).set_index(config.ID_COL)

    def get(self, customer_id: str) -> pd.DataFrame | None:
        if customer_id not in self.df.index:
            return None
        return self.df.loc[[customer_id]].reset_index()

    def profile(self, customer_id: str) -> dict | None:
        row = self.get(customer_id)
        if row is None:
            return None
        rec = row.iloc[0].to_dict()
        rec["Churn"] = "Yes" if rec.get("Churn") == 1 else "No"  # historical label
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}

    def sample_ids(self, n: int = 10, seed: int = 0) -> list[str]:
        return self.df.sample(n=n, random_state=seed).index.tolist()


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()


@lru_cache(maxsize=1)
def get_store() -> CustomerStore:
    return CustomerStore()
