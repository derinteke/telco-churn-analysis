"""Model definitions: an interpretable baseline plus two tuned gradient boosters."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .features import build_preprocessor


def build_models() -> dict[str, Pipeline]:
    """Return {name: sklearn Pipeline}.

    - Logistic Regression: interpretable baseline, needs scaling.
    - LightGBM / XGBoost: gradient boosters with Optuna-tuned hyperparameters.
      Trees do not need scaling. Class imbalance (~27% churn) is handled with
      ``class_weight`` / ``scale_pos_weight``.
    """
    logreg = Pipeline(steps=[
        ("prep", build_preprocessor(scale=True)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )),
    ])

    lgbm = Pipeline(steps=[
        ("prep", build_preprocessor(scale=False)),
        ("clf", LGBMClassifier(
            scale_pos_weight=config.SCALE_POS_WEIGHT,
            random_state=config.RANDOM_STATE,
            verbose=-1,
            **config.LGBM_PARAMS,
        )),
    ])

    xgb = Pipeline(steps=[
        ("prep", build_preprocessor(scale=False)),
        ("clf", XGBClassifier(
            scale_pos_weight=config.SCALE_POS_WEIGHT,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            **config.XGB_PARAMS,
        )),
    ])

    return {"Logistic Regression": logreg, "LightGBM": lgbm, "XGBoost": xgb}
