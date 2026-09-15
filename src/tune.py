"""Hyperparameter tuning with Optuna (documents how config params were found).

Run:  python -m src.tune --model xgboost --trials 50
This reproduces the search; the winning params are pasted into config.py.
"""
from __future__ import annotations

import argparse

import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from . import config
from .data import load_clean
from .features import build_preprocessor, engineer, split_X_y
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_auc(clf) -> float:
    pipe = Pipeline([("prep", build_preprocessor(scale=False)), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()


def lgbm_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 900),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 90),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )
    return _cv_auc(LGBMClassifier(scale_pos_weight=config.SCALE_POS_WEIGHT,
                                  random_state=config.RANDOM_STATE, verbose=-1, **params))


def xgb_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 900),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
    )
    return _cv_auc(XGBClassifier(scale_pos_weight=config.SCALE_POS_WEIGHT,
                                 random_state=config.RANDOM_STATE, eval_metric="logloss",
                                 tree_method="hist", **params))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lightgbm", "xgboost"], default="xgboost")
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    df = engineer(load_clean())
    X, y = split_X_y(df)

    objective = lgbm_objective if args.model == "lightgbm" else xgb_objective
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBest CV ROC-AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)
