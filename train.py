"""End-to-end training script: clean -> engineer -> train -> evaluate -> save.

Usage:
    python train.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_clean
from src.evaluate import print_report, save_confusion_matrix, save_roc_curves
from src.features import engineer, split_X_y
from src.model import build_models


def best_f1_threshold(y_true, y_proba) -> float:
    """Return the probability threshold that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def main() -> None:
    # 1. Load, clean, engineer
    df = engineer(load_clean())
    X, y = split_X_y(df)
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features | churn rate {y.mean():.1%}")

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE,
    )

    # 3. Train & evaluate every model
    models = build_models()
    results, fitted = {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = print_report(name, y_test, y_pred, y_proba)
        save_confusion_matrix(y_test, y_pred, name)
        fitted[name] = pipe

    # 4. ROC comparison plot
    save_roc_curves(fitted, X_test, y_test)

    # 5. Pick best by ROC-AUC, report a tuned threshold too
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = fitted[best_name]
    proba = best.predict_proba(X_test)[:, 1]
    thr = best_f1_threshold(y_test, proba)
    print(f"\nBest model: {best_name} (ROC-AUC {results[best_name]['roc_auc']:.3f})")
    print(f"Best-F1 threshold: {thr:.2f}")

    # 6. Persist
    joblib.dump(best, config.MODELS_DIR / "best_model.joblib")
    results["_best_model"] = best_name
    results["_best_f1_threshold"] = round(thr, 3)
    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {config.MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
