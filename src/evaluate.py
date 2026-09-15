"""Evaluation metrics and plots for churn classifiers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the headline metrics as a dict."""
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def print_report(name: str, y_true, y_pred, y_proba) -> dict:
    m = compute_metrics(y_true, y_pred, y_proba)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC : {m['roc_auc']:.3f}")
    print(f"F1      : {m['f1']:.3f}")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall  : {m['recall']:.3f}")
    print(classification_report(y_true, y_pred, digits=3))
    return m


def save_confusion_matrix(y_true, y_pred, name: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", colorbar=False,
        display_labels=["Stay", "Churn"],
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_curves(models: dict, X_test, y_test):
    """models: {name: fitted_pipeline}. Plots all ROC curves together."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_title("ROC Curves")
    fig.tight_layout()
    path = config.FIGURES_DIR / "roc_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
