"""End-to-end training script: clean -> engineer -> train -> evaluate -> save.

Every run is tracked in MLflow (params, metrics, plots, model). The best
model is registered as ``telco-churn`` in the MLflow model registry.

Usage:
    python train.py                # train + track in MLflow
    python train.py --no-mlflow    # train only
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_clean
from src.evaluate import print_report, save_confusion_matrix, save_roc_curves
from src.features import engineer, split_X_y
from src.model import build_models

EXPERIMENT = "telco-churn"
REGISTERED_MODEL = "telco-churn"


def best_f1_threshold(y_true, y_proba) -> float:
    """Return the probability threshold that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _params_of(pipe) -> dict:
    clf = pipe.named_steps["clf"]
    return {f"clf.{k}": v for k, v in clf.get_params().items()
            if isinstance(v, (int, float, str, bool)) or v is None}


def main(track: bool = True) -> None:
    mlflow = _mlflow() if track else None

    # 1. Load, clean, engineer
    df = engineer(load_clean())
    X, y = split_X_y(df)
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features | churn rate {y.mean():.1%}")

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE,
    )

    # 3. Train & evaluate every model (one nested MLflow run per model)
    models = build_models()
    results, fitted, run_ids = {}, {}, {}
    parent = mlflow.start_run(run_name="train") if mlflow else nullcontext()
    with parent:
        if mlflow:
            mlflow.log_params({
                "n_rows": len(X), "n_features": X.shape[1], "test_size": config.TEST_SIZE,
                "random_state": config.RANDOM_STATE, "churn_rate": round(float(y.mean()), 4),
            })

        for name, pipe in models.items():
            child = mlflow.start_run(run_name=name, nested=True) if mlflow else nullcontext()
            with child as run:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_proba = pipe.predict_proba(X_test)[:, 1]
                results[name] = print_report(name, y_test, y_pred, y_proba)
                results[name]["pr_auc"] = average_precision_score(y_test, y_proba)
                save_confusion_matrix(y_test, y_pred, name)
                fitted[name] = pipe

                if mlflow:
                    run_ids[name] = run.info.run_id
                    mlflow.log_params(_params_of(pipe))
                    mlflow.log_metrics(results[name])
                    cm = config.FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
                    if cm.exists():
                        mlflow.log_artifact(str(cm), "figures")

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

        # 7. Log the winner and register it
        if mlflow:
            mlflow.log_param("best_model", best_name)
            mlflow.log_metrics({"best_roc_auc": results[best_name]["roc_auc"], "best_f1_threshold": thr})
            mlflow.log_artifact(str(config.FIGURES_DIR / "roc_curves.png"), "figures")
            mlflow.log_artifact(str(config.MODELS_DIR / "metrics.json"))
            with mlflow.start_run(run_id=run_ids[best_name], nested=True):
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    best, name="model", input_example=X_test.head(3),
                    # skops (the default) rejects XGBoost/LightGBM types; the model is our own
                    serialization_format="cloudpickle",
                    registered_model_name=REGISTERED_MODEL,
                )
            print(f"MLflow: registered '{REGISTERED_MODEL}' from run {run_ids[best_name]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mlflow", action="store_true", help="Skip MLflow tracking")
    main(track=not ap.parse_args().no_mlflow)
