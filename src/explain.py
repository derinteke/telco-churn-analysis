"""Generate a SHAP feature-importance summary for the LightGBM model."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from . import config
from .data import load_clean
from .features import engineer, split_X_y
from .model import build_models


def main():
    df = engineer(load_clean())
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )

    pipe = build_models()["LightGBM"]  # tree model for SHAP TreeExplainer
    pipe.fit(X_train, y_train)

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feature_names = prep.get_feature_names_out()

    X_test_enc = prep.transform(X_test)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_enc)

    # LightGBM binary -> may return list; take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(
        shap_values, X_test_enc, feature_names=feature_names,
        plot_type="bar", max_display=12, show=False,
    )
    plt.title("SHAP Feature Importance (LightGBM)")
    plt.tight_layout()
    path = config.FIGURES_DIR / "shap_importance.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print("SHAP figure saved to", path)


if __name__ == "__main__":
    main()
