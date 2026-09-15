"""Generate exploratory data analysis figures saved to reports/figures/."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .data import load_clean

sns.set_theme(style="whitegrid", palette="deep")
PRIMARY = "#2563eb"
CHURN_PALETTE = {0: "#94a3b8", 1: "#ef4444"}


def _save(fig, name: str):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def churn_balance(df):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[config.TARGET].map({0: "Stay", 1: "Churn"}).value_counts()
    ax.bar(counts.index, counts.values, color=[PRIMARY, "#ef4444"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=10)
    ax.set_title("Target Balance: Churn vs Stay")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    return _save(fig, "eda_churn_balance.png")


def churn_by_category(df, col: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby(col)[config.TARGET].mean().sort_values()
    ax.barh(rate.index.astype(str), rate.values, color=PRIMARY)
    overall = df[config.TARGET].mean()
    ax.axvline(overall, color="#ef4444", ls="--", label=f"Overall {overall:.0%}")
    for i, v in enumerate(rate.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_title(f"Churn Rate by {col}")
    ax.set_xlabel("Churn rate")
    ax.legend()
    fig.tight_layout()
    return _save(fig, fname)


def tenure_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for churn_val, label in [(0, "Stay"), (1, "Churn")]:
        sns.kdeplot(
            df[df[config.TARGET] == churn_val]["tenure"],
            ax=ax, fill=True, alpha=0.4, label=label,
            color=CHURN_PALETTE[churn_val],
        )
    ax.set_title("Tenure Distribution by Churn")
    ax.set_xlabel("Tenure (months)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "eda_tenure_distribution.png")


def numeric_correlation(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    cols = config.NUMERIC_FEATURES + [config.TARGET]
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    return _save(fig, "eda_correlation.png")


def main():
    df = load_clean()
    churn_balance(df)
    churn_by_category(df, "Contract", "eda_churn_by_contract.png")
    churn_by_category(df, "PaymentMethod", "eda_churn_by_payment.png")
    tenure_distribution(df)
    numeric_correlation(df)
    print("EDA figures saved to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
