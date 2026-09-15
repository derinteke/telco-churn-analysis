"""Feature engineering and the preprocessing pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of the cleaned dataframe.

    New columns
    -----------
    num_services : int
        How many add-on/telco services the customer subscribes to. Fewer
        services tends to mean a looser relationship with the provider.
    tenure_group : str
        Tenure bucketed into readable bands, which lets tree models split on
        "lifecycle stage" directly.
    charge_per_tenure : float
        Total spend normalised by lifetime — a proxy for how heavy a customer is.
    has_internet : int
        Whether the customer has any internet service at all.
    monthly_to_total_ratio : float
        High for new customers (few bills so far), low for long-tenured ones.
    """
    df = df.copy()
    services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PhoneService", "MultipleLines",
    ]
    df["num_services"] = (df[services] == "Yes").sum(axis=1)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 1000],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"],
    ).astype(str)
    df["charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    return df


def split_X_y(df: pd.DataFrame):
    """Return (X, y), dropping the id column."""
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET, config.ID_COL])
    return X, y


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer over the full (base + engineered) feature set.

    Parameters
    ----------
    scale : bool
        Scale numeric features. Useful for linear models; tree models
        (LightGBM/XGBoost) do not need it, so it can be turned off.
    """
    numeric_transformer = (
        Pipeline(steps=[("scaler", StandardScaler())]) if scale else "passthrough"
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="if_binary")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.ALL_NUMERIC),
            ("cat", categorical_transformer, config.ALL_CATEGORICAL),
        ],
        remainder="drop",
    )
