"""Load and clean the Telco Customer Churn dataset."""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Read the raw CSV as-is."""
    return pd.read_csv(config.DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    Fixes known issues in the IBM Telco dataset:
    - ``TotalCharges`` is stored as text and contains blank strings for
      customers with ``tenure == 0`` (they have not been billed yet).
    - ``SeniorCitizen`` is 0/1 but is semantically categorical.
    - The binary target is mapped to 0/1.
    """
    df = df.copy()

    # TotalCharges: coerce to numeric; blanks -> NaN -> 0 (tenure 0 => not billed)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # SeniorCitizen: 0/1 -> readable category (kept as string for one-hot)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Target -> 0/1
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)

    return df


def load_clean() -> pd.DataFrame:
    """Convenience: load and clean in one call."""
    return clean(load_raw())


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d.dtypes)
    print(f"Churn rate: {d[config.TARGET].mean():.1%}")
