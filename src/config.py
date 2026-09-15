"""Project-wide configuration: paths, column groups and tuned hyperparameters."""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Columns -----------------------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"

# Original numeric predictors
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Original categorical predictors
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Engineered features (created in features.engineer)
ENGINEERED_NUMERIC = [
    "num_services", "charge_per_tenure", "has_internet", "monthly_to_total_ratio",
]
ENGINEERED_CATEGORICAL = ["tenure_group"]

# Full lists used by the model pipelines
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

# Class imbalance: churn rate ~27% -> positive class weight = 73/27 = 2.77
SCALE_POS_WEIGHT = 2.77

# --- Tuned hyperparameters (found with Optuna, 5-fold CV on ROC-AUC) ----------
LGBM_PARAMS = {
    "n_estimators": 705, "learning_rate": 0.00769, "num_leaves": 37,
    "max_depth": 3, "min_child_samples": 16, "subsample": 0.6821,
    "colsample_bytree": 0.7151, "reg_alpha": 4.8793, "reg_lambda": 0.9707,
}
XGB_PARAMS = {
    "n_estimators": 368, "learning_rate": 0.02025, "max_depth": 3,
    "min_child_weight": 9, "subsample": 0.6425, "colsample_bytree": 0.7705,
    "reg_alpha": 4.3913, "reg_lambda": 0.6014, "gamma": 0.3891,
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
