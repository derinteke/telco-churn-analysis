"""Project-wide configuration: paths, column groups, hyperparameters and LLM/RAG settings."""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
KNOWLEDGE_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "artifacts" / "rag_index"
MLRUNS_DIR = ROOT / "mlruns"

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep Hugging Face downloads inside the project (not the user profile on C:)
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

# --- LLM (Ollama) ------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")  # chosen by eval: same accuracy as 7b, 4x faster on 4 GB VRAM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))  # Ollama defaults to 4096, too small for tool outputs
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "700"))  # hard cap: small models can loop forever
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))
AGENT_MAX_STEPS = 5

# --- RAG ---------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
CHUNK_SIZE = 600      # characters
CHUNK_OVERLAP = 100
RAG_TOP_K = 3  # hit@3 == hit@4 on eval_retrieval; one fewer distractor for the LLM
RAG_RELEVANCE_MARGIN = 0.08  # drop passages this far below the best (cosine); tuned for mpnet, see ITERATIONS.md

# --- Serving -----------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

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
