import hashlib
import re

import numpy as np
import pytest

from src import config
from src.rag.chunking import load_documents
from src.rag.vector_store import VectorStore


def hashing_embedder(texts, dim: int = 256):
    """Deterministic bag-of-words embedder, so tests need no model download."""
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9\-]+", t.lower()):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    return out


@pytest.fixture(scope="session")
def kb_store():
    chunks = load_documents(config.KNOWLEDGE_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return VectorStore(hashing_embedder).build(chunks)


MODEL_PATH = config.MODELS_DIR / "best_model.joblib"
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model missing: run `python train.py --no-mlflow`"
)
