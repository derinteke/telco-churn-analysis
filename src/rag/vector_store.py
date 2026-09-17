"""FAISS-backed vector store with a pluggable embedding function."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol

import faiss
import numpy as np

from .. import config
from .chunking import Chunk


class Embedder(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Multilingual sentence embeddings, so Turkish questions match English docs."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")


RRF_K = 60  # standard Reciprocal Rank Fusion constant
# Chosen with eval/eval_retrieval.py (15 labelled queries): hit@4 0.93 / MRR 0.832,
# vs dense-only 0.87 / 0.787 and equal weights 0.93 / 0.793.
BM25_WEIGHT = 0.5
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


class BM25:
    """Okapi BM25 over a small in-memory corpus (no extra dependency)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_tokenize(d)) for d in docs]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def coverage(self, query: str) -> float:
        """Share of query tokens that occur in the corpus vocabulary (0..1)."""
        tokens = _tokenize(query)
        return sum(t in self.idf for t in tokens) / len(tokens) if tokens else 0.0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in set(_tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                    out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


class VectorStore:
    """Inner-product index over L2-normalised vectors (= cosine similarity)."""

    INDEX_FILE = "index.faiss"
    META_FILE = "chunks.json"

    def __init__(self, embedder: Embedder | Callable[[list[str]], np.ndarray]):
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> "VectorStore":
        if not chunks:
            raise ValueError("No chunks to index.")
        vectors = _normalize(self.embedder([c.embedding_text for c in chunks]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = list(chunks)
        return self

    def search(self, query: str, k: int = config.RAG_TOP_K, mode: str = "hybrid") -> list[dict]:
        """Dense (cosine), sparse (BM25) or hybrid search.

        Hybrid fuses both rankings with Reciprocal Rank Fusion: dense handles
        paraphrases and Turkish-to-English matching, while BM25 catches exact terms
        ("discount", "approval") that dense ranking can bury.
        """
        if self.index is None:
            raise RuntimeError("Index is empty. Build or load it first.")
        n = len(self.chunks)
        q = _normalize(self.embedder([query]))
        dense_scores, dense_ids = self.index.search(q, n)
        dense_rank = [int(i) for i in dense_ids[0] if i != -1]
        dense_score = {int(i): float(s) for s, i in zip(dense_scores[0], dense_ids[0]) if i != -1}

        if mode == "dense":
            ranked = dense_rank
        else:
            bm25 = self._bm25().scores(query)
            sparse_rank = [i for i in np.argsort(-bm25) if bm25[i] > 0]
            if mode == "bm25":
                ranked = [int(i) for i in sparse_rank]
            else:
                fused: dict[int, float] = {}
                # Scale BM25 by how much of the query exists in the (English) corpus: a Turkish
                # query matching only "fiber" should not let that single word reorder results.
                bm25_weight = BM25_WEIGHT * self._bm25().coverage(query)
                for weight, ranking in ((1.0, dense_rank), (bm25_weight, sparse_rank)):
                    for r, i in enumerate(ranking):
                        fused[int(i)] = fused.get(int(i), 0.0) + weight / (RRF_K + r + 1)
                ranked = sorted(fused, key=fused.get, reverse=True)

        return [
            {**self.chunks[i].to_dict(), "score": round(dense_score.get(i, 0.0), 4)}
            for i in ranked[:k]
        ]

    def _bm25(self) -> "BM25":
        if getattr(self, "_bm25_index", None) is None:
            self._bm25_index = BM25([c.embedding_text for c in self.chunks])
        return self._bm25_index

    def save(self, directory: Path = config.INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / self.INDEX_FILE))
        with open(directory / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, embedder, directory: Path = config.INDEX_DIR) -> "VectorStore":
        index_path = directory / cls.INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found. Run `python -m src.rag.ingest` first.")
        store = cls(embedder)
        store.index = faiss.read_index(str(index_path))
        with open(directory / cls.META_FILE, encoding="utf-8") as f:
            store.chunks = [Chunk(**d) for d in json.load(f)]
        return store
