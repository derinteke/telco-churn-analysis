"""Compare embedding models on the labelled retrieval queries (hybrid search as in production).

Usage:
    python -m eval.compare_embeddings
"""
from __future__ import annotations

import time

import numpy as np

from src import config
from src.rag.chunking import load_documents
from src.rag.vector_store import VectorStore
from eval.eval_retrieval import QUERIES

CANDIDATES = [
    # (model name, query prefix, passage prefix)
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", ""),
]


class PrefixedEmbedder:
    def __init__(self, name: str, q_prefix: str, p_prefix: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self.q_prefix, self.p_prefix = q_prefix, p_prefix
        self.mode = "passage"

    def __call__(self, texts):
        prefix = self.p_prefix if self.mode == "passage" else self.q_prefix
        return self.model.encode([prefix + t for t in texts], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype("float32")


def main():
    chunks = load_documents(config.KNOWLEDGE_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"{'model':62} hit@1  hit@3  MRR    ms/query")
    for name, qp, pp in CANDIDATES:
        emb = PrefixedEmbedder(name, qp, pp)
        store = VectorStore(emb).build(chunks)
        emb.mode = "query"
        hits1 = hits3 = 0
        rr = 0.0
        t0 = time.perf_counter()
        for query, expected in QUERIES:
            sections = [h["section"] for h in store.search(query, k=len(chunks))]
            pos = next((i for i, s in enumerate(sections) if any(e in s for e in expected.split("|"))), None)
            if pos is not None:
                rr += 1 / (pos + 1)
                hits1 += pos < 1
                hits3 += pos < 3
        ms = (time.perf_counter() - t0) * 1000 / len(QUERIES)
        n = len(QUERIES)
        print(f"{name:62} {hits1/n:.2f}   {hits3/n:.2f}   {rr/n:.3f}  {ms:.0f}")


if __name__ == "__main__":
    main()
