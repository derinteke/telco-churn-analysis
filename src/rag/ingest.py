"""Build the FAISS index from the knowledge base.

Usage:
    python -m src.rag.ingest
    python -m src.rag.ingest --query "fiber customer paying by electronic check"
"""
from __future__ import annotations

import argparse
import time

from .. import config
from .chunking import load_documents
from .vector_store import SentenceTransformerEmbedder, VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Run a test query after building the index")
    args = parser.parse_args()

    t0 = time.time()
    chunks = load_documents(config.KNOWLEDGE_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Loaded {len(chunks)} chunks from {config.KNOWLEDGE_DIR}")

    store = VectorStore(SentenceTransformerEmbedder()).build(chunks)
    store.save()
    print(f"Index saved to {config.INDEX_DIR} ({time.time() - t0:.1f}s)")

    if args.query:
        for hit in store.search(args.query):
            print(f"  {hit['score']:.3f}  {hit['source']} | {hit['section']}")


if __name__ == "__main__":
    main()
