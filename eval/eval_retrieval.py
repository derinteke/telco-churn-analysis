"""Retrieval eval: hit@k and MRR for dense, BM25 and hybrid (weighted RRF) search.

Each query has the section that contains the answer. Queries mix English and
Turkish, and exact-term and paraphrased questions.

Usage:
    python -m eval.eval_retrieval
"""
from __future__ import annotations

import numpy as np

from src.rag.vector_store import RRF_K, SentenceTransformerEmbedder, VectorStore

QUERIES = [
    ("What is the maximum total discount allowed without manager approval?", "Stacking rules|4. Escalation"),
    ("Fatura geç ödenirse ne olur?", "payment is late"),
    ("Müşterinin fiber interneti sürekli kopuyor, ne yapmalıyım?", "unstable Fiber"),
    ("electronic check payment incentive", "RET-AUTOPAY"),
    ("RET-LOCK24 ve RET-LOCK12 aynı müşteriye birlikte verilebilir mi?", "Stacking rules"),
    ("Can a customer get money back if they cancel?", "refund when cancelling"),
    ("İlk faturam neden bu kadar yüksek?", "first bill"),
    ("How much does a Fiber 1 Gbps plan cost?", "Internet plans"),
    ("Yeni müşterilere hangi kampanya uygun?", "RET-NEWBIE"),
    ("When should I escalate to the compliance team?", "4. Escalation"),
    ("DSL hızı düşük, ne önermeliyim?", "DSL speed"),
    ("How do we measure whether retention campaigns work?", "Measuring success"),
    ("Erken fesih bedeli nasıl hesaplanır?", "One-year contract|Two-year contract"),
    ("What does the Tech Support add-on include?", "Add-on services|Service level"),
    ("Rakip teklif alan fiber müşterisine indirim", "RET-FIBERVALUE"),
    # Added in iteration 17 after a test-set answer used the wrong chunk for a price question
    ("Fiber 1000 paketinin aylık ücreti ne kadar?", "Internet plans"),
    ("How much does the basic DSL plan cost per month?", "Internet plans"),
    ("Kağıt fatura için ücret alınıyor mu?", "paperless billing"),
    ("Birden fazla telefon hattının ücreti nedir?", "Phone plans"),
    ("Is there a discount for signing a two-year contract?", "Two-year contract"),
]


def evaluate_ranking(store: VectorStore, rank_fn, k: int = 4) -> tuple[float, float]:
    hits, rr = 0, 0.0
    for query, expected in QUERIES:
        sections = [store.chunks[i].section for i in rank_fn(query)]
        pos = next((r for r, s in enumerate(sections) if any(e in s for e in expected.split("|"))), None)
        if pos is not None:
            rr += 1 / (pos + 1)
            hits += pos < k
    return hits / len(QUERIES), rr / len(QUERIES)


def main():
    store = VectorStore.load(SentenceTransformerEmbedder())
    n = len(store.chunks)
    bm25 = store._bm25()

    def dense(q):
        _, ids = store.index.search(np.asarray(store.embedder([q]), dtype="float32"), n)
        return [int(i) for i in ids[0]]

    def sparse(q):
        s = bm25.scores(q)
        return [int(i) for i in np.argsort(-s)]

    def hybrid(w_sparse):
        def rank(q):
            fused = {}
            for w, ranking in ((1.0, dense(q)), (w_sparse, [i for i in sparse(q) if bm25.scores(q)[i] > 0])):
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + w / (RRF_K + r + 1)
            return sorted(fused, key=fused.get, reverse=True)
        return rank

    def production(q):
        return [store.chunks.index(next(c for c in store.chunks if c.id == h["id"])) for h in store.search(q, k=n)]

    rows = [("dense", dense), ("bm25", sparse)] + [(f"hybrid w_bm25={w}", hybrid(w)) for w in (0.25, 0.5, 0.75, 1.0)]
    rows.append(("production (coverage-scaled)", production))
    print(f"{'method':30} hit@4   MRR")
    for name, fn in rows:
        hit, mrr = evaluate_ranking(store, fn)
        print(f"{name:30} {hit:.2f}   {mrr:.3f}")


if __name__ == "__main__":
    main()
