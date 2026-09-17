from src.rag.chunking import chunk_markdown
from src.rag.vector_store import VectorStore
from tests.conftest import hashing_embedder

DOC = """# Campaigns

## RET-A
Offer A for month-to-month customers.

## RET-B
Offer B for fiber customers.
"""


def test_chunks_follow_headings():
    chunks = chunk_markdown(DOC, "doc.md")
    assert [c.section for c in chunks] == ["Campaigns > RET-A", "Campaigns > RET-B"]
    assert all(c.source == "doc.md" for c in chunks)


def test_long_section_is_split_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "x" * 150 for i in range(10))
    chunks = chunk_markdown(f"# Big\n{body}", "big.md", size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert chunks[1].section.endswith("(part 2)")


def test_vector_store_roundtrip(tmp_path):
    store = VectorStore(hashing_embedder).build(chunk_markdown(DOC, "doc.md"))
    store.save(tmp_path)
    loaded = VectorStore.load(hashing_embedder, tmp_path)
    hit = loaded.search("fiber customers offer", k=1)[0]
    assert "RET-B" in hit["section"]


def test_knowledge_base_retrieval(kb_store):
    hits = kb_store.search("electronic check payment incentive RET-AUTOPAY", k=3)
    assert any(h["source"] == "retention_campaigns.md" for h in hits)
