"""Heading-aware Markdown chunking.

Documents are first split on Markdown headings so a chunk never mixes two
unrelated sections (e.g. two different campaigns). Sections longer than
``chunk_size`` are split further on paragraph boundaries with overlap. Each
chunk keeps its heading path ("Active Retention Campaigns > RET-SECURE") and
that path is prepended to the embedded text, which noticeably helps retrieval
for short sections.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    text: str

    @property
    def embedding_text(self) -> str:
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body)] for each heading-delimited section."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            pieces.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n\n{para}" if current else para
        # A single paragraph larger than the window: hard-split it
        while len(current) > size:
            pieces.append(current[:size].strip())
            current = current[size - overlap:]
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_markdown(markdown: str, source: str, size: int = 600, overlap: int = 100) -> list[Chunk]:
    chunks = []
    for section, body in _split_sections(markdown):
        for i, piece in enumerate(_split_long(body, size, overlap)):
            chunks.append(Chunk(
                id=f"{source}::{len(chunks)}", source=source,
                section=section + (f" (part {i + 1})" if i else ""), text=piece,
            ))
    return chunks


def load_documents(directory: Path, size: int = 600, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name, size, overlap))
    return chunks
