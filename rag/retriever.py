"""
rag/retriever.py

Simple top-k retrieval on top of the FAISS index.
Deliberately minimal for Phase 1 — no reranking or hybrid search yet
(those are Phase 4 "advanced features").
"""

from typing import List
from langchain_core.documents import Document

from rag.vectorstore import load_or_build_index

# Load once at import time so repeated calls don't reload the index from disk.
_index = None


def _get_index():
    global _index
    if _index is None:
        _index = load_or_build_index()
    return _index


def retrieve(query: str, k: int = 4) -> List[Document]:
    """Return the top-k most relevant chunks for a query, with metadata intact."""
    index = _get_index()
    return index.similarity_search(query, k=k)


def format_context(chunks: List[Document]) -> str:
    """
    Turn retrieved chunks into a single context string for the prompt,
    tagging each chunk with an id so the LLM can cite [1], [2], etc.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        lines.append(f"[{i}] (source: {source}, page: {page})\n{chunk.page_content}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    # Quick manual smoke test: python -m rag.retriever
    chunks = retrieve("What is the refund policy?", k=3)
    print(format_context(chunks))
