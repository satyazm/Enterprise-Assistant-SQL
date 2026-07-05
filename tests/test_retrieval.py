"""
tests/test_retrieval.py

Basic sanity tests for the retrieval pipeline.
These don't prove correctness of answers — just that the pipeline
doesn't silently break as the codebase grows.

Run with: pytest tests/
NOTE: requires data/pdfs/ to be populated and a valid API key in .env,
since embeddings are computed against a real provider.
"""

from rag.retriever import retrieve, format_context


def test_retrieve_returns_chunks():
    chunks = retrieve("refund policy", k=3)
    assert len(chunks) > 0, "Expected at least one chunk to be retrieved."


def test_chunks_have_required_metadata():
    chunks = retrieve("refund policy", k=3)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata


def test_format_context_includes_citations():
    chunks = retrieve("refund policy", k=2)
    context = format_context(chunks)
    assert "[1]" in context
    assert "source:" in context
