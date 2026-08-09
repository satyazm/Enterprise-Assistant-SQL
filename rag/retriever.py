"""
rag/retriever.py

Hybrid retrieval: FAISS vector search + BM25 keyword search, fused with
reciprocal rank fusion (LangChain's EnsembleRetriever), then narrowed down
to the final top-k with a cross-encoder reranker.

Why hybrid: pure vector search misses exact-term matches — a product code,
an exact policy phrase, a number — that embeddings can blur together in
semantic space; BM25 catches those but misses paraphrases/synonyms that
vector search is good at. Fusing both casts a wider net than either alone.

Why rerank on top of that: fusion ranks by *retrieval* signal (embedding
distance / term overlap), not by how well a chunk actually answers the
query. A cross-encoder scores each (query, chunk) pair jointly instead of
comparing independently-computed vectors, which is a substantially
stronger relevance signal — it's what turns "a wider net of maybe-relevant
chunks" back into a precise top-k. This is the standard retrieve-many,
rerank-to-few pattern used in production RAG systems; naive top-k vector
search alone is a common gap in demo-grade implementations.
"""

from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.chunking import load_and_chunk
from rag.vectorstore import load_or_build_index

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates each retriever contributes before fusion + rerank —
# wider than the final k so the reranker has real signal to sort through,
# rather than just re-ordering an already-narrow list.
CANDIDATE_POOL_SIZE = 15

# Equal weighting: no a priori reason to trust keyword over semantic match
# more, or vice versa, for this document set. Worth tuning against a real
# eval set (see eval/) rather than guessing further.
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5

# Loaded/built once per process, reused across requests — rebuilding the
# BM25 index or reloading the cross-encoder per query would add latency
# for no benefit, since none of these depend on the query itself.
_ensemble_retriever = None
_reranker = None


def _build_bm25_retriever(pdf_dir: str = "data/pdfs") -> BM25Retriever:
    chunks = load_and_chunk(pdf_dir)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = CANDIDATE_POOL_SIZE
    return retriever


def _get_ensemble_retriever():
    global _ensemble_retriever
    if _ensemble_retriever is None:
        vector_index = load_or_build_index()
        vector_retriever = vector_index.as_retriever(search_kwargs={"k": CANDIDATE_POOL_SIZE})
        bm25_retriever = _build_bm25_retriever()
        _ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[VECTOR_WEIGHT, BM25_WEIGHT],
        )
    return _ensemble_retriever


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _rerank(query: str, candidates: List[Document], top_n: int) -> List[Document]:
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in ranked[:top_n]]


def retrieve(query: str, k: int = 4) -> List[Document]:
    """Return the top-k most relevant chunks for a query: FAISS + BM25
    candidates fused via reciprocal rank fusion, then narrowed to k by
    cross-encoder reranking."""
    ensemble = _get_ensemble_retriever()
    candidates = ensemble.invoke(query)
    return _rerank(query, candidates, top_n=k)


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
