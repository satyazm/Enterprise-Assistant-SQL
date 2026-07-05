"""
rag/vectorstore.py

Builds a FAISS index from chunked documents, or loads a previously
persisted index from disk so we don't re-embed on every restart.
"""

import os

from langchain_community.vectorstores import FAISS

from rag.chunking import load_and_chunk
from rag.embeddings import get_embedding_model

INDEX_DIR = "data/faiss_index"


def build_index(pdf_dir: str = "data/pdfs", index_dir: str = INDEX_DIR) -> FAISS:
    """Build a fresh FAISS index from all PDFs in pdf_dir and persist it."""
    chunks = load_and_chunk(pdf_dir)
    if not chunks:
        raise ValueError(f"No chunks produced from {pdf_dir} — check your PDFs.")

    embedding_model = get_embedding_model()
    index = FAISS.from_documents(chunks, embedding_model)

    os.makedirs(index_dir, exist_ok=True)
    index.save_local(index_dir)
    print(f"Built and saved FAISS index with {len(chunks)} chunks to '{index_dir}'.")

    return index


def load_index(index_dir: str = INDEX_DIR) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embedding_model = get_embedding_model()
    return FAISS.load_local(
        index_dir,
        embedding_model,
        allow_dangerous_deserialization=True,  # safe here: it's our own local index
    )


def load_or_build_index(pdf_dir: str = "data/pdfs", index_dir: str = INDEX_DIR) -> FAISS:
    """Load the index if it exists on disk, otherwise build it from scratch."""
    index_file = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_file):
        print(f"Loading existing FAISS index from '{index_dir}'.")
        return load_index(index_dir)

    print("No existing index found — building a new one.")
    return build_index(pdf_dir, index_dir)


if __name__ == "__main__":
    # Quick manual smoke test: python -m rag.vectorstore
    idx = load_or_build_index()
    results = idx.similarity_search("What is the refund policy?", k=2)
    for r in results:
        print(r.metadata, "->", r.page_content[:150])
