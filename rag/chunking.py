"""
rag/chunking.py

Loads PDFs from a directory and splits them into overlapping chunks,
preserving source filename + page number as metadata (needed for citations).
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs(pdf_dir: str) -> List[Document]:
    """Load every PDF in pdf_dir into a list of LangChain Documents (one per page)."""
    documents: List[Document] = []

    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()  # one Document per page, metadata already has 'page' and 'source'

        for page in pages:
            # Normalize metadata so downstream code can rely on these keys
            page.metadata["source"] = filename
            page.metadata["page"] = page.metadata.get("page", 0) + 1  # 1-indexed for humans

        documents.extend(pages)

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Split loaded page-level documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Add a stable chunk index per source file, useful for debugging/citations
    counters = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        chunk.metadata["chunk_id"] = counters[src]

    return chunks


def load_and_chunk(pdf_dir: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[Document]:
    """Convenience wrapper: load all PDFs and chunk them in one call."""
    docs = load_pdfs(pdf_dir)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    # Quick manual smoke test: python -m rag.chunking
    chunks = load_and_chunk("data/pdfs")
    print(f"Loaded {len(chunks)} chunks total.")
    if chunks:
        print("Sample chunk metadata:", chunks[0].metadata)
        print("Sample chunk text:\n", chunks[0].page_content[:300])
