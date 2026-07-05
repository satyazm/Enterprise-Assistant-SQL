"""
rag/embeddings.py

Thin wrapper around an embedding provider so the rest of the codebase
(vectorstore, retriever) never has to know which provider is in use.
Swap the implementation here if you change providers later.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Choose ONE provider. Default: Google Gemini embeddings.
# Set EMBEDDING_PROVIDER=openai in .env to switch to OpenAI instead.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")


def get_embedding_model():
    """
    Returns a LangChain-compatible embedding object exposing
    .embed_documents(list[str]) and .embed_query(str).
    """
    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env")

        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
        )


    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
