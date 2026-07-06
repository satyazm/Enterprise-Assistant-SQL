"""
app/llm.py

Shared LLM dispatcher, used by both the RAG pipeline and the SQL agent
(and, from Phase 3 onward, by every LangGraph node). Kept in one place
so provider swaps only happen here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


def call_llm(prompt: str) -> str:
    """Uses LangChain chat model wrappers so this plugs directly into
    LangGraph nodes later without a rewrite."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env")

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
        )
        response = model.invoke(prompt)
        return response.content

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")

        model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
        )
        response = model.invoke(prompt)
        return response.content

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
