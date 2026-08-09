"""
app/main.py

Phase 1 FastAPI app: a single /ask endpoint that retrieves relevant
chunks from the FAISS index and generates a grounded, cited answer.

Run with:
    uvicorn app.main:app --reload
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.retriever import retrieve, format_context

load_dotenv()

app = FastAPI(title="Enterprise Agentic RAG - Phase 1")

PROMPT_TEMPLATE_PATH = Path("prompts/rag.txt")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


class AskRequest(BaseModel):
    question: str
    k: int = 4


class Source(BaseModel):
    file: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def load_prompt_template() -> str:
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
    return PROMPT_TEMPLATE_PATH.read_text()


def call_llm(prompt: str) -> str:
    """Thin dispatcher so swapping LLM providers only touches this function.

    Uses LangChain chat model wrappers (not raw provider SDKs) deliberately:
    Phase 3 wraps this logic into LangGraph nodes, which expect the standard
    LangChain .invoke() interface, tool-binding, and structured output support.
    Building on that interface now avoids a rewrite later.
    """
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
        return response.text


    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(request.question, k=request.k)
    if not chunks:
        return AskResponse(
            answer="I don't have enough information in the provided documents to answer that.",
            sources=[],
        )

    context = format_context(chunks)
    template = load_prompt_template()
    # rag.txt gained a {history} placeholder for Phase 3's multi-turn memory;
    # this legacy phase-1 app is single-turn, so it always fills in "none".
    prompt = template.format(
        context=context,
        question=request.question,
        history="(no earlier turns in this conversation)",
    )

    answer = call_llm(prompt)

    sources = [
        Source(file=chunk.metadata.get("source", "unknown"), page=chunk.metadata.get("page", "?"))
        for chunk in chunks
    ]

    return AskResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)