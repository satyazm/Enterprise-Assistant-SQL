"""
app/main.py

Phase 2 FastAPI app: /ask now routes each question to SQL, document
retrieval, or both, then synthesizes a final grounded answer.

Run with:
    uvicorn app.main:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.retriever import retrieve, format_context
from tools.sql_tool import answer_from_sql
from app.router import route_question
from app.llm import call_llm

app = FastAPI(title="Enterprise Agentic RAG - Phase 2")

FINAL_PROMPT_PATH = Path("prompts/rag.txt")


class AskRequest(BaseModel):
    question: str
    k: int = 4


class Source(BaseModel):
    file: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source]
    sql_used: str | None = None


def load_prompt_template() -> str:
    if not FINAL_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {FINAL_PROMPT_PATH}")
    return FINAL_PROMPT_PATH.read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    route = route_question(request.question)

    context_parts = []
    sources = []
    sql_used = None

    if route in ("sql", "both"):
        sql_result = answer_from_sql(request.question)
        sql_used = sql_result["sql"]
        context_parts.append(
            f"[Structured database results]\n"
            f"SQL query used: {sql_result['sql']}\n"
            f"Results:\n{sql_result['result_text']}"
        )

    if route in ("retrieval", "both"):
        chunks = retrieve(request.question, k=request.k)
        if chunks:
            context_parts.append(f"[Document excerpts]\n{format_context(chunks)}")
            sources = [
                Source(file=c.metadata.get("source", "unknown"), page=c.metadata.get("page", "?"))
                for c in chunks
            ]

    if not context_parts:
        return AskResponse(
            answer="I don't have enough information to answer that.",
            route=route,
            sources=[],
            sql_used=sql_used,
        )

    combined_context = "\n\n".join(context_parts)
    template = load_prompt_template()
    prompt = template.format(context=combined_context, question=request.question)

    answer = call_llm(prompt)

    return AskResponse(answer=answer, route=route, sources=sources, sql_used=sql_used)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)