"""
app/main.py

Phase 3 FastAPI app: /ask now runs through a LangGraph multi-agent
workflow (planner -> sql/retrieval -> reasoning -> report) instead of
manual if/else routing. Same underlying logic as Phase 2, now
orchestrated as an actual graph with a visible execution trace.

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.workflow import graph

app = FastAPI(title="Enterprise Agentic RAG - Phase 3")


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
    execution_path: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = graph.invoke({
        "question": request.question,
        "k": request.k,
        "sources": [],
        "execution_path": [],
    })

    sources = [Source(**s) for s in result.get("sources", [])]

    return AskResponse(
        answer=result["final_answer"],
        route=result["route"],
        sources=sources,
        sql_used=result.get("sql_query"),
        execution_path=result["execution_path"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
