"""
app/main.py

Phase 3 FastAPI app: /ask now runs through a LangGraph multi-agent
workflow (planner -> sql/retrieval -> reasoning -> report) instead of
manual if/else routing. Same underlying logic as Phase 2, now
orchestrated as an actual graph with a visible execution trace.

Run with:
    uvicorn app.main:app --reload
"""

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.workflow import graph

app = FastAPI(title="Enterprise Agentic RAG - Phase 3")


class AskRequest(BaseModel):
    question: str
    k: int = 4
    # Omit for a one-off, single-turn question. Pass back the session_id
    # from a previous response to continue that conversation — the graph's
    # checkpointer (see graph/workflow.py) uses it to recall prior turns.
    session_id: str | None = None


class Source(BaseModel):
    file: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source]
    sql_used: str | None = None
    execution_path: list[str]
    # Echoed back so the caller can pass it as session_id on the next
    # request to continue this same conversation.
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    result = graph.invoke(
        {
            "question": request.question,
            "k": request.k,
            "sources": [],
            "execution_path": [],
        },
        config={"configurable": {"thread_id": session_id}},
    )

    sources = [Source(**s) for s in result.get("sources", [])]

    return AskResponse(
        answer=result["final_answer"],
        route=result["route"],
        sources=sources,
        sql_used=result.get("sql_query"),
        execution_path=result["execution_path"],
        session_id=session_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
