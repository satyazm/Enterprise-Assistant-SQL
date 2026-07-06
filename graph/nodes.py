"""
graph/nodes.py

Each node is a thin wrapper around logic that already exists and already
works (from Phases 1-2). Nothing about the SQL generation, retrieval, or
LLM calling changes here — only how they're orchestrated.
"""

from pathlib import Path

from app.router import route_question
from app.llm import call_llm
from tools.sql_tool import answer_from_sql
from rag.retriever import retrieve, format_context
from graph.state import GraphState

FINAL_PROMPT_PATH = Path("prompts/rag.txt")


def _load_prompt_template() -> str:
    if not FINAL_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {FINAL_PROMPT_PATH}")
    return FINAL_PROMPT_PATH.read_text()


def planner_node(state: GraphState) -> dict:
    """Classifies the question as sql / retrieval / both."""
    route = route_question(state["question"])
    return {"route": route, "execution_path": ["planner"]}


def sql_node(state: GraphState) -> dict:
    """Generates and safely executes SQL against the structured database."""
    result = answer_from_sql(state["question"])
    return {
        "sql_query": result["sql"],
        "sql_result_text": result["result_text"],
        "execution_path": ["sql"],
    }


def retrieval_node(state: GraphState) -> dict:
    """Retrieves relevant document chunks from the FAISS index."""
    chunks = retrieve(state["question"], k=state.get("k", 4))

    if not chunks:
        return {"retrieved_context": None, "sources": [], "execution_path": ["retrieval"]}

    context = format_context(chunks)
    sources = [
        {"file": c.metadata.get("source", "unknown"), "page": c.metadata.get("page", "?")}
        for c in chunks
    ]
    return {"retrieved_context": context, "sources": sources, "execution_path": ["retrieval"]}


def reasoning_node(state: GraphState) -> dict:
    """Merges whatever ran (SQL, retrieval, or both) into one context block."""
    parts = []

    if state.get("sql_result_text"):
        parts.append(
            f"[Structured database results]\n"
            f"SQL query used: {state.get('sql_query')}\n"
            f"Results:\n{state['sql_result_text']}"
        )

    if state.get("retrieved_context"):
        parts.append(f"[Document excerpts]\n{state['retrieved_context']}")

    combined = "\n\n".join(parts) if parts else None
    return {"combined_context": combined, "execution_path": ["reasoning"]}


def report_node(state: GraphState) -> dict:
    """Generates the final grounded answer from the merged context."""
    combined = state.get("combined_context")

    if not combined:
        return {
            "final_answer": "I don't have enough information to answer that.",
            "execution_path": ["report"],
        }

    template = _load_prompt_template()
    prompt = template.format(context=combined, question=state["question"])
    answer = call_llm(prompt)

    return {"final_answer": answer, "execution_path": ["report"]}
