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

# How many recent turns get spelled out in full inside prompts. Separate
# from MAX_HISTORY_TURNS in state.py (which bounds what's *stored*) — this
# bounds what's *shown to the LLM on every call*, since token cost scales
# with every node that reads history (planner, sql, report all read it).
MAX_HISTORY_TURNS_IN_PROMPT = 3


def _load_prompt_template() -> str:
    if not FINAL_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {FINAL_PROMPT_PATH}")
    return FINAL_PROMPT_PATH.read_text()


def _format_history(history: list) -> str:
    """Renders recent chat_history as plain text for prompt injection.
    Empty/absent history renders as an explicit "none" rather than an
    empty string, so prompts don't end up with a dangling blank section."""
    if not history:
        return "(no earlier turns in this conversation)"

    recent = history[-MAX_HISTORY_TURNS_IN_PROMPT:]
    lines = []
    for turn in recent:
        lines.append(f"Q: {turn['question']}\nA: {turn['answer']}")
    return "\n\n".join(lines)


def planner_node(state: GraphState) -> dict:
    """Classifies the question as sql / retrieval / both."""
    history = _format_history(state.get("chat_history", []))
    route = route_question(state["question"], history)
    return {"route": route, "execution_path": ["planner"]}


def sql_node(state: GraphState) -> dict:
    """Generates and safely executes SQL against the structured database."""
    history = _format_history(state.get("chat_history", []))
    result = answer_from_sql(state["question"], history)
    return {
        "sql_query": result["sql"],
        "sql_result_text": result["result_text"],
        "execution_path": ["sql"],
    }


def retrieval_node(state: GraphState) -> dict:
    """Retrieves relevant document chunks from the FAISS index.

    Follow-up questions ("what about last quarter?") often embed poorly on
    their own, since the entity/topic they refer to lives in the prior
    turn, not the current one. Folding the last turn's Q&A into the search
    text (not the displayed question — just what gets embedded) noticeably
    improves recall on those without an extra LLM call to rewrite the query.
    """
    history = state.get("chat_history", [])
    if history:
        last_turn = history[-1]
        search_text = f"{last_turn['question']} {last_turn['answer']}\n{state['question']}"
    else:
        search_text = state["question"]

    chunks = retrieve(search_text, k=state.get("k", 4))

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
        sql_result_text = state["sql_result_text"]
        label = "[Structured database results — query FAILED, do not treat as data]" \
            if sql_result_text.startswith("[SQL Error]") else "[Structured database results]"
        parts.append(
            f"{label}\n"
            f"SQL query used: {state.get('sql_query')}\n"
            f"Results:\n{sql_result_text}"
        )

    if state.get("retrieved_context"):
        parts.append(f"[Document excerpts]\n{state['retrieved_context']}")

    combined = "\n\n".join(parts) if parts else None
    return {"combined_context": combined, "execution_path": ["reasoning"]}


def report_node(state: GraphState) -> dict:
    """Generates the final grounded answer from the merged context, and
    records this turn into chat_history so the next invoke() on the same
    thread_id sees it (refusals get recorded too, so a follow-up to a
    refusal still has the right conversational context)."""
    combined = state.get("combined_context")
    history = _format_history(state.get("chat_history", []))

    if not combined:
        answer = "I don't have enough information to answer that."
        return {
            "final_answer": answer,
            "execution_path": ["report"],
            "chat_history": [{"question": state["question"], "answer": answer}],
        }

    template = _load_prompt_template()
    prompt = template.format(context=combined, question=state["question"], history=history)
    answer = call_llm(prompt)

    return {
        "final_answer": answer,
        "execution_path": ["report"],
        "chat_history": [{"question": state["question"], "answer": answer}],
    }
