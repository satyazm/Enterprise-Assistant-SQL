"""
app/router.py

Classifies an incoming question as needing SQL data, document retrieval,
or both, before Phase 3 turns this into a full LangGraph planner node.
"""

from pathlib import Path
from app.llm import call_llm

ROUTER_PROMPT_PATH = Path("prompts/router.txt")

VALID_LABELS = {"sql", "retrieval", "both"}


def _load_router_prompt_template() -> str:
    if not ROUTER_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {ROUTER_PROMPT_PATH}")
    return ROUTER_PROMPT_PATH.read_text()


def route_question(question: str, history: str = "(no earlier turns in this conversation)") -> str:
    """Returns one of: 'sql', 'retrieval', 'both'. Defaults to 'both' if the
    LLM's output is unparseable, since that's the safest fallback (better to
    over-fetch than to miss relevant context).

    `history` lets a follow-up like "what about last quarter?" route
    correctly even though it has no source/topic keywords of its own —
    without it, the classifier only ever sees the bare follow-up text."""
    template = _load_router_prompt_template()
    prompt = template.format(question=question, history=history)

    raw = call_llm(prompt).strip().lower()

    for label in VALID_LABELS:
        if label in raw:
            return label

    return "both"
