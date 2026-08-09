"""
graph/state.py

Shared state passed between every node in the LangGraph workflow.
Each node reads what it needs and writes back a partial update —
LangGraph merges these updates automatically.

execution_path uses an additive reducer (operator.add) because
sql_node and retrieval_node can run in parallel (when route == "both"),
and both need to append their name without overwriting each other.

chat_history persists across invoke() calls that share the same
checkpointer thread_id (see graph/workflow.py's MemorySaver), which is
what gives the graph multi-turn memory: within one turn, only report_node
appends to it, but the checkpointer carries the accumulated list into the
*next* turn's initial state automatically.
"""

import operator
from typing import TypedDict, Optional, List, Annotated

# Bounds chat_history so a long-running conversation/process doesn't grow
# the checkpointed state (and every downstream prompt) without limit.
MAX_HISTORY_TURNS = 6


def _append_and_cap_history(existing: List[dict], new: List[dict]) -> List[dict]:
    return (existing + new)[-MAX_HISTORY_TURNS:]


class GraphState(TypedDict):
    # Input
    question: str
    k: int

    # Planner output
    route: str

    # SQL agent output
    sql_query: Optional[str]
    sql_result_text: Optional[str]

    # Retrieval agent output
    retrieved_context: Optional[str]
    sources: List[dict]

    # Reasoning output
    combined_context: Optional[str]

    # Final output
    final_answer: Optional[str]

    # Debug/visualization: which nodes fired, in order encountered
    execution_path: Annotated[List[str], operator.add]

    # Conversation memory: {"question": ..., "answer": ...} per completed turn,
    # oldest first. Persisted across turns via the graph's checkpointer.
    chat_history: Annotated[List[dict], _append_and_cap_history]
