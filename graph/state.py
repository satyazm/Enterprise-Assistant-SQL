"""
graph/state.py

Shared state passed between every node in the LangGraph workflow.
Each node reads what it needs and writes back a partial update —
LangGraph merges these updates automatically.

execution_path uses an additive reducer (operator.add) because
sql_node and retrieval_node can run in parallel (when route == "both"),
and both need to append their name without overwriting each other.
"""

import operator
from typing import TypedDict, Optional, List, Annotated


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
