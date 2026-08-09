"""
graph/workflow.py

Wires the nodes into an actual graph:

                    planner
                   /        \\
              sql_node   retrieval_node      (fan-out: both run if route == "both")
                   \\        /
                  reasoning_node
                        |
                   report_node
                        |
                       END

Compiled once at import time; app/main.py imports `graph` and calls .invoke().
"""

from typing import List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from graph.state import GraphState
from graph.nodes import planner_node, sql_node, retrieval_node, reasoning_node, report_node


def _route_from_planner(state: GraphState) -> List[str]:
    """Fan-out logic: returns the list of next node(s) based on the planner's decision."""
    route = state["route"]
    if route == "sql":
        return ["sql_node"]
    elif route == "retrieval":
        return ["retrieval_node"]
    else:  # "both"
        return ["sql_node", "retrieval_node"]


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("report_node", report_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", _route_from_planner)

    # Both branches converge on reasoning_node — LangGraph waits for all
    # active incoming branches before running it.
    workflow.add_edge("sql_node", "reasoning_node")
    workflow.add_edge("retrieval_node", "reasoning_node")

    workflow.add_edge("reasoning_node", "report_node")
    workflow.add_edge("report_node", END)

    # MemorySaver is an in-process, in-memory checkpointer: state persists
    # for the life of this Python process, keyed by thread_id, and is lost
    # on restart. That's the right tradeoff for a single-instance app like
    # this one — a durable store (Postgres/SQLite checkpointer) would be
    # the next step if this ever ran as multiple replicas.
    #
    # Any invoke() against this compiled graph MUST pass
    # config={"configurable": {"thread_id": "<session id>"}} — LangGraph
    # raises if a checkpointer is set and no thread_id is given.
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Compiled once, reused across requests.
graph = build_graph()


