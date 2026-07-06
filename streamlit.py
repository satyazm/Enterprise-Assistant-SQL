"""
frontend/streamlit_app.py

Continuous chat interface for the Enterprise Agentic Knowledge Platform.
Calls the existing FastAPI /ask endpoint (no logic duplicated here) and
keeps a running conversation history for the session, with each turn's
execution trace, generated SQL, and document sources shown inline.

Run with:
    streamlit run frontend/streamlit_app.py

(Requires the FastAPI server to be running separately:
    uvicorn app.main:app --reload
)
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"

NODE_LABELS = {
    "planner": "🧭 Planner",
    "sql": "🗄️ SQL Agent",
    "retrieval": "📄 Retrieval Agent",
    "reasoning": "🧩 Reasoning",
    "report": "📝 Report",
}

st.set_page_config(page_title="Enterprise Agentic Knowledge Platform", layout="centered")

st.title("Enterprise Agentic Knowledge Platform")
st.caption("Ask a question — the system will route it through SQL, documents, or both.")

# --- Session state: chat history persists across turns within this session ---
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, one per turn

with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant answers questions using:\n"
        "- **Structured data**: customers, products, sales, complaints\n"
        "- **Documents**: company policies and reports\n\n"
        "A planner agent decides which source(s) to use, then a "
        "reasoning step merges the results into one answer."
    )
    st.divider()
    st.markdown("**Example questions:**")
    st.markdown(
        "- Which region had the highest total sales revenue?\n"
        "- What is the refund policy?\n"
        "- Which customers complained about a product and what was their total purchase value?"
    )
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.rerun()


def render_trace(trace: list, route: str):
    trace_display = "  →  ".join(NODE_LABELS.get(node, node) for node in trace)
    st.markdown(f"**{trace_display}**")
    st.caption(f"Route selected: `{route}`")


def render_assistant_turn(turn: dict):
    """Renders one assistant turn: trace, answer, SQL, sources."""
    render_trace(turn.get("execution_path", []), turn.get("route", "unknown"))
    st.markdown(turn.get("answer", "No answer returned."))

    if turn.get("sql_used"):
        with st.expander("🗄️ Generated SQL"):
            st.code(turn["sql_used"], language="sql")

    sources = turn.get("sources", [])
    if sources:
        with st.expander(f"📄 Document sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- **{s.get('file', 'unknown')}**, page {s.get('page', '?')}")


# --- Render existing history ---
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_assistant_turn(turn)

# --- Chat input, anchored at the bottom ---
question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing through the agent workflow..."):
            try:
                response = requests.post(API_URL, json={"question": question, "k": 4}, timeout=60)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the API. Make sure it's running:\n\n"
                    "`uvicorn app.main:app --reload`"
                )
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()

        turn = {
            "question": question,
            "answer": data.get("answer", "No answer returned."),
            "route": data.get("route", "unknown"),
            "execution_path": data.get("execution_path", []),
            "sql_used": data.get("sql_used"),
            "sources": data.get("sources", []),
        }
        render_assistant_turn(turn)

    # Save this turn so it persists across reruns (e.g. when the next question is asked)
    st.session_state.history.append(turn)