# Enterprise Agentic RAG

An enterprise question-answering system that can use:

- **Document retrieval** over PDFs in `data/pdfs/`, via hybrid (vector + keyword) search with cross-encoder reranking
- **Read-only SQL** over a Postgres database, grounded by live value-linking
- **LangGraph orchestration** to route, merge, and answer — with multi-turn conversational memory
- **FastAPI** for the backend API
- **Streamlit** for the chat UI
- **An evaluation harness** and a **SQL guardrail test suite** to back up "it works" and "it's safe" with actual numbers, not vibes

The repo contains the current graph-based implementation plus some earlier phase files kept for reference.

## What it does

Given a question, the app can:

1. Decide whether the answer needs SQL, documents, or both — using the current question *and* the conversation so far, so follow-ups like "what about last quarter?" resolve correctly.
2. Retrieve relevant PDF chunks via hybrid search (FAISS + BM25) narrowed down by a cross-encoder reranker.
3. Generate and execute a read-only SQL query against Postgres.
4. Merge the results into one grounded answer with citations.
5. Remember the turn, so the next question in the same conversation can build on it.

## Architecture

Current request flow:

`Streamlit UI -> FastAPI /ask -> LangGraph planner -> SQL and/or retrieval -> reasoning -> final answer`

Every step reads and writes a shared `GraphState` (see `graph/state.py`), including a `chat_history` field
that's persisted across requests by a LangGraph checkpointer keyed on a `session_id` — that's what gives
the agent memory of earlier turns in the same conversation.

### Main components

- `app/main.py` - FastAPI app for the current LangGraph workflow
- `graph/` - LangGraph state, nodes, workflow wiring, and the checkpointer that gives it multi-turn memory
- `tools/sql_tool.py` - LLM-to-SQL generation plus safe execution
- `database/schema_values.py` - semantic value-linking for SQL grounding
- `database/sql_executor.py` - the SQL safety layer, covered by `tests/test_sql_guardrails.py`
- `rag/` - PDF loading, chunking, embedding, indexing, and hybrid retrieval + reranking
- `database/` - Postgres schema, loader, and SQL execution helpers
- `eval/` - golden-set evaluation harness (routing accuracy, grounding, LLM-judged faithfulness/helpfulness)
- `streamlit_app.py` - chat UI that calls the API

## Repository layout

```text
app/
  main.py          FastAPI API for the current graph-based system
  main1.py         Earlier phase 1 FastAPI app kept for reference
  llm.py           Shared LLM provider wrapper
  router.py        Question router for sql / retrieval / both

graph/
  state.py         Shared LangGraph state schema, incl. chat_history
  nodes.py         Planner, SQL, retrieval, reasoning, and report nodes
  workflow.py      Compiles the LangGraph workflow with a MemorySaver checkpointer

rag/
  chunking.py      PDF loading and text splitting
  embeddings.py    Embedding provider wrapper
  vectorstore.py   FAISS build/load/persist helpers
  retriever.py     Hybrid (FAISS + BM25) retrieval, cross-encoder reranking, context formatting

tools/
  sql_tool.py      Generates SQL, runs it safely, formats results

database/
  sample_db.sql    Postgres schema and indexes
  load_data.py     Loads CSVs into the database
  postgres.py      DB connection and schema description
  schema_values.py Live categorical-value index + semantic matching hints
  sql_executor.py  Safety checks and SQL execution helpers

prompts/
  rag.txt          Final grounded-answer prompt
  router.txt       Planner prompt
  sql.txt          SQL-generation prompt

data/
  pdfs/            Source PDFs
  csv/             Structured sample data
  faiss_index/     Generated FAISS cache

tests/
  test_retrieval.py       Retrieval pipeline sanity checks (needs API key + PDFs)
  test_sql_guardrails.py  SQL safety guardrail suite — pure/offline, no DB or LLM needed

eval/
  dataset.json         16 golden Q&A pairs across retrieval/sql/both/refusal, ground-truth verified
  judge_prompt.txt      LLM-as-judge grading template
  run_eval.py            Runs the real graph end-to-end and scores routing/grounding/faithfulness
  results/                Timestamped raw results + latest_report.md

streamlit_app.py   Streamlit chat frontend
requirements.txt
README.md
```

## Data sources

### Documents

Put text-based PDFs in `data/pdfs/`.

Included examples:

- `01_hr_policy.pdf`
- `02_refund_policy.pdf`
- `03_complaints_electronics_q1.pdf`
- `04_complaints_home_kitchen_q2.pdf`
- `05_complaints_apparel_q3.pdf`
- `06_sales_summary_q2.pdf`
- `07_marketing_summary_q3.pdf`

### Structured data

The database schema expects these CSV files in `data/csv/`:

- `customers.csv`
- `products.csv`
- `sample_sales.csv`
- `complaints.csv`

## Setup

### 1) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Create `.env`

Copy `.env.example` and fill in the values:

Required variables:

```bash
GOOGLE_API_KEY=your_key
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini

DB_HOST=localhost
DB_PORT=5432
DB_NAME=enterprise_rag

DB_USER=raguser
DB_PASSWORD=ragpass

DB_READONLY_USER=readonly_user
DB_READONLY_PASSWORD=readonly_password

CSV_DIR=data/csv
```

Optional:

```bash
OPENAI_API_KEY=...
```

`LLM_PROVIDER` and `EMBEDDING_PROVIDER` currently default to Gemini. The code supports OpenAI for the LLM wrapper, but Gemini is the primary path in this repo.

## Database setup

1. Create the schema:

```bash
psql -U raguser -d enterprise_rag -f database/sample_db.sql
```

2. Load the CSV data:

```bash
python -m database.load_data
```

3. Make sure the read-only DB user in `.env` has `SELECT` access.

## Running the backend

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Ask endpoint:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","k":4}'
```

Example response:

```json
{
  "answer": "...",
  "route": "retrieval",
  "sources": [
    {"file": "refund_policy.pdf", "page": 1}
  ],
  "sql_used": null,
  "execution_path": ["planner", "retrieval", "reasoning", "report"]
}
```

## Running the Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI expects the API server to already be running at `http://localhost:8000/ask`.

## How retrieval works

- PDFs are loaded with `PyPDFLoader`
- Pages are chunked with overlap
- Chunks are embedded with Gemini embeddings
- A local FAISS index is created in `data/faiss_index/`
- The index is reused on later runs unless you delete the cache

If you change PDFs, remove `data/faiss_index/` and restart to rebuild the index.

### Hybrid retrieval + reranking

Naive top-k vector search alone misses exact-term matches (product codes, precise policy
phrasing) that embeddings blur together in semantic space. `rag/retriever.py` instead:

1. Runs the query through **both** FAISS (vector/semantic) and BM25 (keyword) search, pulling a
   wider candidate pool from each (`CANDIDATE_POOL_SIZE`, default 15).
2. Fuses the two candidate lists with LangChain's `EnsembleRetriever` (reciprocal rank fusion,
   deduplicated by content).
3. Reranks the fused candidates with a local cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which scores each `(query, chunk)` pair jointly —
   a much stronger relevance signal than independently-computed embedding distance — and keeps
   only the final top-k.

This is the standard retrieve-many/rerank-to-few pattern used in production RAG systems, not
just single-signal top-k similarity search.

## Conversational memory

The graph is compiled with a LangGraph `MemorySaver` checkpointer (`graph/workflow.py`), keyed
by a `session_id`. Pass the `session_id` from a previous `/ask` response back on the next
request to continue that conversation — the planner, SQL agent, and report generator all get
the last few turns folded into their prompts, so follow-ups like *"what about the West region
instead?"* or *"and last quarter?"* resolve correctly instead of being classified/answered in
isolation. Retrieval also folds the prior turn into the search text it embeds (not the displayed
question) so document search benefits from context too, without an extra LLM call to rewrite
the query.

- Memory is in-process only (`MemorySaver`) — it resets when the server restarts, and won't work
  across multiple replicas. A durable checkpointer (Postgres/SQLite) would be the next step for
  a multi-instance deployment.
- The Streamlit UI generates a `session_id` per browser session automatically and issues a fresh
  one when you click "Clear chat history," so clearing the UI actually clears the agent's memory
  too, not just what's displayed.
- Omit `session_id` in the API and you get the old single-turn behavior.

## How SQL works

- A router classifies the question as `sql`, `retrieval`, or `both`
- `database/schema_values.py` fetches distinct live values from key categorical columns
  (like `products.product_line`, `products.category`, `customers.region`, etc.)
- The SQL agent embeds the user question, semantically matches likely value mentions,
  and injects those matches as `value_hints` into `prompts/sql.txt`
- `tools/sql_tool.py` asks the LLM for a single read-only `SELECT` grounded by those hints
- `database/sql_executor.py` blocks unsafe SQL and enforces a row limit
- `database/postgres.py` connects using the read-only DB user

This avoids brittle hardcoded value lists and helps prevent column/value mismatches like
using `"Home & Kitchen"` as `category` when it belongs to `product_line` in the current data.

### SQL value-linking smoke test

You can run the value-linking module directly:

```bash
python -m database.schema_values
```

It will:

- build an in-memory index from live DB distinct values
- run sample semantic matches
- print hint candidates used by SQL generation

### Common SQL mismatch troubleshooting

If a SQL question returns no rows unexpectedly, first verify whether the filter value belongs to
the correct column.

Example:

- Question: `Which product had the highest total revenue in the Home & Kitchen category, and how many units were sold?`
- Common failure: model uses `WHERE p.category = 'Home & Kitchen'`
- In this dataset, `Home & Kitchen` is a `product_line` value, not a `category` value

Expected behavior with value-linking enabled:

- semantic matching surfaces `products.product_line = 'Home & Kitchen'`
- SQL generation uses that hint in the `WHERE` clause
- query returns real rows (if matching data exists)

Quick checks:

1. Run `python -m database.schema_values` and confirm the hint contains
   `products.product_line = 'Home & Kitchen'`
2. Ensure the API is restarted after code/prompt changes
3. Re-run the same question and inspect the generated SQL in Streamlit

## Tests

```bash
pytest tests/
```

- `tests/test_sql_guardrails.py` is **offline** — no DB, no LLM, no API key. It runs ~30
  parametrized cases against `database/sql_executor.is_safe_sql`/`enforce_row_limit`: every
  `FORBIDDEN_KEYWORDS` entry, comment-based LIMIT bypass (`--`, `/*`), stacked statements,
  case-insensitivity, non-SELECT statement types, a regression test for the exact false-positive
  the code explicitly guards against (`updated_at` vs. `update`), and a documented *known*
  false-positive tradeoff (a string literal like `category = 'update'` gets blocked too — the
  blocklist is a text scan, not a parser, and false positives are an accepted cost for never
  having a false negative). This is the second of two independent defenses; the DB connection
  itself uses a read-only user as the first (see `database/postgres.py`).
- `tests/test_retrieval.py` needs PDFs in `data/pdfs/`, a valid API key, and the embedding
  provider configured, since it exercises the real hybrid retrieval pipeline end-to-end.

## Evaluation

"It works" is backed by a golden-set evaluation harness, not just spot-checking answers by hand.

```bash
python -m eval.run_eval
```

For each of the 16 questions in `eval/dataset.json` (spanning `retrieval`, `sql`, `both`, and
`refusal` categories — including the exact `Home & Kitchen` category/product_line regression
case from above, with the correct answer independently verified against the live DB), this:

1. Runs the real compiled graph end-to-end (same code path as the API).
2. Checks **routing accuracy** — did the planner pick the expected source(s)?
3. Checks **grounding** — does the answer contain the independently-verified ground-truth facts
   (`must_contain`)? Deterministic, catches wrong numbers/names cheaply.
4. Scores **faithfulness** and **helpfulness** via an LLM-as-judge call that only sees the
   context the graph actually used, so it can catch hallucinated elaboration that happens to
   also mention the right keywords.

Results are written to `eval/results/latest_report.md` (a metrics table plus a "failures worth
reading" section) and a timestamped raw JSON. Two LLM calls per question (system-under-test +
judge) on top of the system's own 2-3 calls per question — on Gemini's free tier (20
requests/day for `gemini-2.5-flash`), a full run can exceed the daily cap partway through; it'll
report how far it got and can be re-run once the quota resets.

## Notes

- `app/main1.py` is the older phase 1-only backend.
- `data/faiss_index/` is generated, not source data.
- The app is intentionally read-only for SQL execution.
- Conversational memory (`MemorySaver`) is in-process only — see "Conversational memory" above.
