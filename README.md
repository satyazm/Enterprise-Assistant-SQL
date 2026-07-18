# Enterprise Agentic RAG

An enterprise question-answering system that can use:

- **Document retrieval** over PDFs in `data/pdfs/`
- **Read-only SQL** over a Postgres database
- **LangGraph orchestration** to route, merge, and answer
- **FastAPI** for the backend API
- **Streamlit** for the chat UI

The repo contains the current graph-based implementation plus some earlier phase files kept for reference.

## What it does

Given a question, the app can:

1. Decide whether the answer needs SQL, documents, or both.
2. Retrieve relevant PDF chunks from a cached FAISS index.
3. Generate and execute a read-only SQL query against Postgres.
4. Merge the results into one grounded answer with citations.

## Architecture

Current request flow:

`Streamlit UI -> FastAPI /ask -> LangGraph planner -> SQL and/or retrieval -> reasoning -> final answer`

### Main components

- `app/main.py` - FastAPI app for the current LangGraph workflow
- `graph/` - LangGraph state, nodes, and workflow wiring
- `tools/sql_tool.py` - LLM-to-SQL generation plus safe execution
- `database/schema_values.py` - semantic value-linking for SQL grounding
- `rag/` - PDF loading, chunking, embedding, indexing, and retrieval
- `database/` - Postgres schema, loader, and SQL execution helpers
- `streamlit.py` - chat UI that calls the API

## Repository layout

```text
app/
  main.py          FastAPI API for the current graph-based system
  main1.py         Earlier phase 1 FastAPI app kept for reference
  llm.py           Shared LLM provider wrapper
  router.py        Question router for sql / retrieval / both

graph/
  state.py         Shared LangGraph state schema
  nodes.py         Planner, SQL, retrieval, reasoning, and report nodes
  workflow.py      Compiles the LangGraph workflow

rag/
  chunking.py      PDF loading and text splitting
  embeddings.py    Embedding provider wrapper
  vectorstore.py   FAISS build/load/persist helpers
  retriever.py     Top-k retrieval and context formatting

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
  test_retrieval.py

streamlit.py       Streamlit chat frontend
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
streamlit run streamlit.py
```

The UI expects the API server to already be running at `http://localhost:8000/ask`.

## How retrieval works

- PDFs are loaded with `PyPDFLoader`
- Pages are chunked with overlap
- Chunks are embedded with Gemini embeddings
- A local FAISS index is created in `data/faiss_index/`
- The index is reused on later runs unless you delete the cache

If you change PDFs, remove `data/faiss_index/` and restart to rebuild the index.

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

The retrieval tests need:

- PDFs in `data/pdfs/`
- a valid API key
- the embedding provider configured

## Notes

- `app/main1.py` is the older phase 1-only backend.
- `data/faiss_index/` is generated, not source data.
- The app is intentionally read-only for SQL execution.
