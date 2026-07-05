# Enterprise Agentic RAG — Phase 1

A grounded, cited question-answering API over enterprise-style PDFs (policies, complaint reports, etc.).
This is Phase 1 of a larger multi-agent project — see project plan for Phases 2–4 (SQL agent,
LangGraph multi-agent workflow, evaluation + deployment).

## Locked tech decisions (Week 0)

- **LLM**: Gemini (`gemini-1.5-flash`) — swap via `LLM_PROVIDER` in `.env`
- **Embeddings**: Gemini `text-embedding-004` — swap via `EMBEDDING_PROVIDER` in `.env`
- **Vector DB**: FAISS (local, in-process) — will move to Pinecone in Phase 4
- **Backend**: FastAPI

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in your API key
```

Add a few PDFs to `data/pdfs/` (policies, reports, anything text-based).

## Run

```bash
uvicorn app.main:app --reload
```

Then either open the interactive docs at `http://localhost:8000/docs`, or:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "k": 4}'
```

Response:

```json
{
  "answer": "...grounded answer with [1] style citations...",
  "sources": [
    {"file": "refund_policy.pdf", "page": 1}
  ]
}
```

## First run notes

- The first call builds the FAISS index from `data/pdfs/` and saves it to `data/faiss_index/`
  (this can take a minute depending on how many PDFs you have). Subsequent runs load the
  cached index instantly.
- Delete `data/faiss_index/` and restart if you add/change PDFs, to force a rebuild.

## Tests

```bash
pytest tests/
```

(Requires PDFs in `data/pdfs/` and a valid API key, since tests hit the real embedding provider.)

## Project structure

```
app/main.py          - FastAPI app, /ask endpoint
rag/chunking.py       - PDF loading + text splitting
rag/embeddings.py     - embedding provider wrapper
rag/vectorstore.py    - FAISS index build/load/persist
rag/retriever.py      - top-k retrieval + context formatting
prompts/rag.txt       - grounded-answer prompt template
tests/                - sanity tests
```

## Next phases

- **Phase 2**: add a Postgres-backed SQL agent + router (SQL vs. document questions)
- **Phase 3**: convert router into a LangGraph multi-agent workflow (planner, SQL, retrieval, reasoning, report)
- **Phase 4**: RAGAS/DeepEval evaluation, observability, Docker, deployment
