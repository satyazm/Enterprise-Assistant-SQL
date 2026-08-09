"""
database/schema_values.py

Solves the "value linking" problem in text-to-SQL: the LLM knows column
NAMES from the schema description, but not which VALUES actually live in
which column (e.g. is "Home & Kitchen" a category or a product_line?).

Instead of hardcoding those values into a prompt (brittle -- goes stale
the moment the data changes), this module:
  1. Pulls distinct values for known categorical/enum-like columns
     LIVE from the database.
  2. Embeds them once and keeps them in memory.
  3. At query time, embeds the user's question and finds the closest
     matching (table, column, value) triples semantically.
  4. Those matches get injected into the SQL generation prompt as
     "detected value hints" -- grounding the model in what's actually
     in the database right now.

LOOKUP_COLUMNS below is structural metadata (which columns are worth
indexing), not a hardcoded list of values -- the values themselves are
pulled live from the database, so no value list to maintain by hand.

Note: the index is built once (lazily, on first use) and then cached
in memory for the life of the process -- it does NOT auto-refresh if
the underlying data changes later. Call build_value_index() again
(e.g. after a data reload) to pick up new/changed values.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sqlalchemy import text

from database.postgres import get_engine
from rag.embeddings import get_embedding_model

# (table, column) pairs worth indexing: low-cardinality, categorical-ish
# columns that tend to show up in natural language filters. Add to this
# list if you add new filterable columns later -- no values to maintain.
LOOKUP_COLUMNS = [
    ("products", "product_line"),
    ("products", "category"),
    ("customers", "region"),
    ("customers", "loyalty_tier"),
    ("sales", "channel"),
    ("complaints", "status"),
    ("complaints", "channel"),
]


@dataclass
class ValueEntry:
    table: str
    column: str
    value: str


_value_entries: List[ValueEntry] = []
_value_embeddings: np.ndarray = None
_embedding_model = None


def _fetch_distinct_values() -> List[ValueEntry]:
    """Pulls current distinct values for every lookup column, live from the DB."""
    engine = get_engine()
    entries = []

    with engine.connect() as conn:
        for table, column in LOOKUP_COLUMNS:
            result = conn.execute(
                text(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL")
            )
            for row in result:
                value = row[0]
                if value:
                    entries.append(ValueEntry(table=table, column=column, value=str(value)))

    return entries


def build_value_index():
    """Builds (or rebuilds) the in-memory value index. Called lazily on first
    use, or can be called explicitly at app startup / after data changes."""
    global _value_entries, _value_embeddings, _embedding_model

    _embedding_model = get_embedding_model()
    _value_entries = _fetch_distinct_values()

    if not _value_entries:
        _value_embeddings = np.zeros((0, 1))
        return

    texts = [f"{e.column}: {e.value}" for e in _value_entries]
    vectors = _embedding_model.embed_documents(texts)
    _value_embeddings = np.array(vectors)


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / matrix_norms
    return matrix_norm @ query_norm


def find_relevant_values(
    question: str, top_k: int = 5, min_score: float = 0.55
) -> List[Tuple[ValueEntry, float]]:
    """
    Returns up to top_k (ValueEntry, similarity_score) pairs whose value is
    semantically closest to something mentioned in the question, above
    min_score. Empty list if nothing matches confidently -- callers should
    treat that as "no hint available", not an error.
    """
    global _value_entries, _value_embeddings, _embedding_model

    if _embedding_model is None or not _value_entries:
        build_value_index()

    if not _value_entries:
        return []

    query_vec = np.array(_embedding_model.embed_query(question))
    scores = _cosine_similarity(query_vec, _value_embeddings)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (_value_entries[i], float(scores[i]))
        for i in top_indices
        if scores[i] >= min_score
    ]


def format_value_hints(matches: List[Tuple[ValueEntry, float]]) -> str:
    """Formats matches into a prompt-ready hint block. Empty string if no matches."""
    if not matches:
        return ""

    lines = ["Detected value matches (use these EXACT values in WHERE clauses when relevant):"]
    for entry, score in matches:
        lines.append(f"  - {entry.table}.{entry.column} = '{entry.value}'")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual smoke test: python -m database.schema_values
    build_value_index()
    print(f"Indexed {len(_value_entries)} distinct values across {len(LOOKUP_COLUMNS)} columns.")

    test_questions = [
        "products in the home and kitchen category",
        "customers in the northeast",
        "complaints that are still open",
    ]
    for q in test_questions:
        matches = find_relevant_values(q)
        print(f"\nQ: {q}")
        print(format_value_hints(matches) or "  (no confident matches)")
