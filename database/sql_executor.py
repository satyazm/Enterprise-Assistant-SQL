"""
database/sql_executor.py

Validates and safely executes LLM-generated SQL. Two layers of defense:
  1. This blocklist check (belt)
  2. The DB connection itself uses a read-only user with SELECT-only grants (suspenders)

Never execute LLM-generated SQL against a connection with write access,
even with this blocklist in place — the blocklist is a backstop, not
the primary control.
"""

import re
from sqlalchemy import text
from database.postgres import get_engine

FORBIDDEN_KEYWORDS = [
    "drop", "delete", "update", "insert", "alter", "truncate",
    "grant", "revoke", "create", "attach", "copy", "--", ";--",
]

MAX_ROWS = 100


def is_safe_sql(sql: str) -> bool:
    """Reject anything that isn't a plain SELECT, or that contains forbidden keywords."""
    normalized = sql.strip().lower()

    if not normalized.startswith("select"):
        return False

    for keyword in FORBIDDEN_KEYWORDS:
        # word-boundary match so e.g. "updated_at" doesn't trigger on "update"
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return False

    # Reject multiple statements stacked with a semicolon
    if normalized.rstrip(";").count(";") > 0:
        return False

    return True


def enforce_row_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Append a LIMIT clause if the query doesn't already have one."""
    normalized = sql.strip().rstrip(";")
    if re.search(r"\blimit\s+\d+\b", normalized, re.IGNORECASE):
        return normalized
    return f"{normalized} LIMIT {max_rows}"


def execute_sql(sql: str):
    """
    Validates and executes a SQL string.
    Returns: {"success": True, "columns": [...], "rows": [...]} or
             {"success": False, "error": "..."}
    """
    if not is_safe_sql(sql):
        return {
            "success": False,
            "error": "Generated SQL failed safety validation (must be a single read-only SELECT).",
        }

    safe_sql = enforce_row_limit(sql)
    engine = get_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text(safe_sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return {"success": True, "columns": columns, "rows": rows}
    except Exception as e:
        return {"success": False, "error": f"SQL execution failed: {str(e)}"}


def format_sql_results(result: dict) -> str:
    """Turn execute_sql()'s output into a readable string for the LLM prompt."""
    if not result["success"]:
        return f"[SQL Error] {result['error']}"

    if not result["rows"]:
        return "Query executed successfully but returned no rows."

    columns = result["columns"]
    lines = [" | ".join(columns)]
    for row in result["rows"][:MAX_ROWS]:
        lines.append(" | ".join(str(row[c]) for c in columns))

    return "\n".join(lines)
