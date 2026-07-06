"""
tools/sql_tool.py

Takes a natural language question, generates a read-only SQL query
against the known schema, executes it safely, and returns readable results.
"""

from pathlib import Path

from app.llm import call_llm
from database.postgres import SCHEMA_DESCRIPTION
from database.sql_executor import execute_sql, format_sql_results

SQL_PROMPT_PATH = Path("prompts/sql.txt")


def _load_sql_prompt_template() -> str:
    if not SQL_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {SQL_PROMPT_PATH}")
    return SQL_PROMPT_PATH.read_text()


def _clean_generated_sql(raw: str) -> str:
    """LLMs sometimes wrap SQL in markdown fences despite instructions — strip them."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove a leading language tag like "sql\n"
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:]
    return cleaned.strip()


def answer_from_sql(question: str) -> dict:
    """
    Full SQL agent pipeline: question -> generated SQL -> executed -> formatted text.

    Returns:
        {
            "sql": the generated SQL string,
            "result_text": human-readable results (or error) for the final prompt,
            "success": bool
        }
    """
    template = _load_sql_prompt_template()
    prompt = template.format(schema=SCHEMA_DESCRIPTION, question=question)

    generated = call_llm(prompt)
    sql = _clean_generated_sql(generated)

    if sql.upper() == "NO_QUERY":
        return {
            "sql": None,
            "result_text": "This question cannot be answered from the structured database.",
            "success": False,
        }

    result = execute_sql(sql)
    result_text = format_sql_results(result)

    return {
        "sql": sql,
        "result_text": result_text,
        "success": result["success"],
    }


if __name__ == "__main__":
    # Quick manual smoke test: python -m tools.sql_tool
    out = answer_from_sql("Which region had the highest total sales revenue?")
    print("Generated SQL:", out["sql"])
    print("Result:\n", out["result_text"])
