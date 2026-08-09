"""
tests/test_sql_guardrails.py

Guardrail test suite for the LLM-to-SQL safety layer (database/sql_executor.py).

This is deliberately offline: it only exercises the pure functions
is_safe_sql() and enforce_row_limit(), never a live database or LLM call,
so it runs in CI with no credentials and no network. It's the second of
two independent defenses (database/postgres.py's read-only DB user is the
first) — see the module docstring in sql_executor.py.

Each adversarial case documents *why* it's dangerous if it slipped through
and *which* guard is supposed to catch it, so this doubles as a walkthrough
of the SQL agent's threat model, not just a pass/fail list.
"""

import pytest

from database.sql_executor import is_safe_sql, enforce_row_limit


# --- Attacks that must be blocked -------------------------------------------
#
# (id, payload, why it's dangerous)
ADVERSARIAL_PAYLOADS = [
    ("drop-table", "DROP TABLE customers;", "Direct data-destroying DDL."),
    ("delete-rows", "DELETE FROM sales WHERE 1=1;", "Would wipe every row in sales."),
    ("update-rows", "UPDATE products SET unit_price = 0;", "Would corrupt pricing data."),
    ("insert-rows", "INSERT INTO customers (customer_id) VALUES ('x');", "Unauthorized write."),
    ("alter-table", "ALTER TABLE customers ADD COLUMN backdoor TEXT;", "Schema tampering."),
    ("truncate-table", "TRUNCATE TABLE complaints;", "Wipes a whole table instantly, no WHERE needed."),
    ("grant-privileges", "GRANT ALL PRIVILEGES ON customers TO PUBLIC;", "Privilege escalation."),
    ("revoke-privileges", "REVOKE SELECT ON customers FROM readonly_user;", "Could lock out legitimate access."),
    ("create-table", "CREATE TABLE staging AS SELECT * FROM customers;", "Unauthorized schema change / data exfil via new table."),
    ("attach-database", "ATTACH DATABASE '/tmp/evil.db' AS evil;", "Cross-database access attempt."),
    ("copy-to-file", "COPY customers TO '/tmp/dump.csv';", "Bulk data exfiltration to the filesystem."),
    (
        "stacked-statement-with-keyword",
        "SELECT * FROM customers; DROP TABLE customers;",
        "Classic stacked-query injection — second statement is destructive.",
    ),
    (
        "stacked-statement-no-keyword",
        "SELECT * FROM customers; SELECT * FROM products;",
        "Stacking defense must catch this even with no forbidden keyword present — "
        "isolates the semicolon-count check from the keyword check.",
    ),
    (
        "line-comment-bypass",
        "SELECT * FROM sales -- ignore everything after this",
        "A trailing -- comment would otherwise swallow the LIMIT clause "
        "enforce_row_limit() appends as plain text, defeating the row cap.",
    ),
    (
        "block-comment-bypass",
        "SELECT * FROM sales /* sneaky */ WHERE 1=1",
        "Same class of attack as line comments, different syntax.",
    ),
    (
        "comment-hides-keyword",
        "SELECT * FROM customers -- '; DROP TABLE customers; --",
        "Keyword is inside a comment, so the keyword loop alone wouldn't catch it — "
        "must be caught by the comment check.",
    ),
    (
        "mixed-case-keyword",
        "SeLeCT * FROM customers; DrOp TaBLE customers;",
        "Confirms keyword matching isn't case-sensitive (normalized.lower() upstream).",
    ),
    (
        "non-select-statement-explain",
        "EXPLAIN SELECT * FROM customers;",
        "Doesn't start with SELECT/WITH — outside the allowed statement shape.",
    ),
    (
        "non-select-statement-show",
        "SHOW ALL;",
        "Same as above — a non-SELECT statement type.",
    ),
]


@pytest.mark.parametrize("case_id,payload,reason", ADVERSARIAL_PAYLOADS, ids=[c[0] for c in ADVERSARIAL_PAYLOADS])
def test_adversarial_payload_is_blocked(case_id, payload, reason):
    assert is_safe_sql(payload) is False, f"{case_id} should be blocked: {reason}"


# --- Legitimate queries that must NOT be blocked ----------------------------

SAFE_PAYLOADS = [
    ("plain-select", "SELECT * FROM customers"),
    ("select-with-where", "SELECT customer_id, first_name FROM customers WHERE region = 'West'"),
    (
        "select-with-cte",
        "WITH regional AS (SELECT region, SUM(line_total) AS rev FROM sales GROUP BY region) "
        "SELECT * FROM regional ORDER BY rev DESC",
    ),
    ("select-with-join", "SELECT c.first_name, s.line_total FROM customers c JOIN sales s ON c.customer_id = s.customer_id"),
    ("select-already-has-limit", "select * from sales limit 5"),
    (
        "column-name-contains-keyword-substring",
        "SELECT updated_at FROM sales",
        # Regression test for the exact false-positive this code explicitly
        # guards against (see the comment in is_safe_sql): "updated_at"
        # contains "update" as a substring but not as a whole word, since
        # \b requires a boundary immediately after "update" and the next
        # character is "d", not a boundary.
    ),
]


@pytest.mark.parametrize("case_id,payload", [(c[0], c[1]) for c in SAFE_PAYLOADS], ids=[c[0] for c in SAFE_PAYLOADS])
def test_legitimate_query_is_allowed(case_id, payload):
    assert is_safe_sql(payload) is True, f"{case_id} is a legitimate query and should be allowed"


# --- Known, accepted limitation ----------------------------------------------

def test_known_limitation_keyword_inside_string_literal_is_a_false_positive():
    """The blocklist is a dumb text scan, not a SQL parser — it can't tell a
    keyword used as a string VALUE from one used as a SQL command. This is a
    deliberate tradeoff: false positives (blocking a legitimate query) are a
    usability nuisance; false negatives (letting a real DROP through) are a
    security incident. Documented here so it's a known tradeoff, not a
    surprise bug report."""
    query = "SELECT * FROM products WHERE category = 'update'"
    assert is_safe_sql(query) is False


# --- Row-limit enforcement ---------------------------------------------------

def test_enforce_row_limit_appends_limit_when_missing():
    result = enforce_row_limit("SELECT * FROM sales")
    assert result == "SELECT * FROM sales LIMIT 100"


def test_enforce_row_limit_leaves_existing_limit_untouched():
    result = enforce_row_limit("SELECT * FROM sales LIMIT 10")
    assert result == "SELECT * FROM sales LIMIT 10"


def test_enforce_row_limit_strips_trailing_semicolon_before_appending():
    result = enforce_row_limit("SELECT * FROM sales;")
    assert result == "SELECT * FROM sales LIMIT 100"


def test_enforce_row_limit_respects_custom_max_rows():
    result = enforce_row_limit("SELECT * FROM sales", max_rows=25)
    assert result == "SELECT * FROM sales LIMIT 25"
