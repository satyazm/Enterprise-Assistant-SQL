"""
database/postgres.py

Single place the app connects to Postgres from. Uses the READ-ONLY
DB user deliberately — the app (and any LLM-generated SQL) should
never be able to write to the database.
"""

import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "enterprise_rag")

DB_READONLY_USER = os.getenv("DB_READONLY_USER")
DB_READONLY_PASSWORD = os.getenv("DB_READONLY_PASSWORD")

if not DB_READONLY_USER or not DB_READONLY_PASSWORD:
    raise EnvironmentError(
        "DB_READONLY_USER / DB_READONLY_PASSWORD not set in .env. "
        "Create the read-only DB user first (see database/sample_db.sql setup steps)."
    )

READONLY_DB_URL = (
    f"postgresql://{DB_READONLY_USER}:{DB_READONLY_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

_engine = None


def get_engine():
    """Returns a singleton SQLAlchemy engine connected as the read-only user."""
    global _engine
    if _engine is None:
        _engine = create_engine(READONLY_DB_URL, pool_pre_ping=True)
    return _engine


# Human-readable schema description injected into the SQL-generation prompt.
# Keep this in sync with database/sample_db.sql if you change columns.
SCHEMA_DESCRIPTION = """
Table: customers
  customer_id (VARCHAR, PK), first_name, last_name, email, phone,
  street_address, city, state, zip_code (INTEGER), region, loyalty_tier, signup_date (DATE)

Table: products
  product_id (VARCHAR, PK), product_name, product_line, category,
  unit_price (NUMERIC), unit_cost (NUMERIC), margin_pct (NUMERIC)

  Note: product_line is a broad grouping (e.g. "Home & Kitchen"); category is a
  more specific subcategory within it (e.g. "Cookware"). If the question's exact
  wording appears in the "Detected value matches" hints below, use those exact
  values -- they're pulled live from the database and are more reliable than guessing.

Table: sales
  sale_id (VARCHAR, PK), sale_date (DATE), customer_id (VARCHAR, FK -> customers),
  product_id (VARCHAR, FK -> products), product_name, product_line,
  quantity (INTEGER), unit_price (NUMERIC), discount_pct (INTEGER),
  line_total (NUMERIC), channel, region

Table: complaints
  complaint_id (VARCHAR, PK), complaint_date (DATE), customer_id (VARCHAR, FK -> customers),
  product_id (VARCHAR, FK -> products), product_name, product_line, category,
  channel, status, resolution_days (NUMERIC, nullable - NULL means unresolved),
  satisfaction_score (NUMERIC, nullable - NULL means unresolved)
""".strip()