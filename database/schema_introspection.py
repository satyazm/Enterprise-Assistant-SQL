"""
database/schema_introspection.py

Automatically builds a schema description for the SQL-generation prompt by
querying Postgres's own catalogs, instead of relying solely on the
hand-maintained SCHEMA_DESCRIPTION string in database/postgres.py — which
drifts the moment someone adds a column or table and forgets to update it.

Uses pg_catalog rather than information_schema.table_constraints for PK/FK
detection deliberately: information_schema.table_constraints only returns
rows for a role that has some privilege *beyond* plain SELECT on the table
(documented Postgres behavior) — since this app connects with a
deliberately SELECT-only read-only user (see database/postgres.py), that
view is silently empty for us and every column would look like a non-key
column. pg_catalog's own tables (pg_constraint, pg_attribute) don't have
that restriction and are readable by any role, which is what makes this
actually work under the same read-only user everything else here uses.

Queried fresh on every call, deliberately not cached: these are cheap
catalog queries (a few ms even on schemas much bigger than this one), so
unlike schema_values.py's cached value index, there's no staleness
tradeoff worth accepting here.

This only produces STRUCTURE — tables, columns, types, PK/FK relationships.
It can't know business semantics like "product_line is broader than
category" — that kind of domain knowledge still lives in
database/postgres.py's SCHEMA_NOTES and gets appended by the caller
regardless of whether the live or static schema description is in use.
"""

from typing import Dict, List

from sqlalchemy import text

from database.postgres import get_engine

_COLUMNS_QUERY = """
SELECT table_name, column_name, data_type, ordinal_position
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
"""

# contype: 'p' = primary key, 'f' = foreign key. confrelid is only
# meaningful for foreign keys (the table it references); NULL/'-' for PKs.
_CONSTRAINTS_QUERY = """
SELECT
    con.conrelid::regclass::text AS table_name,
    a.attname AS column_name,
    con.contype,
    NULLIF(con.confrelid, 0)::regclass::text AS to_table
FROM pg_constraint con
JOIN pg_attribute a
  ON a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
WHERE con.contype IN ('p', 'f')
  AND con.connamespace = 'public'::regnamespace;
"""


def get_live_schema_description() -> str:
    """Introspects every table/column in the public schema (with data
    types, primary keys, and foreign key relationships) and formats it the
    same way as the hand-written SCHEMA_DESCRIPTION, so it drops into the
    SQL-generation prompt as a straight swap.

    Raises if the DB is unreachable or the public schema has no tables —
    callers should catch that and fall back to the static description
    rather than silently generating an empty/misleading prompt."""
    engine = get_engine()

    with engine.connect() as conn:
        columns = conn.execute(text(_COLUMNS_QUERY)).fetchall()
        constraints = conn.execute(text(_CONSTRAINTS_QUERY)).fetchall()

    if not columns:
        raise ValueError("information_schema returned no columns for the public schema.")

    primary_keys = {(row.table_name, row.column_name) for row in constraints if row.contype == "p"}
    foreign_keys = {(row.table_name, row.column_name): row.to_table for row in constraints if row.contype == "f"}

    tables: Dict[str, List[str]] = {}
    for row in columns:
        key = (row.table_name, row.column_name)
        descriptor = f"{row.column_name} ({row.data_type.upper()}"
        if key in primary_keys:
            descriptor += ", PK"
        if key in foreign_keys:
            descriptor += f", FK -> {foreign_keys[key]}"
        descriptor += ")"
        tables.setdefault(row.table_name, []).append(descriptor)

    blocks = [
        f"Table: {table_name}\n  " + ", ".join(column_descriptors)
        for table_name, column_descriptors in tables.items()
    ]
    return "\n\n".join(blocks)


if __name__ == "__main__":
    # Quick manual smoke test: python -m database.schema_introspection
    print(get_live_schema_description())
