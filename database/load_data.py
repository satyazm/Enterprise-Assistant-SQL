"""
database/load_data.py

Loads customers.csv, products.csv, sample_sales.csv, and complaints.csv
into Postgres, in FK-safe order (customers/products first, then
sales/complaints which reference them).

Run once after creating the schema:
    python -m database.load_data
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Uses the ADMIN connection (needs INSERT privileges), not the read-only one.
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "enterprise_rag")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpass")

ADMIN_DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Path to your uploaded CSVs — adjust if you move them into the repo.
CSV_DIR = os.getenv("CSV_DIR", "data/csv")

TABLES_IN_ORDER = [
    ("customers", "customers.csv", ["signup_date"]),
    ("products", "products.csv", []),
    ("sales", "sample_sales.csv", ["sale_date"]),
    ("complaints", "complaints.csv", ["complaint_date"]),
]


def load_all():
    engine = create_engine(ADMIN_DB_URL)

    for table_name, csv_filename, date_cols in TABLES_IN_ORDER:
        csv_path = os.path.join(CSV_DIR, csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Expected CSV not found: {csv_path}. "
                f"Set CSV_DIR in .env or move your CSVs into {CSV_DIR}/"
            )

        df = pd.read_csv(csv_path, parse_dates=date_cols)
        df.to_sql(table_name, engine, if_exists="append", index=False, method="multi")
        print(f"Loaded {len(df)} rows into '{table_name}' from {csv_filename}")

    print("All tables loaded successfully.")


if __name__ == "__main__":
    load_all()
