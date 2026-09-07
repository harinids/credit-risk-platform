"""
Postgres loader for Docker deployment - runs once as the "db-init" service,
loading raw CSVs into the Postgres database before the API starts.

Same table structure and sampling strategy as src/nl2sql/db_setup.py (the
local SQLite loader used for dev/testing) - both go through SQLAlchemy, so
switching between them is just a connection-string change, as designed.

Reads connection details from environment variables (POSTGRES_USER,
POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT), matching
src/api/config.py's Settings class, via env_file: .env in docker-compose.

Usage (inside container):
    python load_data_to_db.py
"""

import os
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

DATA_DIR = Path("/app/data/raw")

FILE_TABLE_MAP = {
    "application_train.csv": ("application", None),
    "bureau.csv": ("bureau", None),
    "previous_application.csv": ("previous_application", None),
    "POS_CASH_balance.csv": ("pos_cash_balance", 500_000),
    "installments_payments.csv": ("installments_payments", 500_000),
    "credit_card_balance.csv": ("credit_card_balance", 500_000),
}


def build_engine():
    user = os.getenv("POSTGRES_USER", "credit_admin")
    password = os.getenv("POSTGRES_PASSWORD", "change_me")
    db = os.getenv("POSTGRES_DB", "credit_risk")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def wait_for_db(engine, max_retries: int = 10, delay_seconds: int = 3):
    """Postgres healthcheck in compose should already ensure this, but
    a short retry loop here is cheap insurance against a race condition."""
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Database connection established.")
            return
        except OperationalError as exc:
            print(f"  DB not ready (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(delay_seconds)
    raise RuntimeError("Could not connect to database after retries.")


def table_has_rows(engine, table_name: str) -> bool:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return False
    with engine.connect() as conn:
        count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
    return count > 0


def load_csv(engine, filename: str, table_name: str, sample_n):
    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        print(f"  SKIP: {filename} not found at {csv_path}")
        return
    if table_has_rows(engine, table_name):
        print(f"  SKIP: '{table_name}' already loaded")
        return
    print(f"  Loading {filename} -> '{table_name}'" + (f" (sampled to {sample_n:,} rows)" if sample_n else ""))
    df = pd.read_csv(csv_path, nrows=sample_n, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"    -> {len(df):,} rows loaded")


def main():
    print("Starting Postgres data load...")
    engine = build_engine()
    wait_for_db(engine)

    for filename, (table_name, sample_n) in FILE_TABLE_MAP.items():
        load_csv(engine, filename, table_name, sample_n)

    inspector = inspect(engine)
    print(f"\nTables in database: {inspector.get_table_names()}")
    with engine.connect() as conn:
        for table in inspector.get_table_names():
            count = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            print(f"  {table}: {count:,} rows")

    print("\nData load complete.")


if __name__ == "__main__":
    main()
