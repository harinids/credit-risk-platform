"""
Local SQLite loader for NL-to-SQL development/testing — no Docker/Postgres
needed. Same table structure the Postgres loader (scripts/load_data_to_db.py)
creates, so the NL-to-SQL logic built against this will need only a
connection-string change to work against Postgres later (both go through
SQLAlchemy).

Usage:
    python -m src.nl2sql.db_setup
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect, text

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Same mapping as the Postgres loader, but we skip the biggest tables'
# full row counts aren't needed for query-pattern development — we'll
# load full bureau/previous_application (needed for JOIN-based questions)
# but can sample the largest behavioral tables to keep local iteration fast.
FILE_TABLE_MAP = {
    "application_train.csv": ("application", None),
    "bureau.csv": ("bureau", None),
    "previous_application.csv": ("previous_application", None),
    "POS_CASH_balance.csv": ("pos_cash_balance", 500_000),
    "installments_payments.csv": ("installments_payments", 500_000),
    "credit_card_balance.csv": ("credit_card_balance", 500_000),
}


def build_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


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
        print(f"  SKIP: {filename} not found")
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
    print(f"Building local SQLite DB at {DB_PATH}")
    engine = build_engine()

    for filename, (table_name, sample_n) in FILE_TABLE_MAP.items():
        load_csv(engine, filename, table_name, sample_n)

    inspector = inspect(engine)
    print(f"\nTables in database: {inspector.get_table_names()}")

    with engine.connect() as conn:
        for table in inspector.get_table_names():
            count = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            print(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    main()