from pathlib import Path
import re
import sqlite3
import pandas as pd


DB_PATH = Path("data/raw_data/database.sqlite")
# Data - https://www.kaggle.com/datasets/hugomathien/soccer
OUTPUT_DIR = Path("data/raw_data/european_league_matches")
CSV_DIR = OUTPUT_DIR / "csv"
PARQUET_DIR = OUTPUT_DIR / "parquet"


def safe_filename(name: str) -> str:
    """
    Convert SQLite table names into safe file names.
    Example: 'Player Attributes' -> 'player_attributes'
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def quote_identifier(identifier: str) -> str:
    """
    Safely quote SQLite table/column names.
    """
    return '"' + identifier.replace('"', '""') + '"'


def get_tables(conn: sqlite3.Connection) -> list[str]:
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
    """
    tables = pd.read_sql_query(query, conn)["name"].tolist()
    return tables


def get_table_info(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    query = f"PRAGMA table_info({quote_identifier(table_name)});"
    info = pd.read_sql_query(query, conn)
    info.insert(0, "table_name", table_name)
    return info


def get_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    query = f"SELECT COUNT(*) AS row_count FROM {quote_identifier(table_name)};"
    return int(pd.read_sql_query(query, conn)["row_count"].iloc[0])


def export_table(conn: sqlite3.Connection, table_name: str) -> dict:
    safe_name = safe_filename(table_name)

    print(f"\nExporting table: {table_name}")

    query = f"SELECT * FROM {quote_identifier(table_name)};"
    df = pd.read_sql_query(query, conn)

    csv_path = CSV_DIR / f"{safe_name}.csv"
    parquet_path = PARQUET_DIR / f"{safe_name}.parquet"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {csv_path}")

    parquet_saved = True
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet: {parquet_path}")
    except Exception as e:
        parquet_saved = False
        print(f"Could not save Parquet for {table_name}.")
        print("Install pyarrow if needed: pip install pyarrow")
        print(f"Error: {e}")

    return {
        "table_name": table_name,
        "file_name": safe_name,
        "rows": len(df),
        "columns": len(df.columns),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path) if parquet_saved else None,
    }


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Could not find SQLite file at: {DB_PATH.resolve()}"
        )

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading SQLite database from: {DB_PATH.resolve()}")

    conn = sqlite3.connect(DB_PATH)

    try:
        tables = get_tables(conn)

        if not tables:
            print("No tables found in the SQLite database.")
            return

        print("\nTables found:")
        for table in tables:
            print(f"- {table}")

        export_summary = []
        schema_summary = []

        for table in tables:
            row_count = get_row_count(conn, table)
            print(f"{table}: {row_count:,} rows")

            table_info = get_table_info(conn, table)
            schema_summary.append(table_info)

            export_result = export_table(conn, table)
            export_summary.append(export_result)

        export_summary_df = pd.DataFrame(export_summary)
        schema_summary_df = pd.concat(schema_summary, ignore_index=True)

        summary_path = OUTPUT_DIR / "sqlite_export_summary.csv"
        schema_path = OUTPUT_DIR / "sqlite_schema_summary.csv"

        export_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        schema_summary_df.to_csv(schema_path, index=False, encoding="utf-8-sig")

        print("\nDone.")
        print(f"Export summary saved to: {summary_path}")
        print(f"Schema summary saved to: {schema_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()