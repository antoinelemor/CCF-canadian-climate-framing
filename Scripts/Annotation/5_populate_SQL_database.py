"""
PROJECT:
--------
CCF-Canadian-Climate-Framing

TITLE:
------
5_populate_SQL_database.py

MAIN OBJECTIVE:
---------------
Create (if absent) the local PostgreSQL database **CCF** and populate it
with two tables drawn from the project’s CSV files:
1. CCF_full_data      – every raw article in the corpus  
2. CCF_processed_data – the tokenised and annotated sentences

Dependencies:
-------------
- argparse
- getpass
- io
- sys
- pathlib
- typing
- pandas
- psycopg2
- tqdm.auto

MAIN FEATURES:
--------------
1) DB creation – safely creates the CCF database only if it
   does not already exist (runs in the `postgres` maintenance database with
   autocommit enabled).
2) Flexible login logic – hard-coded credentials can be overridden by
   `--force-password-prompt`, allowing secure deployments on shared hosts.
3) Connection helper – returns a tuned psycopg2 connection
   (`client_min_messages=warning`, explicit transaction control).
4) Automatic table synthesis – infers column names and SQL types from
   a pandas DataFrame and issues `CREATE TABLE` statements on-the-fly.
5) High-throughput bulk loads – streams data to PostgreSQL via the
   `COPY` command from an in-memory CSV buffer (≈50× faster than INSERT).
6) Column filtering – keeps only the relevant fields for
   CCF_processed_data to minimise storage and query overhead.
7) Robust error handling – explicit commits/rollbacks ensure the
   database is never left in an inconsistent state after a failed `COPY`.
8) Progress feedback – clear, colour-aware messages (and ready-to-use
   tqdm progress bars for future chunked loads).

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import argparse
import getpass
import io
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import psycopg2
import psycopg2.extras as extras
from psycopg2 import sql, extensions
from tqdm.auto import tqdm

# ############# CONFIGURATION ############# #

DB_PARAMS: Dict[str, Any] = {
    "host": "localhost",
    "port": 5432,
    "user": "",
    "password": "",  
}
DB_NAME: str = "CCF"
FULL_CSV: Path = Path("Database/Database/CCF.media_database.csv")
PROC_CSV: Path = Path("Database/Database/CCF.media_processed_texts_annotated.csv")
PROC_KEEP_COLS: List[str] = [
    "doc_id",
    "news_type",
    "title",
    "author",
    "media",
    "words_count",
    "date",
    "language",
    "page_number",
    "sentences",
    "sentence_id",
]
CHUNK_SIZE: int = 1_000  # number of rows per COPY iteration


# ############# HELPER FUNCTIONS ############ #

def create_database_if_absent(params: Dict[str, Any], db_name: str) -> None:
    """Create *db_name* using a connection to the *postgres* maintenance DB.

    The function is *idempotent* – if the database already exists nothing
    happens. We rely on an **autocommit** connection because `CREATE DATABASE`
    cannot run inside a transaction block.
    """
    tmp_params = params.copy()
    tmp_params["dbname"] = "postgres"  # connect to maintenance DB

    # Explicit connection (no context manager) so we can tweak isolation level
    conn = psycopg2.connect(**tmp_params)
    # Force autocommit at the driver level (safer than `conn.autocommit = True`)
    conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,)
            )
            if cur.fetchone():
                print(f"Database '{db_name}' already exists – re‑using it.")
                return

            print(f"Creating database '{db_name}' …")
            cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(db_name)))
            print("✔ Done.")
    finally:
        conn.close()


def get_connection(params: Dict[str, Any], db_name: str) -> psycopg2.extensions.connection:
    """Return a *psycopg2* connection to *db_name* using *params*."""
    conn_params = params.copy()
    conn_params["dbname"] = db_name
    conn_params["options"] = "-c client_min_messages=warning"
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = False  # explicit commits for data loading
    return conn


def create_table_like_dataframe(
    conn: psycopg2.extensions.connection, table_name: str, df: pd.DataFrame
) -> None:
    """Create *table_name* in *conn* with columns/types inferred from *df*.

    If the table already exists we assume it has the correct structure and skip
    creation.
    """
    type_map = {
        "object": "TEXT",
        "int64": "BIGINT",
        "float64": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
    }
    col_defs = [
        f"{sql.Identifier(col).as_string(conn)} {type_map.get(str(dtype), 'TEXT')}"
        for col, dtype in df.dtypes.items()
    ]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
              FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name = %s;
            """,
            (table_name,),
        )
        if cur.fetchone():
            print(f"Table '{table_name}' already exists – skipping creation.")
            return

        print(f"Creating table '{table_name}' …")
        create_stmt = sql.SQL("CREATE TABLE {} ({});").format(
            sql.Identifier(table_name), sql.SQL(", ").join(sql.SQL(defn) for defn in col_defs)
        )
        cur.execute(create_stmt)
        conn.commit()
        print("✔ Done.")


def copy_dataframe(
    conn: psycopg2.extensions.connection, table_name: str, df: pd.DataFrame
) -> None:
    """Bulk‑insert *df* into *table_name* using PostgreSQL `COPY`."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    with conn.cursor() as cur:
        print(f"Inserting {len(df):,} rows into '{table_name}' …")
        try:
            cur.copy_expert(
                sql.SQL("COPY {} FROM STDIN WITH (FORMAT CSV);").format(sql.Identifier(table_name)),
                buffer,
            )
            conn.commit()
            print("✔ Done.")
        except Exception as exc:
            conn.rollback()
            print(f"✘ COPY failed → {exc}")
            raise


# ############## MAIN LOGIC ############## #

def main() -> None:
    """CLI entry‑point."""

    parser = argparse.ArgumentParser(
        description="Populate the CCF PostgreSQL database with the raw and processed media tables.",
    )
    parser.add_argument(
        "--force-password-prompt",
        action="store_true",
        help="Prompt for the DB password instead of using the hard‑coded one.",
    )
    args = parser.parse_args()

    if args.force_password_prompt:
        DB_PARAMS["password"] = getpass.getpass("PostgreSQL password: ")

    # 1 ─ Ensure the database exists (no‑op if already present)
    create_database_if_absent(DB_PARAMS, DB_NAME)

    # 2 ─ Open connection to **CCF** and populate tables
    with get_connection(DB_PARAMS, DB_NAME) as conn:
        # 2·1 ─ Raw articles
        full_df = pd.read_csv(FULL_CSV)
        create_table_like_dataframe(conn, "CCF_full_data", full_df)
        copy_dataframe(conn, "CCF_full_data", full_df)

        # 2·2 ─ Processed/annotated sentences (column subset)
        proc_df = pd.read_csv(PROC_CSV, usecols=lambda c: c in PROC_KEEP_COLS)
        create_table_like_dataframe(conn, "CCF_processed_data", proc_df)
        copy_dataframe(conn, "CCF_processed_data", proc_df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user – exiting…")
