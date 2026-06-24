"""Apply the TamuBot Postgres schema (idempotent).

Mirrors ``ingestion/setup_atlas.py`` for the pgvector backend. Reads the adjacent
``schema.sql`` and executes it against ``config.POSTGRES_URI``. The DDL uses
``IF NOT EXISTS`` throughout, so re-running is a no-op.

Usage:
    python -m tamubot.ingestion.postgres.setup_postgres          # apply schema
    python -m tamubot.ingestion.postgres.setup_postgres --verify # apply + list tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tamubot.core import config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _require_uri() -> str:
    if not config.POSTGRES_URI:
        print("ERROR: POSTGRES_URI not set in environment (.env).", file=sys.stderr)
        sys.exit(1)
    return config.POSTGRES_URI


def apply_schema(uri: str | None = None) -> None:
    """Execute schema.sql against the target database. Idempotent."""
    import psycopg

    uri = uri or _require_uri()
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    # autocommit so each DDL statement commits independently; psycopg3 executes
    # the multi-statement (parameter-less) script in one round-trip.
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(sql)
    print(f"Applied schema from {SCHEMA_PATH.name}")


def list_tables(uri: str | None = None) -> list[str]:
    """Return the public-schema table names (verification helper)."""
    import psycopg

    uri = uri or _require_uri()
    with psycopg.connect(uri) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the TamuBot Postgres schema (idempotent).")
    parser.add_argument("--verify", action="store_true", help="After applying, list created tables.")
    args = parser.parse_args()

    uri = _require_uri()
    apply_schema(uri)

    if args.verify:
        tables = list_tables(uri)
        print(f"\n{len(tables)} table(s) in public schema:")
        for t in tables:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
