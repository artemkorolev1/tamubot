"""Shared psycopg3 connection pool for the Postgres retrieval backend.

Lazy singleton, mirroring the ``_get_db()`` pattern in ``rag/tools/mongo.py``.
Registers the pgvector type adapters once per new connection so ``list[float]``
round-trips to ``vector`` transparently.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from tamubot.core import config

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool


def _configure(conn) -> None:
    """Per-connection setup: register pgvector adapters."""
    from pgvector.psycopg import register_vector

    register_vector(conn)


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """Return the process-wide connection pool (lazy singleton)."""
    from psycopg_pool import ConnectionPool

    if not config.POSTGRES_URI:
        raise RuntimeError("POSTGRES_URI is not set — cannot open a Postgres pool.")
    pool = ConnectionPool(
        conninfo=config.POSTGRES_URI,
        min_size=1,
        max_size=config.POSTGRES_POOL_MAX,
        configure=_configure,
        open=True,
    )
    return pool
