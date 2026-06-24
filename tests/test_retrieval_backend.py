"""Tests for the retrieval backend selector + the Postgres query primitives.

The dispatch tests are pure (no DB). The ``queries`` tests are integration
tests against a live, backfilled Postgres — they skip automatically when the
pool can't be opened (e.g. POSTGRES_URI unset on the host).
"""

import importlib

import pytest

from tamubot.core import config
from tamubot.rag.tools import backend


# ── Dispatch (pure) ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "flag,expected_mod",
    [
        ("postgres", "tamubot.rag.tools.queries"),
        ("mongodb", "tamubot.rag.tools.mongo"),
        ("vertex", "tamubot.rag.tools.mongo"),  # legacy non-postgres → mongo
    ],
)
def test_impl_dispatches_on_flag(monkeypatch, flag, expected_mod):
    monkeypatch.setattr(config, "RETRIEVAL_BACKEND", flag)
    assert backend._impl().__name__ == expected_mod


def test_dispatch_is_evaluated_per_call(monkeypatch):
    # Flipping the flag between calls must change the routed module — proves the
    # selector reads config at call time, not import time.
    monkeypatch.setattr(config, "RETRIEVAL_BACKEND", "mongodb")
    assert backend._impl().__name__.endswith("mongo")
    monkeypatch.setattr(config, "RETRIEVAL_BACKEND", "postgres")
    assert backend._impl().__name__.endswith("queries")


# ── Postgres primitives (integration; skip if no live PG) ────────────────────


@pytest.fixture(scope="module")
def pg_ready():
    from tamubot.ingestion.postgres.pool import get_pool

    try:
        pool = get_pool()
        with pool.connection() as conn:
            n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable / not backfilled: {exc}")
    if not n:
        pytest.skip("chunks table empty — run the backfill first")
    return n


def test_fetch_anchor_chunks_shape(pg_ready):
    from tamubot.rag.tools import queries

    # Pick a course_id that exists.
    from tamubot.ingestion.postgres.pool import get_pool

    with get_pool().connection() as conn:
        course_id = conn.execute(
            "SELECT course_id FROM chunks WHERE course_id IS NOT NULL LIMIT 1"
        ).fetchone()[0]

    chunks, gaps, ok = queries.fetch_anchor_chunks([course_id])
    assert ok is True and gaps == []
    assert chunks, "expected at least one anchor chunk"
    c = chunks[0]
    # Mongo-compatible projection keys (no leaked _id).
    for key in ("course_id", "chunk_index", "content", "header_path", "source", "source_file"):
        assert key in c
    assert "_id" not in c


def test_filter_clauses_exclude_boilerplate_and_dupes(monkeypatch):
    # Pure check of the WHERE-builder (no DB): default config excludes both.
    monkeypatch.setattr(config, "INCLUDE_BOILERPLATE", False)
    monkeypatch.setattr(config, "INCLUDE_DUPLICATE", False)
    monkeypatch.delenv("CHUNK_TAG_FILTER", raising=False)
    from tamubot.rag.tools.queries import _filter_clauses

    conds, params = _filter_clauses("ISEN 625", None)
    assert "is_boilerplate IS NOT TRUE" in conds
    assert "is_duplicate IS NOT TRUE" in conds
    assert params["course_id"] == "ISEN 625"


def test_filter_clauses_default_tag_is_semantic(monkeypatch):
    # Unset CHUNK_TAG_FILTER defaults to the production 'semantic' tag (no double-count).
    monkeypatch.delenv("CHUNK_TAG_FILTER", raising=False)
    from tamubot.rag.tools.queries import _filter_clauses

    conds, params = _filter_clauses(None, None)
    assert "chunk_tag = %(chunk_tag)s" in conds
    assert params["chunk_tag"] == "semantic"


def test_filter_clauses_empty_tag_searches_all(monkeypatch):
    # CHUNK_TAG_FILTER="" is the escape hatch to search every tag variant.
    monkeypatch.setattr(config, "INCLUDE_BOILERPLATE", True)
    monkeypatch.setattr(config, "INCLUDE_DUPLICATE", True)
    monkeypatch.setenv("CHUNK_TAG_FILTER", "")
    from tamubot.rag.tools.queries import _filter_clauses

    conds, _ = _filter_clauses(None, None)
    assert conds == []
