"""Tests for index_health helpers — use mocks, no real Atlas connection."""

from unittest.mock import MagicMock

from tamubot.ingestion.validation.index_health import (
    check_atlas_index_ready,
    check_vector_count_matches_chunks,
)


def test_atlas_index_ready_pass():
    collection = MagicMock()
    collection.list_search_indexes.return_value = [
        {"name": "default", "status": "READY", "queryable": True},
    ]
    out = check_atlas_index_ready(collection, index_name="default")
    assert out.passed
    assert out.metadata["status"] == "READY"


def test_atlas_index_ready_fail_not_queryable():
    collection = MagicMock()
    collection.list_search_indexes.return_value = [
        {"name": "default", "status": "BUILDING", "queryable": False},
    ]
    out = check_atlas_index_ready(collection, index_name="default")
    assert out.passed is False
    assert out.metadata["status"] == "BUILDING"


def test_atlas_index_ready_fail_index_missing():
    collection = MagicMock()
    collection.list_search_indexes.return_value = []
    out = check_atlas_index_ready(collection, index_name="default")
    assert out.passed is False
    assert "missing" in out.metadata["reason"]


def test_vector_count_matches_chunks_pass():
    collection = MagicMock()
    collection.count_documents.return_value = 42
    out = check_vector_count_matches_chunks(
        collection,
        filter_query={"source_file": "stem", "chunk_tag": "v6b_semantic"},
        expected_count=42,
    )
    assert out.passed


def test_vector_count_matches_chunks_fail():
    collection = MagicMock()
    collection.count_documents.return_value = 40
    out = check_vector_count_matches_chunks(
        collection,
        filter_query={"source_file": "stem", "chunk_tag": "v6b_semantic"},
        expected_count=42,
    )
    assert out.passed is False
    assert out.metadata["delta"] == -2
