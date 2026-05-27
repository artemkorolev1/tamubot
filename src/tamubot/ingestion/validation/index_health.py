"""Atlas Vector Search health helpers. Accept a pymongo Collection so tests
can pass a MagicMock."""

from __future__ import annotations

from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome


def check_atlas_index_ready(collection: Any, index_name: str) -> CheckOutcome:
    """Pass iff the named search index is READY and queryable."""
    indexes = list(collection.list_search_indexes())
    match = next((i for i in indexes if i.get("name") == index_name), None)
    if match is None:
        return CheckOutcome(
            passed=False,
            metadata={
                "reason": f"index {index_name!r} missing",
                "found_indexes": [i.get("name") for i in indexes],
            },
        )
    status = match.get("status", "UNKNOWN")
    queryable = bool(match.get("queryable", False))
    return CheckOutcome(
        passed=status == "READY" and queryable,
        metadata={"status": status, "queryable": queryable, "index_name": index_name},
    )


def check_vector_count_matches_chunks(
    collection: Any,
    filter_query: dict,
    expected_count: int,
) -> CheckOutcome:
    """Pass iff count_documents(filter_query) == expected_count."""
    actual = collection.count_documents(filter_query)
    delta = actual - expected_count
    return CheckOutcome(
        passed=actual == expected_count,
        metadata={
            "actual_count": actual,
            "expected_count": expected_count,
            "delta": delta,
            "filter": filter_query,
        },
    )
