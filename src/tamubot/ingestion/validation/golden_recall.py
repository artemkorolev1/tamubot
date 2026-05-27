"""Golden-set recall@K helper. Pure-math + retrieve_fn parameter so tests can
substitute a fake retriever (no real Atlas / Voyage)."""

from __future__ import annotations

from typing import Any, Callable

from tamubot.ingestion.validation.types import CheckOutcome


def compute_recall_at_k(
    queries: list[dict[str, Any]],
    retrieve_fn: Callable[[str, int], list[str]],
    k: int = 5,
    min_recall: float = 0.8,
) -> CheckOutcome:
    """Compute mean recall@k across queries.

    queries: list of {"question": str, "expected_chunk_ids": list[str]}.
    retrieve_fn(question, k) -> list of chunk ids (most-relevant first).
    """
    if not queries:
        return CheckOutcome(passed=True, metadata={"recall_at_k": None, "n_queries": 0})
    recalls: list[float] = []
    for q in queries:
        expected = set(q.get("expected_chunk_ids", []))
        if not expected:
            continue
        retrieved = set(retrieve_fn(q["question"], k))
        recalls.append(len(expected & retrieved) / len(expected))
    if not recalls:
        return CheckOutcome(passed=True, metadata={"recall_at_k": None, "n_queries": 0})
    mean_recall = sum(recalls) / len(recalls)
    return CheckOutcome(
        passed=mean_recall >= min_recall,
        metadata={
            "recall_at_k": round(mean_recall, 4),
            "k": k,
            "min_recall": min_recall,
            "n_queries": len(recalls),
        },
    )
