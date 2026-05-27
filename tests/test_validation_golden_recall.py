"""Tests for golden_recall — uses mocks; no real Atlas / Voyage calls."""

from tamubot.ingestion.validation.golden_recall import compute_recall_at_k


def test_full_recall_pass():
    queries = [
        {"question": "q1", "expected_chunk_ids": ["A", "B"]},
        {"question": "q2", "expected_chunk_ids": ["C"]},
    ]

    def fake_retrieve(question, k):
        if question == "q1":
            return ["A", "B", "X", "Y", "Z"]
        return ["C", "X", "Y", "Z", "W"]

    out = compute_recall_at_k(queries, retrieve_fn=fake_retrieve, k=5)
    assert out.passed
    assert out.metadata["recall_at_k"] == 1.0


def test_partial_recall_warn():
    queries = [
        {"question": "q1", "expected_chunk_ids": ["A", "B"]},
    ]

    def fake_retrieve(question, k):
        return ["A", "X", "Y", "Z", "W"]

    out = compute_recall_at_k(queries, retrieve_fn=fake_retrieve, k=5, min_recall=0.8)
    assert out.passed is False
    assert out.metadata["recall_at_k"] == 0.5
