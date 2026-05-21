"""Tests for retrieval_node fanout — single vs. multi-subquery, partial failure."""

from __future__ import annotations

from unittest.mock import patch

from tamubot.rag.nodes.retrieval_node import retrieval_node

_CHUNK_INDEX_COUNTER = {"n": 0}


def _chunk(cid: str, **extra):
    # Each helper-built chunk gets a unique chunk_index so the new dedup +
    # per-course cap safety nets in retrieval_node treat them as distinct docs.
    # Tests that need a specific chunk_index can override via **extra.
    _CHUNK_INDEX_COUNTER["n"] += 1
    return {
        "_id": cid,
        "course_id": "ISEN 625",
        "chunk_index": _CHUNK_INDEX_COUNTER["n"],
        "content": cid,
        **extra,
    }


def _state(**kwargs):
    s = {
        "query": "orig",
        "rewritten_query": "rq",
        "function": "semantic_general",
        "course_ids": [],
        "subqueries": ["rq"],
        "intent_type": "GENERAL",
        "recursive_search": False,
        "retrieval_cache": {},
        "node_trace": [],
        "timing_ms": {},
    }
    s.update(kwargs)
    return s


# ---------------------------------------------------------------------------
# semantic_general
# ---------------------------------------------------------------------------


def test_semantic_general_single_subquery_unchanged():
    """Single-subquery state must use the legacy path (no RRF).

    Use distinct course_ids per chunk so the per-course cap (MAX_CHUNKS_PER_COURSE)
    leaves all three through — this test asserts the no-fusion path, not the cap.
    """
    with (
        patch(
            "tamubot.rag.tools.mongo.semantic_search",
            return_value=[
                _chunk("a", course_id="ISEN 600"),
                _chunk("b", course_id="ISEN 601"),
                _chunk("c", course_id="ISEN 602"),
            ],
        ),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(_state(subqueries=["only one"]))

    ids = [c["_id"] for c in out["retrieved_chunks"]]
    assert ids[:3] == ["a", "b", "c"]
    # rrf_score is only added on fusion
    assert "rrf_score" not in out["retrieved_chunks"][0]
    assert out["subqueries_chunk_counts"] == [3]


def test_semantic_general_three_subqueries_fans_out_and_fuses():
    calls = []

    def fake_semantic(query, k):
        calls.append(query)
        # Each subquery returns a different chunk list with one shared doc 'a'.
        return {
            "q1": [_chunk("a"), _chunk("b")],
            "q2": [_chunk("a"), _chunk("c")],
            "q3": [_chunk("a"), _chunk("d")],
        }[query]

    with (
        patch("tamubot.rag.tools.mongo.semantic_search", side_effect=fake_semantic),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(_state(subqueries=["q1", "q2", "q3"]))

    assert calls == ["q1", "q2", "q3"], "every subquery must run semantic_search"
    # Doc 'a' appears in all three lists — must be ranked #1 after RRF
    ids = [c["_id"] for c in out["retrieved_chunks"]]
    assert ids[0] == "a"
    # Fused output carries rrf_score
    assert "rrf_score" in out["retrieved_chunks"][0]
    assert len(out["subqueries_chunk_counts"]) == 3


def test_one_subquery_raises_others_succeed():
    def fake_semantic(query, k):
        if query == "boom":
            raise RuntimeError("downstream error")
        return [_chunk(f"{query}_x"), _chunk(f"{query}_y")]

    with (
        patch("tamubot.rag.tools.mongo.semantic_search", side_effect=fake_semantic),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(_state(subqueries=["alpha", "boom", "beta"]))

    assert out["retrieved_chunks"], "surviving variants must still produce results"
    assert "retrieval_partial_errors" in out
    assert any("boom" in e for e in out["retrieval_partial_errors"])
    # Failing variant counts as 0
    assert out["subqueries_chunk_counts"][1] == 0


def test_all_variants_raise_returns_error():
    def boom(query, k):
        raise RuntimeError("everything is on fire")

    with (
        patch("tamubot.rag.tools.mongo.semantic_search", side_effect=boom),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(_state(subqueries=["a", "b", "c"]))

    # error_guard_middleware wraps the propagated exception into the state
    assert out["retrieved_chunks"] == []
    assert "error" in out or out.get("retrieval_partial_errors")


# ---------------------------------------------------------------------------
# hybrid_course
# ---------------------------------------------------------------------------


def test_hybrid_course_single_subquery_single_course_unchanged():
    with (
        patch(
            "tamubot.rag.tools.mongo.hybrid_search",
            return_value=[_chunk("h1"), _chunk("h2")],
        ),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["ISEN 625"],
                subqueries=["only"],
            )
        )

    assert [c["_id"] for c in out["retrieved_chunks"]][:2] == ["h1", "h2"]
    assert "rrf_score" not in out["retrieved_chunks"][0]


def test_hybrid_course_multi_subquery_fans_out():
    seen = []

    def fake_hybrid(query, course_id, k):
        seen.append((query, course_id))
        return [_chunk(f"{query}_{course_id}_1"), _chunk(f"{query}_{course_id}_2")]

    with (
        patch("tamubot.rag.tools.mongo.hybrid_search", side_effect=fake_hybrid),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["ISEN 625"],
                subqueries=["s1", "s2"],
            )
        )

    # 2 subqueries × 1 course = 2 hybrid_search calls
    assert len(seen) == 2
    assert {q for q, _ in seen} == {"s1", "s2"}
    # Fused output present
    assert out["retrieved_chunks"]
    assert "rrf_score" in out["retrieved_chunks"][0]
