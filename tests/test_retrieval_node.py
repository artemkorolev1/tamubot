"""Tests for retrieval_node safety nets — dedup, per-course cap, summary fallback."""

from __future__ import annotations

from unittest.mock import patch

from tamubot.rag.nodes.retrieval_node import MAX_CHUNKS_PER_COURSE, retrieval_node


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
# Change 1 — dedup on single-subquery hybrid_course
# ---------------------------------------------------------------------------


def test_hybrid_course_dedups_single_subquery():
    """Duplicate (course_id, chunk_index) entries must collapse to one."""
    dup_chunks = [
        {"course_id": "ISEN 608", "chunk_index": 3, "content": "A"},
        {"course_id": "ISEN 608", "chunk_index": 3, "content": "A"},
        {"course_id": "ISEN 608", "chunk_index": 4, "content": "B"},
    ]

    with (
        patch(
            "tamubot.rag.tools.mongo.hybrid_search",
            return_value=dup_chunks,
        ),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["ISEN 608"],
                subqueries=["only"],
            )
        )

    chunks = out["retrieved_chunks"]
    assert len(chunks) == 2, f"expected 2 deduped chunks, got {len(chunks)}"
    assert {c["chunk_index"] for c in chunks} == {3, 4}


# ---------------------------------------------------------------------------
# Change 2 — per-course cap on semantic_general
# ---------------------------------------------------------------------------


def test_semantic_general_caps_per_course():
    """Per-course cap keeps at most MAX_CHUNKS_PER_COURSE per course_id, rank-order preserved.

    Reads the constant rather than hardcoding it — the cap is a tuning knob.
    A course with fewer chunks than the cap is unaffected; one over the cap is trimmed.
    """
    cap = MAX_CHUNKS_PER_COURSE
    n_over_cap = cap + 6  # well above cap, should be trimmed
    n_under_cap = max(1, cap - 1)  # below cap, must pass through untouched
    isen645 = [{"course_id": "ISEN 645", "chunk_index": i, "content": f"645-{i}"} for i in range(n_over_cap)]
    isen625 = [{"course_id": "ISEN 625", "chunk_index": i, "content": f"625-{i}"} for i in range(n_under_cap)]
    fake_results = isen645 + isen625

    with (
        patch(
            "tamubot.rag.tools.mongo.semantic_search",
            return_value=fake_results,
        ),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(_state(function="semantic_general", subqueries=["only"]))

    chunks = out["retrieved_chunks"]
    by_course: dict[str, list[dict]] = {}
    for c in chunks:
        by_course.setdefault(c["course_id"], []).append(c)

    assert len(by_course.get("ISEN 645", [])) == cap, f"ISEN 645 must be capped to {cap}"
    assert len(by_course.get("ISEN 625", [])) == n_under_cap, "ISEN 625 under cap must pass through"

    # Order preserved: all ISEN 645 chunks come before ISEN 625 chunks.
    course_order = [c["course_id"] for c in chunks]
    first_625 = course_order.index("ISEN 625")
    last_645 = max(i for i, cid in enumerate(course_order) if cid == "ISEN 645")
    assert last_645 < first_625, "645 chunks must precede 625 chunks"

    # The first `cap` chunks of ISEN 645 must be the ones kept; ISEN 625 untouched.
    assert [c["chunk_index"] for c in by_course["ISEN 645"]] == list(range(cap))
    assert [c["chunk_index"] for c in by_course["ISEN 625"]] == list(range(n_under_cap))


# ---------------------------------------------------------------------------
# Change 3 — course_summary widened fallback
# ---------------------------------------------------------------------------


def test_course_summary_falls_back_when_thin():
    """One thin summary chunk (token_count=50) must trigger hybrid supplementation."""
    thin_summary = [
        {
            "course_id": "ISEN 608",
            "chunk_index": 0,
            "content": "thin summary",
            "token_count": 50,
            "is_summary": True,
        }
    ]
    hybrid_hits = [
        {"course_id": "ISEN 608", "chunk_index": 11, "content": "H1"},
        {"course_id": "ISEN 608", "chunk_index": 12, "content": "H2"},
        {"course_id": "ISEN 608", "chunk_index": 13, "content": "H3"},
    ]

    with (
        patch(
            "tamubot.rag.tools.mongo.get_course_summary_chunks",
            return_value=thin_summary,
        ),
        patch(
            "tamubot.rag.tools.mongo.hybrid_search",
            return_value=hybrid_hits,
        ),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, chunks, top_k, **kw: chunks[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="course_summary",
                course_ids=["ISEN 608"],
                subqueries=["only"],
            )
        )

    chunks = out["retrieved_chunks"]
    contents = [c.get("content") for c in chunks]
    # Summary first, then hybrid hits
    assert contents[0] == "thin summary", "summary chunk must come first"
    assert contents[1:] == ["H1", "H2", "H3"], "hybrid hits must follow, in order"
    assert len(chunks) == 4


# ---------------------------------------------------------------------------
# Change 4 — SUMMARY_AS_PRIMER: course-overview primer on hybrid_course
# ---------------------------------------------------------------------------


def _summary_chunks(*course_ids: str) -> list[dict]:
    return [
        {
            "course_id": cid,
            "content": f"Overview text for {cid}.",
            "header_path": "Course Overview",
            "section": "",
            "source": "course_summary",
            "source_file": f"{cid}.pdf",
            "chunk_index": 0,
            "pipeline_version": "v4",
        }
        for cid in course_ids
    ]


def test_hybrid_course_primer_flag_off_no_primer():
    """With SUMMARY_AS_PRIMER=False, no context_primer key is emitted."""
    chunks = [{"course_id": "ISEN 625", "chunk_index": 1, "content": "A"}]

    with (
        patch("tamubot.core.config.SUMMARY_AS_PRIMER", False),
        patch("tamubot.rag.tools.mongo.hybrid_search", return_value=chunks),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, ch, top_k, **kw: ch[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["ISEN 625"],
                subqueries=["only"],
            )
        )

    assert "context_primer" not in out
    assert out["retrieved_chunks"] == chunks


def test_hybrid_course_primer_flag_on_returns_primer():
    """Flag on + course_ids → context_primer carries [Course <id>] sub-header per course."""
    hybrid_hits = [{"course_id": "ISEN 625", "chunk_index": 1, "content": "A"}]

    with (
        patch("tamubot.core.config.SUMMARY_AS_PRIMER", True),
        patch(
            "tamubot.rag.tools.mongo.get_course_summary_chunks",
            return_value=_summary_chunks("ISEN 625"),
        ),
        patch("tamubot.rag.tools.mongo.hybrid_search", return_value=hybrid_hits),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, ch, top_k, **kw: ch[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["ISEN 625"],
                subqueries=["only"],
            )
        )

    assert out["retrieved_chunks"] == hybrid_hits, "retrieved_chunks untouched by primer"
    primer = out.get("context_primer")
    assert primer is not None
    assert "[Course ISEN 625]" in primer
    assert "Overview text for ISEN 625." in primer


def test_hybrid_course_primer_multi_course_concatenated():
    """Multi-course primer concatenates per-course content with [Course <id>] sub-headers."""
    with (
        patch("tamubot.core.config.SUMMARY_AS_PRIMER", True),
        patch(
            "tamubot.rag.tools.mongo.get_course_summary_chunks",
            return_value=_summary_chunks("CSCE 638", "CSCE 605"),
        ),
        patch("tamubot.rag.tools.mongo.hybrid_search", return_value=[]),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, ch, top_k, **kw: ch[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["CSCE 638", "CSCE 605"],
                subqueries=["only"],
            )
        )

    primer = out["context_primer"]
    assert "[Course CSCE 638]" in primer
    assert "[Course CSCE 605]" in primer


def test_hybrid_course_primer_missing_summaries_no_primer():
    """Flag on + no summary docs in Mongo → primer is absent (silently skipped)."""
    with (
        patch("tamubot.core.config.SUMMARY_AS_PRIMER", True),
        patch("tamubot.rag.tools.mongo.get_course_summary_chunks", return_value=[]),
        patch("tamubot.rag.tools.mongo.hybrid_search", return_value=[]),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, ch, top_k, **kw: ch[:top_k],
        ),
    ):
        out = retrieval_node(
            _state(
                function="hybrid_course",
                course_ids=["UNKNOWN 999"],
                subqueries=["only"],
            )
        )
    assert "context_primer" not in out


def test_hybrid_course_primer_not_emitted_for_semantic_general():
    """Primer is hybrid_course-only; semantic_general must not emit one."""
    with (
        patch("tamubot.core.config.SUMMARY_AS_PRIMER", True),
        patch("tamubot.rag.tools.mongo.semantic_search", return_value=[]),
        patch(
            "tamubot.rag.tools.voyage.rerank",
            side_effect=lambda q, ch, top_k, **kw: ch[:top_k],
        ),
    ):
        out = retrieval_node(_state(function="semantic_general", subqueries=["only"]))
    assert "context_primer" not in out
