"""Tests for tools/rrf.rrf_fuse — pure RRF math + edge cases."""

from __future__ import annotations

import pytest

from tamubot.rag.tools.rrf import rrf_fuse


def _doc(cid: str, **extra):
    return {"chunk_id": cid, **extra}


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def test_single_list_passthrough_preserves_order():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    out = rrf_fuse([docs], k=60, final_k=3)
    assert [d["chunk_id"] for d in out] == ["a", "b", "c"]


def test_score_matches_canonical_formula():
    # k=60, doc 'a' appears at rank 1 in both lists.
    # score(a) = 1/(60+1) + 1/(60+1) = 2/61
    out = rrf_fuse(
        [[_doc("a"), _doc("b")], [_doc("a"), _doc("c")]],
        k=60,
        final_k=3,
    )
    a_score = next(d["rrf_score"] for d in out if d["chunk_id"] == "a")
    assert a_score == pytest.approx(2 / 61)


def test_shared_doc_outranks_singletons():
    out = rrf_fuse(
        [[_doc("x"), _doc("y")], [_doc("x"), _doc("z")]],
        k=60,
        final_k=3,
    )
    assert out[0]["chunk_id"] == "x"
    # The remaining order is by single-list rank 1 score (= 1/61) — both equal,
    # so accept either order for y and z.
    rest = {d["chunk_id"] for d in out[1:]}
    assert rest == {"y", "z"}


def test_lower_rank_loses_to_higher_rank():
    # Doc 'a' is rank 2 in both lists; 'b' is rank 1 in list 1; 'c' rank 1 in list 2.
    # score(a) = 1/62 + 1/62 ≈ 0.0323
    # score(b) = 1/61               ≈ 0.0164
    # score(c) = 1/61               ≈ 0.0164
    # → 'a' wins.
    out = rrf_fuse(
        [[_doc("b"), _doc("a")], [_doc("c"), _doc("a")]],
        k=60,
        final_k=3,
    )
    assert out[0]["chunk_id"] == "a"


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_empty_input():
    assert rrf_fuse([], k=60, final_k=5) == []


def test_all_lists_empty():
    assert rrf_fuse([[], [], []], k=60, final_k=5) == []


def test_one_empty_list_ignored():
    out = rrf_fuse([[_doc("a"), _doc("b")], []], k=60, final_k=2)
    assert [d["chunk_id"] for d in out] == ["a", "b"]


def test_dedupe_by_chunk_id_first_representative_wins():
    out = rrf_fuse(
        [[_doc("a", source="L1")], [_doc("a", source="L2")]],
        k=60,
        final_k=1,
    )
    assert len(out) == 1
    assert out[0]["source"] == "L1"  # first-seen wins


def test_final_k_honored():
    docs = [_doc(f"c{i}") for i in range(10)]
    out = rrf_fuse([docs], k=60, final_k=3)
    assert len(out) == 3


def test_final_k_zero_returns_empty():
    out = rrf_fuse([[_doc("a"), _doc("b")]], k=60, final_k=0)
    assert out == []


def test_missing_chunk_id_skipped():
    out = rrf_fuse([[{"no_id": True}, _doc("a")]], k=60, final_k=5)
    assert [d["chunk_id"] for d in out] == ["a"]


def test_custom_id_key():
    out = rrf_fuse(
        [[{"_id": "x"}, {"_id": "y"}], [{"_id": "x"}]],
        k=60,
        final_k=2,
        id_key="_id",
    )
    assert out[0]["_id"] == "x"
