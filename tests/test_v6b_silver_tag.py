"""Tests for the v6b silver_tag three-pass tagger.

Boilerplate + within-syllabus dedup + cross-syllabus dedup, with the
chunk_count_preserved invariant exercised on every test.
"""

from __future__ import annotations

import pandas as pd

from tamubot.ingestion.pipeline_v6b.util.signature_index import (
    build_lsh_from_df,
    chunk_id_of,
)
from tamubot.ingestion.pipeline_v6b.util.tagging import (
    build_local_signatures,
    flag_boilerplate,
    flag_cross_syllabus_dups,
    flag_within_syllabus_dups,
    tag_chunks,
)
from tamubot.ingestion.pipeline_v6b.util.text_normalize import ReferenceIndex

_AGGIE_HONOR = (
    "An Aggie does not lie cheat or steal or tolerate those who do this principle "
    "is the foundation of the Aggie Honor Code and applies to all academic work "
    "produced in this course including homework exams quizzes and final projects."
)

_PROGRAMMING_POLICY = (
    "All programming assignments must be submitted via the course management system "
    "by eleven fifty nine pm on the due date late submissions will incur a ten percent "
    "penalty per day up to a maximum of three days after which no credit will be awarded."
)


def _chunk(idx: int, content: str) -> dict:
    return {
        "chunk_index": idx,
        "content": content,
        "is_boilerplate": False,
        "boilerplate_cluster": None,
        "cluster_confidence": None,
        "is_duplicate": False,
        "duplicate_of_chunk_id": None,
    }


def _make_reference_index(tmp_path, entries: list[str]) -> ReferenceIndex:
    """Build a tiny boilerplate reference parquet and load it."""
    rows = []
    for i, text in enumerate(entries):
        rows.append(
            {
                "cluster_id": f"bp_test_{i:04d}",
                "representative_text": text,
                "normalized_text": text.lower(),
                "doc_frequency": 10,
                "distinct_depts": 5,
                "ngram_signature": [],
            }
        )
    df = pd.DataFrame(rows)
    p = tmp_path / "ref.parquet"
    df.to_parquet(p, index=False)
    return ReferenceIndex(p)


# ---------- Boilerplate pass ----------

def test_boilerplate_pass_flags_exact_match(tmp_path):
    ref = _make_reference_index(tmp_path, [_AGGIE_HONOR])
    chunks = [_chunk(0, "Welcome to the course."), _chunk(1, _AGGIE_HONOR)]
    flag_boilerplate(chunks, ref)
    assert chunks[0]["is_boilerplate"] is False
    assert chunks[1]["is_boilerplate"] is True
    assert chunks[1]["boilerplate_cluster"] == "bp_test_0000"
    assert chunks[1]["cluster_confidence"] == 1.0


def test_boilerplate_pass_with_empty_reference_is_noop(tmp_path):
    ref = ReferenceIndex.empty()
    chunks = [_chunk(0, _AGGIE_HONOR)]
    flag_boilerplate(chunks, ref)
    assert chunks[0]["is_boilerplate"] is False


# ---------- Within-syllabus dedup pass ----------

def test_within_syllabus_dedup_flags_near_duplicates():
    # Identical content with one chunk having a strictly-longer suffix. 5-gram Jaccard
    # at the 0.92 threshold demands very close overlap; a single edit blows ~5 n-grams
    # away. Identical-prefix-with-tail is the realistic "chunker split duplicated content"
    # scenario this pass exists to catch.
    base = (
        "lecture meets every monday wednesday and friday at ten am in heldenfels two "
        "hundred and ten students must bring laptops and arrive on time prepared for in "
        "class exercises and submit weekly homework via the course management system "
        "before the deadline that the instructor will announce at the start of the week"
    )
    chunks = [
        _chunk(0, base),
        _chunk(1, base),
        _chunk(2, "Office hours are by appointment only please email the instructor to schedule."),
    ]
    sigs = build_local_signatures(chunks)
    flag_within_syllabus_dups(chunks, "202531_TEST_101_500_99999", sigs)
    # Exactly one of the near-duplicate pair gets flagged. Canonical is the longer one (chunk 1).
    flags = [c["is_duplicate"] for c in chunks]
    assert flags.count(True) == 1
    assert chunks[2]["is_duplicate"] is False
    # The non-canonical chunk points to the canonical's chunk_id.
    flagged_idx = flags.index(True)
    canonical_idx = 1 if flagged_idx == 0 else 0
    assert chunks[flagged_idx]["duplicate_of_chunk_id"] == chunk_id_of(
        "202531_TEST_101_500_99999", chunks[canonical_idx]["chunk_index"]
    )


def test_within_syllabus_dedup_skips_short_distinct_chunks():
    chunks = [
        _chunk(0, "Short distinct chunk one with unique words like elephant."),
        _chunk(1, "Short distinct chunk two with unique words like rhinoceros."),
    ]
    sigs = build_local_signatures(chunks)
    flag_within_syllabus_dups(chunks, "TEST", sigs)
    assert all(not c["is_duplicate"] for c in chunks)


# ---------- Cross-syllabus dedup pass ----------

def _build_cross_index_from_chunks(stems_and_chunks: dict[str, list[dict]]):
    """Build a cross-syllabus LSH from synthetic data."""
    from tamubot.ingestion.pipeline_v6b.util.text_normalize import minhash_of
    rows = []
    for stem, chunks in stems_and_chunks.items():
        for c in chunks:
            mh = minhash_of(c["content"])
            rows.append(
                {
                    "chunk_id": chunk_id_of(stem, c["chunk_index"]),
                    "stem": stem,
                    "dept": stem.split("_")[1],
                    "chunk_index": c["chunk_index"],
                    "token_count": len(c["content"].split()),
                    "hashvalues": mh.hashvalues.tolist(),
                }
            )
    df = pd.DataFrame(rows)
    lsh, sigs = build_lsh_from_df(df, threshold=0.95)
    metadata = {
        str(r.chunk_id): {"stem": str(r.stem), "chunk_index": int(r.chunk_index), "dept": str(r.dept)}
        for r in df.itertuples(index=False)
    }
    return lsh, sigs, metadata


def test_cross_syllabus_dedup_flags_chunk_present_in_other_syllabus():
    stem_self = "202531_TEST_101_500_11111"
    stem_other = "202531_TEST_101_501_11112"  # lex-larger than self → self should be canonical
    shared_text = _PROGRAMMING_POLICY
    chunks_self = [_chunk(0, shared_text), _chunk(1, "Unique content for the self syllabus only.")]
    chunks_other = [_chunk(0, shared_text)]
    cross_lsh, cross_sigs, cross_metadata = _build_cross_index_from_chunks(
        {stem_self: chunks_self, stem_other: chunks_other}
    )
    sigs = build_local_signatures(chunks_self)
    flag_cross_syllabus_dups(chunks_self, stem_self, sigs, cross_lsh, cross_sigs, cross_metadata)
    # stem_self < stem_other lexicographically → self is canonical, NOT flagged.
    assert chunks_self[0]["is_duplicate"] is False


def test_cross_syllabus_dedup_flags_self_when_other_is_canonical():
    stem_self = "202531_TEST_101_999_99999"  # lex-larger
    stem_other = "202531_TEST_101_500_11111"  # lex-smaller → canonical
    shared_text = _PROGRAMMING_POLICY
    chunks_self = [_chunk(0, shared_text)]
    chunks_other = [_chunk(0, shared_text)]
    cross_lsh, cross_sigs, cross_metadata = _build_cross_index_from_chunks(
        {stem_self: chunks_self, stem_other: chunks_other}
    )
    sigs = build_local_signatures(chunks_self)
    flag_cross_syllabus_dups(chunks_self, stem_self, sigs, cross_lsh, cross_sigs, cross_metadata)
    assert chunks_self[0]["is_duplicate"] is True
    assert chunks_self[0]["duplicate_of_chunk_id"] == chunk_id_of(stem_other, 0)


def test_cross_syllabus_skips_self_match_only():
    """A chunk that only matches itself in the index must not be flagged."""
    stem_self = "202531_TEST_101_500_11111"
    chunks_self = [_chunk(0, _PROGRAMMING_POLICY)]
    cross_lsh, cross_sigs, cross_metadata = _build_cross_index_from_chunks({stem_self: chunks_self})
    sigs = build_local_signatures(chunks_self)
    flag_cross_syllabus_dups(chunks_self, stem_self, sigs, cross_lsh, cross_sigs, cross_metadata)
    assert chunks_self[0]["is_duplicate"] is False


# ---------- Integrated three-pass ----------

def test_tag_chunks_never_drops(tmp_path):
    ref = ReferenceIndex.empty()
    chunks = [_chunk(i, f"Distinct chunk number {i} with unique vocabulary {i*i}.") for i in range(7)]
    n_in = len(chunks)
    tagged, stats = tag_chunks(chunks, "TEST", ref, None, {}, {})
    assert len(tagged) == n_in
    assert stats["tagged_boilerplate"] == 0
    assert stats["tagged_duplicate"] == 0


def test_tag_chunks_boilerplate_wins_over_dedup(tmp_path):
    """A chunk flagged as boilerplate should not also be tagged as duplicate."""
    ref = _make_reference_index(tmp_path, [_AGGIE_HONOR])
    # Two chunks both contain the boilerplate text — without the boilerplate-wins-first ordering,
    # they'd flag each other as dupes.
    chunks = [_chunk(0, _AGGIE_HONOR), _chunk(1, _AGGIE_HONOR)]
    tagged, stats = tag_chunks(chunks, "TEST", ref, None, {}, {})
    assert all(c["is_boilerplate"] for c in tagged)
    assert all(not c["is_duplicate"] for c in tagged)
    assert stats["tagged_boilerplate"] == 2
    assert stats["tagged_duplicate"] == 0


def test_tag_chunks_preserves_unchanged_chunk_count():
    """Mix of boilerplate + duplicate + unique chunks — count out == count in."""
    ref = ReferenceIndex.empty()
    chunks = [
        _chunk(0, _AGGIE_HONOR),
        _chunk(1, "Unique chunk one with elephant content."),
        _chunk(2, _AGGIE_HONOR),
        _chunk(3, "Unique chunk two with rhinoceros content."),
        _chunk(4, "Unique chunk three with giraffe content."),
    ]
    n_in = len(chunks)
    tagged, _ = tag_chunks(chunks, "TEST", ref, None, {}, {})
    assert len(tagged) == n_in
