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


def _make_reference_index_rows(tmp_path, rows: list[dict]) -> ReferenceIndex:
    """Build a reference parquet from explicit rows (to control distinct_depts)."""
    df = pd.DataFrame(
        [
            {
                "cluster_id": r["cluster_id"],
                "representative_text": r["representative_text"],
                "normalized_text": r["representative_text"].lower(),
                "doc_frequency": r.get("doc_frequency", 80),
                "distinct_depts": r.get("distinct_depts", 4),
                "ngram_signature": [],
            }
            for r in rows
        ]
    )
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


# ---------- Header-anchored boilerplate fallback ----------
#
# A genuine cross-corpus university policy whose body has been mangled (OCR
# ligature loss + a real STAT-style variant) drops the body-Jaccard far below
# BP_THRESHOLD, so the standard body matcher declines. The header-anchored
# fallback recovers it: the leading policy header equals a known multi-dept
# policy header AND the body clears a low floor. BP_THRESHOLD is NOT lowered.

# Canonical policy body the reference cluster carries.
_MAKEUP_POLICY = (
    "## Makeup Work Policy\n\n"
    "Excused absences are governed by Student Rule 7. A student who is absent for an "
    "excused reason is entitled to make up any quiz exam or other work missed and the "
    "instructor will provide a reasonable opportunity to complete the missed work within "
    "the timelines set out in the student rules and the university calendar for the term."
)

# Same policy as it appears in the STAT syllabus: single-# header, trailing
# colon, an OCR ligature dropped ("definitions" -> "defnitions"), and reworded
# enough that the 5-gram Jaccard lands below BP_THRESHOLD but above the floor.
_MAKEUP_POLICY_VARIANT = (
    "# Makeup Work Policy:\n\n"
    "Excused absences are governed by Student Rule 7. A student who is absent for an "
    "excused reason is entitled to make up any quiz exam or other work missed and the "
    "instructor must provide a fair chance to finish the missed work soon under the "
    "defnitions in the student rules for this term."
)


def test_header_anchored_flags_below_threshold_policy_variant(tmp_path):
    ref = _make_reference_index_rows(
        tmp_path,
        [{"cluster_id": "bp_makeup", "representative_text": _MAKEUP_POLICY,
          "doc_frequency": 105, "distinct_depts": 4}],
    )
    # Sanity: the body matcher alone declines (Jaccard < BP_THRESHOLD).
    assert ref.match(_MAKEUP_POLICY_VARIANT, threshold=0.80)[0] is None
    chunks = [_chunk(0, _MAKEUP_POLICY_VARIANT)]
    flag_boilerplate(chunks, ref)
    assert chunks[0]["is_boilerplate"] is True
    assert chunks[0]["boilerplate_cluster"] == "bp_makeup"
    assert chunks[0]["boilerplate_match_source"] == "header_anchored"
    # Recorded confidence is the (below-threshold) body Jaccard, for audit.
    assert 0.05 <= chunks[0]["cluster_confidence"] < 0.80


def test_header_anchored_does_not_flag_unknown_header(tmp_path):
    # NEGATIVE: a chunk whose header is NOT in the allow-list and whose body
    # Jaccard is below BP_THRESHOLD must NOT be flagged (no false positive).
    ref = _make_reference_index_rows(
        tmp_path,
        [{"cluster_id": "bp_makeup", "representative_text": _MAKEUP_POLICY,
          "doc_frequency": 105, "distinct_depts": 4}],
    )
    course_chunk = (
        "# Course Project Milestones:\n\n"
        "Please refer to Student Rule 7 for excused absences including defnitions and the "
        "related documentation and timelines while you plan your project milestones each term."
    )
    assert ref.match(course_chunk, threshold=0.80)[0] is None  # below BP_THRESHOLD
    chunks = [_chunk(0, course_chunk)]
    flag_boilerplate(chunks, ref)
    assert chunks[0]["is_boilerplate"] is False


def test_header_anchored_skips_one_off_low_dept_cluster(tmp_path):
    # NEGATIVE: a policy-looking header whose only reference cluster spans too
    # few departments is a one-off, not a real cross-corpus policy → no
    # allow-list entry → not header-flagged even with matching header text.
    ref = _make_reference_index_rows(
        tmp_path,
        [{"cluster_id": "bp_attend", "representative_text":
          "## Attendance Policy\n\nThe instructor takes roll at the start of every lecture "
          "and unexcused absences beyond three will lower the participation grade for this "
          "specific course section as described in the course management system this term.",
          "doc_frequency": 6, "distinct_depts": 3}],
    )
    chunk = _chunk(0, "# Attendance Policy:\n\nThe instructor takes roll at the start of "
                      "every lecture and unexcused absences beyond three will lower the grade.")
    chunks = [chunk]
    flag_boilerplate(chunks, ref)
    assert chunks[0]["is_boilerplate"] is False


def test_header_anchored_skips_zero_body_overlap(tmp_path):
    # NEGATIVE: a same-titled but genuinely course-specific section with ZERO
    # body overlap (Jaccard < floor) is NOT flagged — the floor guards it.
    ref = _make_reference_index_rows(
        tmp_path,
        [{"cluster_id": "bp_makeup", "representative_text": _MAKEUP_POLICY,
          "doc_frequency": 105, "distinct_depts": 4}],
    )
    course_specific = (
        "# Makeup Work Policy:\n\n"
        "For this seminar specifically your makeup presentation must reschedule with the "
        "guest speaker liaison and upload slides to the shared workshop drive within fortyeight "
        "hours of the rescheduled robotics demonstration slot assigned by the lab coordinator."
    )
    chunks = [_chunk(0, course_specific)]
    flag_boilerplate(chunks, ref)
    assert chunks[0]["is_boilerplate"] is False


def test_header_anchored_rejects_course_specific_signal(tmp_path):
    # NEGATIVE (over-hide guard): a chunk under a standard policy header whose body
    # clears the Jaccard floor BUT carries an instructor's own grade percentage is a
    # CUSTOMIZED section — header-anchoring must NOT flag it (the ECEN_749 /
    # CSCE_765 "Late Work Policy … 20% penalty" corpus-wide over-hide class).
    ref = _make_reference_index_rows(
        tmp_path,
        [{"cluster_id": "bp_makeup", "representative_text": _MAKEUP_POLICY,
          "doc_frequency": 105, "distinct_depts": 4}],
    )
    # Same wording as the flaggable variant (clears the floor) + a course-specific %.
    customized = _MAKEUP_POLICY_VARIANT + (
        "\n\nLate makeup submissions will be accepted with a penalty of 20% per day "
        "for this section only."
    )
    # Sanity: the body matcher alone still declines (below BP_THRESHOLD).
    assert ref.match(customized, threshold=0.80)[0] is None
    chunks = [_chunk(0, customized)]
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


def test_within_syllabus_dedup_three_identical_keeps_one_canonical():
    # Union-find over all pairs → one component, one canonical, deterministically.
    chunks = [_chunk(0, _PROGRAMMING_POLICY), _chunk(1, _PROGRAMMING_POLICY), _chunk(2, _PROGRAMMING_POLICY)]
    sigs = build_local_signatures(chunks)
    flag_within_syllabus_dups(chunks, "TEST", sigs)
    flags = [c["is_duplicate"] for c in chunks]
    assert flags.count(True) == 2  # exactly one canonical survives
    assert chunks[0]["is_duplicate"] is False  # equal length → lowest index wins
    for i in (1, 2):
        assert chunks[i]["duplicate_of_chunk_id"] == chunk_id_of("TEST", 0)


def test_cross_syllabus_dedup_excludes_same_stem_twins():
    # An intra-doc duplicate pair is owned by within-syllabus dedup; the cross
    # pass must not re-flag the within-canonical against its own same-stem twin.
    stem = "202531_TEST_101_500_11111"
    chunks_self = [_chunk(0, _PROGRAMMING_POLICY), _chunk(1, _PROGRAMMING_POLICY)]
    cross_lsh, cross_sigs, cross_metadata = _build_cross_index_from_chunks({stem: chunks_self})
    # Simulate within-pass already collapsed the pair (chunk 1 is the duplicate).
    chunks_self[1]["is_duplicate"] = True
    sigs = build_local_signatures(chunks_self)
    flag_cross_syllabus_dups(chunks_self, stem, sigs, cross_lsh, cross_sigs, cross_metadata)
    assert chunks_self[0]["is_duplicate"] is False  # only match is a same-stem twin


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


def test_cross_syllabus_dedup_guard_keeps_url_line_visible():
    """Dedup over-hide guard (CSCE_629): a cross-syllabus near-duplicate carrying a
    URL must stay visible even when another stem is the lex-canonical, because the
    canonical lives in a DIFFERENT stem and collapsing would drop the link from this
    section's retrieval."""
    url_text = (
        "Important course logistics for this section are posted on the learning "
        "management system. Also check daily for announcements and updated material "
        "at https://canvas.tamu.edu/ which is the authoritative source for this "
        "course schedule readings homework deadlines and any policy changes."
    )
    stem_self = "202531_TEST_101_999_99999"  # lex-larger → would normally be flagged
    stem_other = "202531_TEST_101_500_11111"  # lex-smaller → canonical
    chunks_self = [_chunk(0, url_text)]
    chunks_other = [_chunk(0, url_text)]
    cross_lsh, cross_sigs, cross_metadata = _build_cross_index_from_chunks(
        {stem_self: chunks_self, stem_other: chunks_other}
    )
    sigs = build_local_signatures(chunks_self)
    flag_cross_syllabus_dups(chunks_self, stem_self, sigs, cross_lsh, cross_sigs, cross_metadata)
    assert chunks_self[0]["is_duplicate"] is False
    assert chunks_self[0]["dedup_overhide_protected"] is True


def test_cross_syllabus_dedup_guard_does_not_protect_plain_boilerplate():
    """Control: a signal-free cross-syllabus duplicate is still collapsed as before —
    the guard only spares actionable section-specific lines."""
    stem_self = "202531_TEST_101_999_99999"  # lex-larger → flagged
    stem_other = "202531_TEST_101_500_11111"  # lex-smaller → canonical
    chunks_self = [_chunk(0, _AGGIE_HONOR)]
    chunks_other = [_chunk(0, _AGGIE_HONOR)]
    cross_lsh, cross_sigs, cross_metadata = _build_cross_index_from_chunks(
        {stem_self: chunks_self, stem_other: chunks_other}
    )
    sigs = build_local_signatures(chunks_self)
    flag_cross_syllabus_dups(chunks_self, stem_self, sigs, cross_lsh, cross_sigs, cross_metadata)
    assert chunks_self[0]["is_duplicate"] is True


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
