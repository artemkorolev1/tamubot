"""Unit tests for the pure transforms in ingestion/postgres/backfill.py.

These exercise the source-doc → row-dict mapping logic with no DB or network,
so they run under plain pytest on the host (the loader itself is smoke-tested
against the live PG separately).
"""

from tamubot.ingestion.postgres.backfill import (
    EMBEDDING_MODEL,
    chunk_row,
    dept_of,
    parse_topics,
    stem_of,
    structured_rows,
    _document_rows,
)


# ── dept_of ──────────────────────────────────────────────────────────────────


def test_dept_of_splits_course_id():
    assert dept_of("ISEN 625") == "ISEN"
    assert dept_of("csce 608") == "CSCE"


def test_dept_of_handles_empty():
    assert dept_of(None) is None
    assert dept_of("") is None
    assert dept_of("   ") is None


# ── stem_of ──────────────────────────────────────────────────────────────────


def test_stem_of_strips_json_suffix():
    assert stem_of("202641_ISEN_625_601_52030.json") == "202641_ISEN_625_601_52030"


def test_stem_of_passthrough_without_suffix():
    assert stem_of("202641_ISEN_625_601_52030") == "202641_ISEN_625_601_52030"
    assert stem_of(None) is None


# ── parse_topics ─────────────────────────────────────────────────────────────


def test_parse_topics_extracts_comma_list():
    summary = (
        "ISEN 625 | Simulation\n"
        "Instructor: X\n"
        "Topics: KN method, OptQuest, Agent Based Simulation\n"
        "Prerequisites: STAT 212"
    )
    assert parse_topics(summary) == ["KN method", "OptQuest", "Agent Based Simulation"]


def test_parse_topics_dedupes_case_insensitively_preserving_order():
    assert parse_topics("Topics: A, b, a, B, c") == ["A", "b", "c"]


def test_parse_topics_empty_when_no_line():
    assert parse_topics("No topics here") == []
    assert parse_topics(None) == []
    assert parse_topics("Topics:   ") == []


# ── chunk_row ────────────────────────────────────────────────────────────────


def _sample_chunk():
    return {
        "source_file": "202641_ISEN_625_601_52030.json",
        "chunk_index": 0,
        "chunk_tag": "semantic",
        "course_id": "ISEN 625",
        "crn": "52030",
        "term": "Fall 2026",
        "section": "601",
        "instructor_name": "Amarnath Banerjee",
        "content": "## Catalog Description\n\nSimulation Methods.",
        "header_path": "Catalog Description",
        "anchor": "ISEN 625 — Catalog Description:",
        "page": 2,
        "token_count": 77,
        "has_table": False,
        "split_reason": "semantic",
        "flags": [],
        "embedding": [0.1, 0.2, 0.3],
    }


def test_chunk_row_maps_core_fields():
    c = _sample_chunk()
    c["source"] = "simple_syllabus"
    row = chunk_row(c)
    assert row["doc_type"] == "syllabus"
    assert row["source_file"] == "202641_ISEN_625_601_52030.json"
    assert row["chunk_index"] == 0
    assert row["chunk_tag"] == "semantic"
    assert row["crn"] == "52030"
    assert row["source"] == "simple_syllabus"
    assert row["embedding"] == [0.1, 0.2, 0.3]
    assert row["embedding_model"] == EMBEDDING_MODEL


def test_chunk_row_defaults_v6_flags_for_pre_v6_chunks():
    # pre-v6 chunks have no is_boilerplate / is_duplicate fields
    row = chunk_row(_sample_chunk())
    assert row["is_boilerplate"] is False
    assert row["is_duplicate"] is False
    assert row["boilerplate_cluster"] is None


def test_chunk_row_preserves_v6_tagging_when_present():
    c = _sample_chunk()
    c.update(is_boilerplate=True, boilerplate_cluster="cluster-7", is_duplicate=True)
    row = chunk_row(c)
    assert row["is_boilerplate"] is True
    assert row["boilerplate_cluster"] == "cluster-7"
    assert row["is_duplicate"] is True


def test_chunk_row_missing_embedding_has_no_model():
    c = _sample_chunk()
    del c["embedding"]
    row = chunk_row(c)
    assert row["embedding"] is None
    assert row["embedding_model"] is None


def test_chunk_row_falls_back_to_semantic_tag():
    c = _sample_chunk()
    c["chunk_tag"] = None
    assert chunk_row(c)["chunk_tag"] == "semantic"


# ── structured_rows ──────────────────────────────────────────────────────────


def test_structured_rows_splits_extract():
    extract = {
        "assessment_weights": [
            {"component": "Exam 1", "weight_pct": 18},
            {"component": "Quizzes", "weight_pct": 8},
        ],
        "letter_grade_cutoffs": [
            {"grade": "A", "min_percent": 90},
            {"grade": "F", "min_percent": 0},
        ],
        "learning_outcomes": ["Model systems", "Analyze output"],
        "meeting_schedule": [{"day": "MW", "time": "08:00", "location": "ETB 1020"}],
        "prerequisites": ["STAT 212", "ISEN 609"],
        "attendance_policy": "Miss up to two seminars.",
        "academic_integrity_policy": "Aggie Honor Code applies.",
    }
    rows = structured_rows(extract)
    assert rows["assessments"] == [
        {"component": "Exam 1", "weight_pct": 18},
        {"component": "Quizzes", "weight_pct": 8},
    ]
    assert rows["cutoffs"][1] == {"grade": "F", "min_percent": 0}
    assert rows["outcomes"] == [
        {"ordinal": 0, "text": "Model systems"},
        {"ordinal": 1, "text": "Analyze output"},
    ]
    assert rows["meetings"] == [{"ordinal": 0, "day": "MW", "time": "08:00", "location": "ETB 1020"}]
    assert rows["prerequisites"] == [
        {"ordinal": 0, "text": "STAT 212"},
        {"ordinal": 1, "text": "ISEN 609"},
    ]
    assert rows["attendance_policy"] == "Miss up to two seminars."
    assert rows["academic_integrity_policy"] == "Aggie Honor Code applies."


def test_structured_rows_empty_extract():
    rows = structured_rows({})
    assert rows == {
        "assessments": [],
        "cutoffs": [],
        "outcomes": [],
        "meetings": [],
        "prerequisites": [],
        "attendance_policy": None,
        "academic_integrity_policy": None,
    }


def test_structured_rows_drops_blank_outcomes():
    rows = structured_rows({"learning_outcomes": ["Keep", "", None, "Also keep"]})
    assert [o["text"] for o in rows["outcomes"]] == ["Keep", "Also keep"]


# ── _document_rows ───────────────────────────────────────────────────────────


def test_document_rows_one_per_source_file():
    chunks = [
        {"source_file": "a.json", "course_id": "ISEN 625", "section": "601", "source": "simple_syllabus", "crn": "52030"},
        {"source_file": "a.json", "course_id": "ISEN 625", "section": "601", "source": "simple_syllabus", "crn": "52030"},
        {"source_file": "b.json", "course_id": "CSCE 608", "section": "600", "source": "howdy_portal", "crn": "46648"},
    ]
    docs = _document_rows(chunks)
    assert len(docs) == 2
    a = next(d for d in docs if d["source_file"] == "a.json")
    assert a["doc_type"] == "syllabus"
    assert a["dept"] == "ISEN"
    assert a["title"] == "ISEN 625 601"
    assert a["source"] == "simple_syllabus"
    assert a["crn"] == "52030"


def test_document_rows_carries_crn_for_dual_source():
    # Two source_files for the SAME crn (HP + Simple Syllabus copies) → 2 docs, same crn.
    chunks = [
        {"source_file": "x_HP.json", "course_id": "ISEN 645", "section": "600", "source": "howdy_portal", "crn": "63016"},
        {"source_file": "x.json", "course_id": "ISEN 645", "section": "600", "source": "simple_syllabus", "crn": "63016"},
    ]
    docs = _document_rows(chunks)
    assert len(docs) == 2
    assert {d["crn"] for d in docs} == {"63016"}
    assert {d["source"] for d in docs} == {"howdy_portal", "simple_syllabus"}
