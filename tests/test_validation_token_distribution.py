"""Tests for chunk-distribution validators using chunker_v4 flags."""

from tamubot.ingestion.validation.token_distribution import (
    check_chunk_count_nonzero,
    check_low_no_header_rate,
    check_no_header_only_chunks,
    check_no_oversized_chunks,
)


def test_chunk_count_nonzero_pass():
    out = check_chunk_count_nonzero([{}, {}, {}])
    assert out.passed
    assert out.metadata["chunk_count"] == 3


def test_chunk_count_nonzero_fail():
    out = check_chunk_count_nonzero([])
    assert out.passed is False


def test_no_oversized_chunks_pass():
    chunks = [{"flags": []}, {"flags": ["HAS_TABLE"]}]
    assert check_no_oversized_chunks(chunks).passed


def test_no_oversized_chunks_fail():
    chunks = [{"flags": ["OVERSIZED"]}, {"flags": []}]
    out = check_no_oversized_chunks(chunks)
    assert out.passed is False
    assert out.metadata["oversized_count"] == 1


def test_low_no_header_rate_pass():
    chunks = [{"flags": []}] * 9 + [{"flags": ["NO_HEADER"]}]  # 10% exactly
    out = check_low_no_header_rate(chunks, max_rate=0.10)
    assert out.passed


def test_low_no_header_rate_fail():
    chunks = [{"flags": []}] * 8 + [{"flags": ["NO_HEADER"]}] * 2  # 20%
    out = check_low_no_header_rate(chunks, max_rate=0.10)
    assert out.passed is False
    assert out.metadata["no_header_rate"] == 0.20


# --- Check 2: header-only orphan chunk (CHUNK_ORPHAN_HEADER / STAT_620) --------


def test_header_only_chunk_flagged():
    """A chunk whose content is only header lines (no body) is a body-less orphan."""
    chunks = [
        {"content": "# Course Schedule\n\n## Week 1", "header_path": "Course Schedule"},
        {"content": "# Grading\n\nHomework is 40% of the grade.", "header_path": "Grading"},
    ]
    out = check_no_header_only_chunks(chunks)
    assert out.passed is False
    assert out.metadata["header_only_count"] == 1
    assert "Course Schedule" in out.metadata["offending_header_paths"]


def test_chunk_with_body_not_flagged():
    chunks = [
        {"content": "## Attendance\n\nStudents must attend all sessions.", "header_path": "Attendance"},
        {"content": "Plain body text with no header.", "header_path": ""},
    ]
    out = check_no_header_only_chunks(chunks)
    assert out.passed
    assert out.metadata["header_only_count"] == 0


def test_empty_content_chunk_not_treated_as_header_only():
    """Empty/blank content is a different failure class, not an orphan header."""
    out = check_no_header_only_chunks([{"content": "\n\n   \n", "header_path": "Blank"}])
    assert out.passed
    assert out.metadata["header_only_count"] == 0
