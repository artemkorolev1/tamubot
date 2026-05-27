"""Tests for chunk schema validation."""

from tamubot.ingestion.validation.schema_validation import check_chunks_schema_valid


def _valid_chunk():
    return {
        "crn": "12345",
        "chunk_index": 0,
        "content": "hello",
        "course_id": "ISEN 620",
        "section": "500",
        "term": "Fall 2025",
        "chunk_tag": "v6b_semantic",
        "pipeline_version": "v6b",
        "source_file": "stem",
    }


def test_valid_chunks_pass():
    out = check_chunks_schema_valid([_valid_chunk(), _valid_chunk()])
    assert out.passed
    assert out.metadata["invalid_count"] == 0


def test_missing_required_field_fails():
    bad = _valid_chunk()
    del bad["crn"]
    out = check_chunks_schema_valid([_valid_chunk(), bad])
    assert out.passed is False
    assert out.metadata["invalid_count"] == 1
    assert "crn" in out.metadata["sample_errors"][0]


def test_empty_chunks_passes():
    """An empty list of chunks is structurally valid (count check is separate)."""
    assert check_chunks_schema_valid([]).passed
