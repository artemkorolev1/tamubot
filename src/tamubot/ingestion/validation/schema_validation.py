"""Chunk schema validators. Validates the required-key set written by
silver_chunk_semantic. Kept here (not in tamubot.rag.models) because this is
ingestion-side validation, not the canonical schema definition."""

from __future__ import annotations

from tamubot.ingestion.validation.types import CheckOutcome

REQUIRED_CHUNK_FIELDS = (
    "crn",
    "chunk_index",
    "content",
    "course_id",
    "section",
    "term",
    "chunk_tag",
    "pipeline_version",
    "source_file",
)


def check_chunks_schema_valid(chunks: list[dict]) -> CheckOutcome:
    errors: list[str] = []
    for i, c in enumerate(chunks):
        missing = [f for f in REQUIRED_CHUNK_FIELDS if f not in c]
        if missing:
            errors.append(f"chunk[{i}] missing: {','.join(missing)}")
    return CheckOutcome(
        passed=len(errors) == 0,
        metadata={
            "invalid_count": len(errors),
            "total": len(chunks),
            "sample_errors": errors[:10],
        },
    )
