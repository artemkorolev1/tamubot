"""Pydantic v2 models for the V4 pipeline collections: chunks_v4, courses_v4."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from tamubot.rag.models import Instructor


class ChunkDocV4(BaseModel):
    """One document in the *chunks_v4* collection.

    Semantic chunks with header_path hierarchy — replaces flat token chunks from v3.
    """

    # Unique key for idempotent upserts: crn + chunk_index
    crn: str
    chunk_index: int

    # Chunk content
    content: str
    has_table: bool = False

    # Denormalized course metadata
    course_id: str
    section: str
    term: str
    instructor_name: Optional[str] = None

    # V4 chunk metadata
    header_path: Optional[str] = None
    token_count: Optional[int] = None
    flags: list[str] = Field(default_factory=list)
    split_reason: Optional[str] = None
    page: Optional[int] = None
    source: Optional[str] = None  # "simple_syllabus" or "howdy_portal"

    # Anchor for embedding (built from header_path)
    anchor: str = ""

    # Embedding (populated during ingestion)
    embedding: Optional[list[float]] = None

    # Housekeeping
    pipeline_version: str = "v4"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class SummaryStatement(BaseModel):
    """One page-anchored fact extracted from a course's chunks.

    Together, a list of these forms the page-citable summary for a course.
    """

    text: str
    page: Optional[int] = None
    header_path: Optional[str] = None  # for diagnostics; not shown to user


class CourseDocV4(BaseModel):
    """One document per course section (CRN) in *courses_v4*."""

    crn: str
    course_id: str
    section: str
    term: str
    instructor: Optional[Instructor] = None
    teaching_assistants: list[str] = Field(default_factory=list)
    meeting_times: Optional[str] = None
    location: Optional[str] = None
    credit_hours: Optional[str] = None
    chunk_count: int = 0
    syllabus_url: Optional[str] = None

    # V4 fields
    source: Optional[str] = None  # "simple_syllabus" or "howdy_portal"
    course_type: Optional[str] = None
    format: Optional[str] = None  # "in-person", "online", etc.
    prerequisites: Optional[str] = None
    course_summary: Optional[str] = None
    summary_statements: list[SummaryStatement] = Field(default_factory=list)
    chunk_config: Optional[dict] = None

    # Housekeeping
    pipeline_version: str = "v4"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
