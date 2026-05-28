"""Pydantic v2 models for the V4 pipeline collections: chunks_v4, courses_v4."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

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

    # Chunking-strategy tag (e.g. "semantic", "600t_100o_fixed") + size/overlap for fixed runs
    chunk_tag: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    # Anchor for embedding (built from header_path)
    anchor: str = ""

    # Embedding (populated during ingestion)
    embedding: Optional[list[float]] = None

    # Housekeeping
    pipeline_version: str = "v4"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


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
    chunk_config: Optional[dict] = None

    # Housekeeping
    pipeline_version: str = "v4"
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class MeetingSlot(BaseModel):
    """One scheduled meeting (lecture or office hours)."""

    day: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None


class AssessmentWeight(BaseModel):
    """A graded component and its weight toward the final grade (e.g. Midterm = 25%)."""

    component: Optional[str] = None
    weight_pct: Optional[float] = None


class LetterGradeCutoff(BaseModel):
    """A letter grade and its minimum percentage cutoff (e.g. A >= 90)."""

    grade: Optional[str] = None
    min_percent: Optional[float] = None


class SyllabusExtract(BaseModel):
    """Structured fields extracted from a single syllabus by NuExtract3.

    `grading` is split into two distinct concepts: `assessment_weights`
    (component -> weight toward grade) and `letter_grade_cutoffs` (letter ->
    minimum percent). The model conflated these when they shared one field.
    Unknown fields the model occasionally emits are dropped (extra="ignore").
    """

    model_config = ConfigDict(extra="ignore")

    course_code: Optional[str] = None
    course_title: Optional[str] = None
    instructor_name: Optional[str] = None
    instructor_email: Optional[str] = None
    credit_hours: Optional[int] = None
    meeting_schedule: list[MeetingSlot] = Field(default_factory=list)
    assessment_weights: list[AssessmentWeight] = Field(default_factory=list)
    letter_grade_cutoffs: list[LetterGradeCutoff] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    attendance_policy: Optional[str] = None
    academic_integrity_policy: Optional[str] = None
