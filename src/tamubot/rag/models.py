"""Pydantic v2 models for the three MongoDB collections: chunks, policies, courses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class Instructor(BaseModel):
    name: str
    email: Optional[str] = None
    office: Optional[str] = None
    office_hours: Optional[str] = None


class CourseMetadata(BaseModel):
    course_id: str  # e.g. "CSCE 120"
    section: str
    term: str
    crn: str
    instructor: Optional[Instructor] = None
    teaching_assistants: list[str] = Field(default_factory=list)
    meeting_times: Optional[str] = None
    location: Optional[str] = None
    credit_hours: Optional[str] = None


# ---------------------------------------------------------------------------
# chunks collection
# ---------------------------------------------------------------------------


class ChunkDoc(BaseModel):
    """One document in the *chunks* collection.

    Denormalized: course metadata is embedded directly so every chunk is
    self-contained for retrieval.
    """

    # Unique key for idempotent upserts:  crn + chunk_index
    crn: str
    chunk_index: int

    # Chunk content
    title: str
    content: str
    has_table: bool = False

    # Contextual anchor prepended before embedding (not stored in content)
    anchor: str = ""

    # Denormalized course metadata
    course_id: str
    section: str
    term: str
    instructor_name: Optional[str] = None

    # Embedding (populated during ingestion)
    embedding: Optional[list[float]] = None

    # Housekeeping
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# policies collection
# ---------------------------------------------------------------------------


class PolicyDoc(BaseModel):
    """Deduplicated university boilerplate policies.

    Keyed by SHA-256 hash of the policy name so identical names across
    syllabi collapse into a single document.
    """

    policy_hash: str  # SHA-256 hex of lowercased policy name
    policy_name: str
    # Full text will be filled in once a golden-copy source is available.
    full_text: Optional[str] = None
    courses_referencing: list[str] = Field(default_factory=list)  # list of CRNs
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# courses collection
# ---------------------------------------------------------------------------


class CourseDoc(BaseModel):
    """One document per course *section* (CRN), for aggregate queries."""

    crn: str
    course_id: str
    section: str
    term: str
    instructor: Optional[Instructor] = None
    teaching_assistants: list[str] = Field(default_factory=list)
    meeting_times: Optional[str] = None
    location: Optional[str] = None
    credit_hours: Optional[str] = None
    categories_present: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    boilerplate_policies: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    completeness_warnings: list[str] = Field(default_factory=list)
    source_file: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
