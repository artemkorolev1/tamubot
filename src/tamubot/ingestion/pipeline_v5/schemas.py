"""Pydantic models for every stage artifact. Producer asset returns the model;
consumer asset reads it. Schema mismatch fails the asset, not silently corrupts
downstream."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HeaderEntry(BaseModel):
    text: str
    level: int
    page: int | None = None


class HeadersSidecar(BaseModel):
    """Docling-emitted header→page map. Written alongside bronze markdown.
    Read by enrich + chunk to anchor sections to PDF pages."""

    schema_version: Literal["v1"] = "v1"
    headers: list[HeaderEntry] = Field(default_factory=list)


class RawPdfArtifact(BaseModel):
    """Materialization metadata for a copied raw PDF."""

    stem: str
    dept: str
    source_path: str
    output_path: str
    size_kb: float
    sha256: str


class BronzeMarkdownArtifact(BaseModel):
    """Materialization metadata for a Docling-converted bronze markdown."""

    stem: str
    dept: str
    output_path: str
    sidecar_path: str
    converter: Literal["docling", "pymupdf"]
    timing_s: float
    header_count: int
    hierarchy_depth: dict[int, int]
    token_count_estimate: int
    sha256: str
