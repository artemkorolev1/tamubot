"""Filesystem path resolution for v6b. Mirrors pipeline_v5.paths layout under a
v6b/ root so artifacts don't collide with v5 or v6."""

from __future__ import annotations

from pathlib import Path

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT, dept_from_stem


def dept_root(dept: str) -> Path:
    return DATA_ROOT / dept.upper()


def v6b_root(dept: str) -> Path:
    return dept_root(dept) / "v6b"


def raw_path(stem: str) -> Path:
    """v6b shares raw PDFs with v5 — never copy, always read through."""
    return DATA_ROOT / dept_from_stem(stem) / "v5" / "raw" / f"{stem}.pdf"


def bronze_blocks_path(stem: str) -> Path:
    return v6b_root(dept_from_stem(stem)) / "bronze" / f"{stem}.blocks.json"


def bronze_md_path(stem: str) -> Path:
    return v6b_root(dept_from_stem(stem)) / "bronze" / f"{stem}.md"


def bronze_headers_path(stem: str) -> Path:
    return v6b_root(dept_from_stem(stem)) / "bronze" / f"{stem}.headers.json"


def silver_modal_path(stem: str) -> Path:
    return v6b_root(dept_from_stem(stem)) / "silver" / "01_modal" / f"{stem}.blocks.json"


def silver_modal_markdown_path(stem: str) -> Path:
    """Merged markdown — used by silver_chunk_semantic."""
    return v6b_root(dept_from_stem(stem)) / "silver" / "01_modal" / f"{stem}.md"


def silver_chunk_semantic_path(stem: str) -> Path:
    return v6b_root(dept_from_stem(stem)) / "silver" / "02_chunk" / f"{stem}.json"


def silver_tag_path(stem: str, variant: str = "semantic") -> Path:
    """variant kept for forward-compat; only "semantic" supported today."""
    if variant != "semantic":
        raise ValueError(f"variant must be 'semantic' (block variant removed), got {variant!r}")
    return v6b_root(dept_from_stem(stem)) / "silver" / "03_tag" / f"{stem}.json"


def silver_embed_path(stem: str) -> Path:
    """Embedded-chunk output for v6b_silver_embed. Separate from silver_tag's
    output so silver_embed does not mutate upstream artifacts."""
    return v6b_root(dept_from_stem(stem)) / "silver" / "04_embed" / f"{stem}.json"


def modal_cache_path() -> Path:
    """Cross-stem cache for vision LLM calls. Keyed by image/table content hash."""
    return DATA_ROOT / "_meta" / "modal_cache.jsonl"


def boilerplate_reference_path() -> Path:
    """Shared with v6 — both pipelines read the same parquet."""
    return DATA_ROOT / "_meta" / "boilerplate_reference.parquet"


def report_path(dept: str) -> Path:
    return v6b_root(dept) / "pilot_bakeoff.xlsx"
