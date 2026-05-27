"""v6c path resolution — mirrors pipeline_v6.paths under a v6c/ root."""

from __future__ import annotations

from pathlib import Path

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT, dept_from_stem


def dept_root(dept: str) -> Path:
    return DATA_ROOT / dept.upper()


def v6c_root(dept: str) -> Path:
    return dept_root(dept) / "v6c"


def raw_path(stem: str) -> Path:
    """v6c shares raw PDFs with v5 — never copy, always read through."""
    return DATA_ROOT / dept_from_stem(stem) / "v5" / "raw" / f"{stem}.pdf"


def bronze_md_path(stem: str) -> Path:
    return v6c_root(dept_from_stem(stem)) / "bronze" / f"{stem}.md"


def bronze_sidecar_path(stem: str) -> Path:
    return v6c_root(dept_from_stem(stem)) / "bronze" / f"{stem}.headers.json"


def bronze_meta_path(stem: str) -> Path:
    return v6c_root(dept_from_stem(stem)) / "bronze" / f"{stem}.meta.json"


def bronze_odl_json_path(stem: str) -> Path:
    """Raw ODL JSON output (full structural tree). Kept for debugging."""
    return v6c_root(dept_from_stem(stem)) / "bronze" / f"{stem}.odl.json"


def bakeoff_report_path(dept: str) -> Path:
    return v6c_root(dept) / "pilot_bakeoff.xlsx"
