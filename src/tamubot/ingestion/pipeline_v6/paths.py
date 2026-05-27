"""v6 path resolution — parallel to v5 paths but rooted at `<dept>/v6/`."""

from __future__ import annotations

from pathlib import Path

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT, dept_from_stem


def dept_root(dept: str) -> Path:
    return DATA_ROOT / dept.upper()


def v6_root(dept: str) -> Path:
    return dept_root(dept) / "v6"


def raw_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "raw" / f"{stem}.pdf"


def bronze_md_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "bronze" / f"{stem}.md"


def bronze_sidecar_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "bronze" / f"{stem}.headers.json"


def bronze_meta_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "bronze" / f"{stem}.meta.json"


def chunks_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "chunks" / f"{stem}.json"


def tagged_chunks_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "tagged" / f"{stem}.json"


def validate_path(stem: str) -> Path:
    return v6_root(dept_from_stem(stem)) / "validate" / f"{stem}.json"


def bakeoff_report_path(dept: str) -> Path:
    return v6_root(dept) / "pilot_bakeoff.xlsx"


def reference_set_path() -> Path:
    return DATA_ROOT / "_meta" / "boilerplate_reference.parquet"
