"""Filesystem path resolution per (dept, stem, stage). Single source of truth
for the on-disk layout — assets and IO managers go through these helpers,
never hardcode."""

from __future__ import annotations

from pathlib import Path

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT, dept_from_stem


def dept_root(dept: str) -> Path:
    return DATA_ROOT / dept.upper()


def v5_root(dept: str) -> Path:
    return dept_root(dept) / "v5"


def raw_path(stem: str) -> Path:
    return v5_root(dept_from_stem(stem)) / "raw" / f"{stem}.pdf"


def bronze_md_path(stem: str) -> Path:
    return v5_root(dept_from_stem(stem)) / "bronze" / f"{stem}.md"


def bronze_sidecar_path(stem: str) -> Path:
    return v5_root(dept_from_stem(stem)) / "bronze" / f"{stem}.headers.json"


SILVER_STAGES = {
    "post_convert": "00_post_convert",
    "image_recovery": "01_image_recovery",
    "false_positive": "02_false_positive",
    "boilerplate": "03_boilerplate",
    "relocate_textbook": "03b_relocate_textbook",
    "hierarchy": "04_hierarchy",
    "enrich": "05_enrich",
    "validate": "06_validate",
    "chunk": "07_chunk",
}


def silver_path(stem: str, stage: str, ext: str = "md") -> Path:
    if stage not in SILVER_STAGES:
        raise ValueError(f"Unknown silver stage: {stage!r}")
    return v5_root(dept_from_stem(stem)) / "silver" / SILVER_STAGES[stage] / f"{stem}.{ext}"


def gold_path(stem: str) -> Path:
    return v5_root(dept_from_stem(stem)) / "gold" / f"{stem}.json"


def logs_dir(dept: str) -> Path:
    return dept_root(dept) / "logs"


def report_path(dept: str) -> Path:
    return v5_root(dept) / "silver" / "pipeline_v5_report.xlsx"
