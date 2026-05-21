"""Discover source PDFs across simple_syllabus and howdy_portal trees,
returning a stem → source-path mapping. Used to register dynamic partitions
and to resolve `raw_pdf`'s upstream input."""

from __future__ import annotations

from pathlib import Path

from tamubot.ingestion.pipeline_v5.util import RAW_SOURCE, RAW_SOURCE_HP


def discover_source_pdfs(dept: str, term_code: str | None = None) -> dict[str, Path]:
    """{stem: source_pdf_path} for a department. Simple Syllabus wins on
    stem-conflict (mirrors the iterate-pipeline skill convention)."""
    dept = dept.upper()
    found: dict[str, Path] = {}

    for dept_dir in RAW_SOURCE_HP.glob(f"{dept}/*"):
        if not dept_dir.is_dir():
            continue
        for season_dir in sorted(dept_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            for pdf in sorted(season_dir.glob("*.pdf")):
                if term_code and not pdf.stem.startswith(term_code):
                    continue
                found[f"{pdf.stem}_HP"] = pdf

    for dept_dir in RAW_SOURCE.glob(f"{dept}/*"):
        if not dept_dir.is_dir():
            continue
        for season_dir in sorted(dept_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            for pdf in sorted(season_dir.glob("*.pdf")):
                if term_code and not pdf.stem.startswith(term_code):
                    continue
                found[pdf.stem] = pdf

    return dict(sorted(found.items()))
