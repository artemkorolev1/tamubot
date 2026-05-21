"""Shared helpers for pipeline_v5. No imports from pipeline_v4."""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Callable
from pathlib import Path

DATA_ROOT = Path("data/syllabi")
RAW_SOURCE = Path("tamu_data/raw/simple_syllabus")
RAW_SOURCE_HP = Path("tamu_data/raw/howdy_portal")

_SEASON_TO_CODE = {"spring": "11", "summer": "21", "fall": "41"}
_CODE_TO_SEASON = {"11": "Spring", "21": "Summer", "41": "Fall"}


def term_to_code(term: str) -> str:
    """'Fall 2025' → '202541'."""
    parts = term.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid term: {term!r}. Expected 'Season YYYY'.")
    season, year = parts[0].lower(), parts[1]
    if season not in _SEASON_TO_CODE:
        raise ValueError(f"Unknown season: {parts[0]!r}.")
    if not year.isdigit() or len(year) != 4:
        raise ValueError(f"Invalid year: {year!r}.")
    return f"{year}{_SEASON_TO_CODE[season]}"


def code_to_term(code: str) -> str:
    """'202541' → 'Fall 2025'."""
    if len(code) != 6 or not code.isdigit():
        return code
    return f"{_CODE_TO_SEASON.get(code[4:], '?' + code[4:])} {code[:4]}"


def dept_from_stem(stem: str) -> str:
    """'202541_CSCE_605_600_62077' → 'CSCE'. Tolerates '_HP' suffix."""
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot extract dept from stem: {stem!r}")
    return parts[1].upper()


def course_from_stem(stem: str) -> str:
    """'202541_CSCE_605_600_62077' → 'CSCE-605'."""
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot extract course from stem: {stem!r}")
    return f"{parts[1].upper()}-{parts[2]}"


def term_code_from_stem(stem: str) -> str:
    """'202541_CSCE_605_600_62077' → '202541'."""
    return stem.split("_")[0]


def section_from_stem(stem: str) -> str | None:
    parts = stem.split("_")
    return parts[3] if len(parts) >= 4 else None


def code_version_of(fn: Callable) -> str:
    """Auto-derived code_version: sha256 of the function's source. Bumps
    automatically on any logic edit, so downstream Dagster assets go stale."""
    try:
        src = inspect.getsource(fn)
    except OSError, TypeError:
        src = repr(fn)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def hash_file(path: Path) -> str:
    """SHA256 (first 16 hex chars) of file bytes. Used for dedupe and
    materialization metadata."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


_LEADING_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\s+")


def build_page_lookup(headers: list[dict]) -> dict[str, int]:
    """Lowercase header-text → page mapping. Indexes both raw and
    leading-number-stripped variants. Lifted from pipeline_v4._build_page_lookup."""
    lookup: dict[str, int] = {}
    for h in headers:
        text = h.get("text", "").strip().lower()
        page = h.get("page")
        if text and page is not None:
            lookup[text] = page
            stripped = _LEADING_NUM_RE.sub("", text)
            if stripped != text:
                lookup[stripped] = page
    return lookup
