"""In-memory index of known TAMU course IDs + titles.

Used by the router validator to:
  * Reject hallucinated course_ids (e.g. LLM emits `ISEN 999` — not in index).
  * Recover a course_id from a paraphrased title (e.g. user query
    "Management of Engineering Systems" → "ISEN 601").

Spec note: the design doc names ``chunks_v4`` as the source, but chunks have no
``course_title`` field. The titles live in ``courses_v4.course_summary`` as the
first line, formatted ``"COURSE_ID[/ALIAS...] | Title | Term"`` — we parse that.

Loaded lazily once per process. Mongo failure raises (no silent empty-index
fallback — see CLAUDE.md "fail loud").
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from tamubot.rag.utils import normalize_course_id


@dataclass(frozen=True)
class CourseRef:
    course_id: str  # "ISEN 601"
    title_lower: str  # "management of engineering systems"


_INDEX: Optional[dict[str, CourseRef]] = None
_TITLE_PAIRS: Optional[list[tuple[str, str]]] = None  # (title_lower, course_id), longest title first
_LOCK = Lock()


def _parse_summary_first_line(line: str) -> tuple[list[str], Optional[str]]:
    """Parse ``"ISEN 689 / AERO 636 | Title | Term"`` → (["ISEN 689","AERO 636"], "Title").

    Returns ([], None) on malformed input.
    """
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 2:
        return [], None
    raw_ids = [tok.strip() for tok in parts[0].replace(",", "/").split("/")]
    course_ids = [normalize_course_id(tok) for tok in raw_ids if tok]
    course_ids = [cid for cid in course_ids if cid]
    title = parts[1].strip()
    if not title:
        return course_ids, None
    return course_ids, title


def _build_from_docs(docs) -> tuple[dict[str, CourseRef], list[tuple[str, str]]]:
    """Build the {course_id: CourseRef} index and (title, course_id) pairs list.

    Index includes every parsed course_id (cross-listed aliases too) so that
    ``is_known_course_id`` accepts them for grounding. ``title_pairs`` only
    contains the doc's *primary* course_id per title — aliases live under the
    canonical doc, and Mongo chunks are stored under the primary id, so
    ``match_title`` must return the primary to avoid retrieving zero chunks
    from an alias filter.
    """
    index: dict[str, CourseRef] = {}
    primary_title_pairs: list[tuple[str, str]] = []
    seen_primary_titles: set[str] = set()

    for doc in docs:
        summary = doc.get("course_summary") or ""
        first_line = summary.split("\n", 1)[0]
        course_ids, title = _parse_summary_first_line(first_line)
        primary = normalize_course_id(doc.get("course_id") or "")
        title_lower = title.lower() if title else ""

        # Register every parsed id (primary + aliases) for grounding lookups.
        for cid in course_ids:
            if cid not in index:
                index[cid] = CourseRef(course_id=cid, title_lower=title_lower)
        if primary and primary not in index:
            index[primary] = CourseRef(course_id=primary, title_lower=title_lower)

        # Title → primary id mapping (one entry per (title, primary)).
        if title_lower and primary:
            key = f"{title_lower}|{primary}"
            if key not in seen_primary_titles:
                seen_primary_titles.add(key)
                primary_title_pairs.append((title_lower, primary))

    # Sort by descending title length so match_title finds the longest first.
    title_pairs = sorted(primary_title_pairs, key=lambda p: len(p[0]), reverse=True)
    return index, title_pairs


def _load() -> tuple[dict[str, CourseRef], list[tuple[str, str]]]:
    from tamubot.rag.tools.mongo import COURSES_COLLECTION, _get_db

    db = _get_db()
    docs = db[COURSES_COLLECTION].find({}, {"course_id": 1, "course_summary": 1})
    return _build_from_docs(docs)


def get_index() -> dict[str, CourseRef]:
    """Return the {course_id: CourseRef} map. Loads on first call."""
    global _INDEX, _TITLE_PAIRS
    if _INDEX is None:
        with _LOCK:
            if _INDEX is None:
                idx, pairs = _load()
                _TITLE_PAIRS = pairs
                _INDEX = idx
    return _INDEX


def _get_title_pairs() -> list[tuple[str, str]]:
    get_index()
    assert _TITLE_PAIRS is not None
    return _TITLE_PAIRS


def is_known_course_id(cid: str) -> bool:
    if not cid:
        return False
    normalized = normalize_course_id(cid)
    return normalized in get_index()


def match_title(text: str) -> Optional[str]:
    """Longest exact substring match against known course titles, case-insensitive.

    Returns the matched course_id, or None when no title matches.
    """
    if not text:
        return None
    haystack = text.lower()
    for title_lower, cid in _get_title_pairs():
        if title_lower and title_lower in haystack:
            return cid
    return None


def _reset_for_tests() -> None:
    """Test helper — clears the cached index."""
    global _INDEX, _TITLE_PAIRS
    with _LOCK:
        _INDEX = None
        _TITLE_PAIRS = None
