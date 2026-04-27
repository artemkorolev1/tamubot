"""Filter: demote false-positive headers back to body text.

Docling sometimes promotes body text to headers. This filter catches known
false-positive patterns and removes the ``#`` prefix, returning them to
plain body text.

CLI: python -m tamubot.ingestion.filters.false_positive input_dir/ output_dir/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tamubot.ingestion.filters.base import FilterResult

# ── False-positive registry ──────────────────────────────────────────────────

# Exact matches (compared case-insensitively, stripped, colon-removed)
_EXACT_FP: frozenset[str] = frozenset(
    s.lower()
    for s in [
        "Notes",
        "URL for Resource",
        "Texas A&M College Station",
        "Texas A&M University",
    ]
)

# Prefix matches — if the normalized header starts with any of these, it's a FP.
# Handles variants like "This material Is: Required", "This material Is: Recommended".
_PREFIX_FP: list[str] = [
    "this material is",
]

# Regex matches — compiled patterns tested against normalized header text.
_REGEX_FP: list[tuple[re.Pattern, str]] = [
    # Numbered list items promoted to headers by Docling (e.g. "1. The time, date...")
    (re.compile(r"^\d+\.\s"), "numbered_list_item"),
    # Time strings promoted to headers (e.g. "9:35 -10:50 AM (CDT)")
    (re.compile(r"^\d{1,2}:\d{2}\s*[-–]"), "time_string"),
    # Room/building lines promoted to headers (e.g. "Room 2026, Emerging Technology Building")
    (re.compile(r"^room\s+\d", re.IGNORECASE), "room_location"),
    # Policy statements promoted to headers (e.g. "All quizzes and exams are closed book")
    (re.compile(r"^all\s+(quizzes|exams|tests|homework)", re.IGNORECASE), "policy_statement"),
    # Single sentence about absence promoted to header
    (re.compile(r"^(a|an)\s+absence\s+for\s+", re.IGNORECASE), "policy_statement"),
]

# Text patterns to strip from output (not headers — inline artifacts)
TEXT_CLEANUP_PATTERNS: list[str] = [
    "<!-- image -->",
    "|    |\n|----|",
]

# Regex-based text cleanups: fix spaces inside URLs from PDF line breaks.
# Matches a URL followed by a space and a URL-path continuation (starts with
# alphanumeric and contains / or . or = or ? typical of URL paths).
_URL_SPACE_RE = re.compile(r"(https?://\S+)" r"\s+" r"([A-Za-z0-9][\w./?=&%-]*(?:/|\.[\w]+|[?=&])\S*)")

# Repeated "Course Syllabus" page headers injected by Docling across page breaks.
# Matches any heading level (# to ######) with optional ":" suffix.
_PAGE_HEADER_RE = re.compile(r"^\s*#{1,6}\s+Course\s+Syllabus:?\s*$", re.MULTILINE | re.IGNORECASE)

# Spaced-out capital letters from Docling small-caps rendering.
# Detects runs like "T ITLE IX AND S TATEMENT" (single uppercase letter + space, 3+ times)
_SPACED_CAPS_RE = re.compile(r"\b((?:[A-Z]\s){3,}[A-Z])\b")


def _fix_broken_urls(text: str) -> tuple[str, int]:
    """Remove spaces injected into URLs by PDF line-break artifacts.

    Only merges when the text after the space looks like a URL-path
    continuation (contains / or . or query params), not regular prose.
    """
    new_text, count = _URL_SPACE_RE.subn(r"\1\2", text)
    return new_text, count


def _collapse_spaced_caps(text: str) -> tuple[str, int]:
    """Collapse spaced-out capitals like 'T ITLE IX' → 'TITLE IX'."""

    def _collapse(m: re.Match) -> str:
        return m.group(1).replace(" ", "")

    new_text, count = _SPACED_CAPS_RE.subn(_collapse, text)
    return new_text, count


def _strip_page_headers(text: str) -> tuple[str, int]:
    """Remove repeated 'Course Syllabus' page-header lines."""
    # Keep the FIRST occurrence (likely the real title), strip the rest
    first_found = False
    lines = text.splitlines()
    out = []
    removed = 0
    for line in lines:
        if _PAGE_HEADER_RE.match(line):
            if not first_found:
                first_found = True
                out.append(line)
            else:
                removed += 1
        else:
            out.append(line)
    return "\n".join(out), removed


_HEADER_RE = re.compile(r"^\s*(#{1,6})\s+(.+)$")


def _is_false_positive(header_text: str) -> str | None:
    """Return a reason string if *header_text* is a false positive, else None."""
    norm = header_text.strip().rstrip(":").lower()

    # 1. Exact match
    if norm in _EXACT_FP:
        return "exact_match"

    # 2. Prefix match (e.g. "this material is: required")
    for prefix in _PREFIX_FP:
        if norm == prefix or norm.startswith(prefix + ":") or norm.startswith(prefix + " "):
            return "prefix_match"

    # 3. Regex match (e.g. numbered list items promoted to headers)
    for pattern, reason in _REGEX_FP:
        if pattern.search(norm):
            return reason

    return None


def _estimate_tokens(text: str) -> int:
    return max(0, round(len(text) / 4))


class FalsePositiveFilter:
    """Demote known false-positive headers back to body text."""

    name: str = "false_positive"

    def apply(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any] | None = None,
    ) -> FilterResult:
        config = config or {}
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = FilterResult()
        md_files = sorted(input_dir.glob("*.md"))
        result.input_count = len(md_files)

        total_demoted = 0

        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            demoted: list[dict[str, Any]] = []
            out_lines: list[str] = []

            for line in text.splitlines():
                m = _HEADER_RE.match(line)
                if m:
                    level = len(m.group(1))
                    header_text = m.group(2).strip()
                    reason = _is_false_positive(header_text)
                    if reason:
                        out_lines.append(header_text)
                        demoted.append(
                            {
                                "header": header_text,
                                "level": level,
                                "reason": reason,
                            }
                        )
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)

            out_text = "\n".join(out_lines)

            # Apply text cleanup patterns (inline artifacts)
            cleanups_applied = 0
            for pattern in TEXT_CLEANUP_PATTERNS:
                count = out_text.count(pattern)
                if count:
                    out_text = out_text.replace(pattern, "")
                    cleanups_applied += count

            # Fix broken URLs (spaces from PDF line-break artifacts)
            out_text, url_fixes = _fix_broken_urls(out_text)
            cleanups_applied += url_fixes

            # Collapse spaced-out capitals (Docling small-caps artifact)
            out_text, caps_fixes = _collapse_spaced_caps(out_text)
            cleanups_applied += caps_fixes

            # Strip repeated "Course Syllabus" page headers
            out_text, page_hdr_fixes = _strip_page_headers(out_text)
            cleanups_applied += page_hdr_fixes

            (output_dir / md_path.name).write_text(out_text, encoding="utf-8")

            if demoted or cleanups_applied:
                result.modified_count += 1
                total_demoted += len(demoted)

            result.log_entries.append(
                {
                    "file": md_path.name,
                    "demoted_count": len(demoted),
                    "demoted_headers": demoted,
                    "cleanups_applied": cleanups_applied,
                }
            )

        result.metrics = {
            "total_demoted": total_demoted,
        }
        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tamubot.ingestion.filters.false_positive INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    filt = FalsePositiveFilter()
    res = filt.apply(in_dir, out_dir)

    print(f"Files processed : {res.input_count}")
    print(f"Files modified  : {res.modified_count}")
    print(f"Headers demoted : {res.metrics['total_demoted']}")

    for entry in res.log_entries:
        if entry["demoted_count"]:
            print(f"\n  {entry['file']} ({entry['demoted_count']} demoted):")
            for d in entry["demoted_headers"]:
                print(f"    [{d['reason']}] {'#' * d['level']} {d['header']}")
