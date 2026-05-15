"""Filter: reconstruct heading hierarchy for flat-header markdown.

If Docling output has only one distinct heading level (e.g. all ``##``), this
filter infers a richer hierarchy from numbering patterns and known syllabus
section structure.

CLI: python -m tamubot.ingestion.filters.hierarchy input_dir/ output_dir/
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from tamubot.ingestion.filters.base import FilterResult

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Numbering patterns → implied nesting depth (0-indexed offset from base)
_NUM_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"^\d+\.\d+\.\d+"), 2),  # 1.1.1  → depth 2
    (re.compile(r"^\d+\.\d+"), 1),  # 1.1    → depth 1
    (re.compile(r"^\d+\."), 0),  # 1.     → depth 0
]

# ── Known syllabus section hierarchy ─────────────────────────────────────────
# Lowercased, colon-stripped header text → target level

_LEVEL1_HEADERS: frozenset[str] = frozenset()  # course title detected separately

_LEVEL2_HEADERS: frozenset[str] = frozenset(
    s.lower()
    for s in [
        "Course Description",
        "Course Information",
        "Course Overview",
        "Catalog Description",
        "Prerequisites",
        "Course Prerequisites",
        "Corequisites",
        "Grading",
        "Grading Policy",
        "Schedule",
        "Course Schedule",
        "Tentative Schedule",
        "Weekly Schedule",
        "Instructor Information",
        "Instructor",
        "Instructor Details",
        "Contact Information",
        "Course Objectives",
        "Learning Outcomes",
        "Course Learning Outcomes",
        "Required Materials",
        "Textbook",
        "Textbooks",
        "Required Textbooks",
        "Textbook and/or Resource Materials",
        "Course Policies",
        "Attendance",
        "Attendance Policy",
        "Late Work Policy",
        "Course Specific Late Work Policy",
        "Academic Integrity",
        "University Policies",
        "Americans with Disabilities Act (ADA) Policy",
        "Important Dates",
        "Assignments",
        "Exams",
        "Examinations",
        "Homework",
        "Special Course Designation",
        "Additional Course Information",
        "Additional Course Details",
    ]
)

_LEVEL3_HEADERS: frozenset[str] = frozenset(
    s.lower()
    for s in [
        "Grading Scale",
        "Grading Breakdown",
        "Late Policy",
        "Late Work",
        "Late Submission Policy",
        "Makeup Exams",
        "Makeup Work",
        "Makeup Policy",
        "Office Hours",
        "Teaching Assistant",
        "Teaching Assistants",
        "TA Information",
        "Midterm",
        "Midterm Exam",
        "Final Exam",
        "Final Project",
        "Participation",
        "Class Participation",
        "Extra Credit",
        "Regrade Policy",
        "Regrading Policy",
        "Quizzes",
        "Projects",
        "Standard Letter Grading Scale",
    ]
)

# Subsection patterns inside Course Schedule / multi-week breakdowns.
# Used only when the input is flat-H1 (bronze had no Docling hierarchy info).
_SCHEDULE_SUBSECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"^Week\s+\d+\b", re.IGNORECASE),  # "Week 1 - ..."
    re.compile(r"^Part\s+[IVX]+\b", re.IGNORECASE),  # "Part I:", "Part II:"
    re.compile(r"^Module\s+\d+\b", re.IGNORECASE),  # "Module 3 - ..."
    re.compile(r"^Unit\s+\d+\b", re.IGNORECASE),  # "Unit 4 - ..."
    re.compile(r"^Lecture\s+\d+\b", re.IGNORECASE),  # "Lecture 5: ..."
]


def _known_level(header_text: str) -> int | None:
    """Return target heading level (2 or 3) from known-section map, or None."""
    norm = header_text.strip().rstrip(":").lower()
    if norm in _LEVEL2_HEADERS:
        return 2
    if norm in _LEVEL3_HEADERS:
        return 3
    return None


def _infer_level_from_numbering(header_text: str, base_level: int) -> int | None:
    """Return target level inferred from leading numbering pattern."""
    stripped = header_text.strip()
    for pat, depth in _NUM_PATTERNS:
        if pat.match(stripped):
            return base_level + depth
    return None


def _is_schedule_subsection(header_text: str) -> bool:
    """Detect Week/Part/Module/Unit/Lecture subsection patterns inside a schedule."""
    stripped = header_text.strip()
    return any(pat.match(stripped) for pat in _SCHEDULE_SUBSECTION_PATTERNS)


def _estimate_tokens(text: str) -> int:
    return max(0, round(len(text) / 4))


class HierarchyFilter:
    """Reconstruct heading hierarchy for flat-header markdown."""

    name: str = "hierarchy"

    def apply(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any] | None = None,
        report_path: Path | None = None,
    ) -> FilterResult:
        config = config or {}
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if report_path is None:
            from tamubot.ingestion.report_writer import get_report

            report_path = get_report()

        result = FilterResult()
        pattern = config.get("file_pattern", "*.md")
        md_files = sorted(input_dir.glob(pattern))
        if limit := config.get("limit"):
            md_files = md_files[:limit]
        result.input_count = len(md_files)

        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            lines = text.splitlines()

            # Collect all header levels present
            headers: list[tuple[int, int, str]] = []  # (line_idx, level, text)
            for i, line in enumerate(lines):
                m = _HEADER_RE.match(line)
                if m:
                    headers.append((i, len(m.group(1)), m.group(2).strip()))

            distinct_levels = set(h[1] for h in headers)
            level_dist_before = {lv: 0 for lv in sorted(distinct_levels)}
            for _, lv, _ in headers:
                level_dist_before[lv] = level_dist_before.get(lv, 0) + 1

            corrected = False

            if not headers:
                # No headers — pass-through
                shutil.copy2(md_path, output_dir / md_path.name)
            else:
                # Apply known-section and numbering correction to all files.
                # For flat files (one distinct level), use that as base_level.
                # For multi-level files, use 2 (standard ## base).
                base_level = next(iter(distinct_levels)) if len(distinct_levels) == 1 else 2

                # Flat-H1 input means Docling produced no hierarchy info (bronze
                # has only H1 headers). Treat the first H1 as the doc title and
                # default-promote the rest to H2, demoting known subsections to H3.
                is_flat_h1 = distinct_levels == {1}
                first_header_idx = headers[0][0] if headers else -1

                out_lines = list(lines)
                for line_idx, _old_level, header_text in headers:
                    # Priority order:
                    #   1. known section map (L2/L3 by name)
                    #   2. numbering pattern (1.1.1 → +depth)
                    #   3. (flat-H1 only) schedule subsection pattern → L3
                    #   4. (flat-H1 only) default to L2 (or keep L1 for doc title)
                    new_level = _known_level(header_text)
                    if new_level is None:
                        new_level = _infer_level_from_numbering(header_text, base_level)
                    if new_level is None and is_flat_h1:
                        if line_idx == first_header_idx:
                            new_level = 1  # preserve document title
                        elif _is_schedule_subsection(header_text):
                            new_level = 3
                        else:
                            new_level = 2
                    if new_level is not None and new_level != _old_level:
                        out_lines[line_idx] = f"{'#' * new_level} {header_text}"
                        corrected = True
                    # else: keep original line unchanged

                out_text = "\n".join(out_lines)
                (output_dir / md_path.name).write_text(out_text, encoding="utf-8")

            if corrected:
                result.modified_count += 1

            # Compute level distribution after
            after_text = (output_dir / md_path.name).read_text(encoding="utf-8")
            level_dist_after: dict[int, int] = {}
            for line in after_text.splitlines():
                m = _HEADER_RE.match(line)
                if m:
                    lv = len(m.group(1))
                    level_dist_after[lv] = level_dist_after.get(lv, 0) + 1

            result.log_entries.append(
                {
                    "file": md_path.name,
                    "corrected": corrected,
                    "levels_before": level_dist_before,
                    "levels_after": level_dist_after,
                    "header_count": len(headers),
                }
            )

            if report_path:
                from tamubot.ingestion.report_writer import update_hierarchy

                update_hierarchy(
                    report_path,
                    md_path.stem,
                    corrected,
                    level_dist_before,
                    level_dist_after,
                )

        result.metrics = {
            "files_corrected": result.modified_count,
            "files_passthrough": result.input_count - result.modified_count,
        }
        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tamubot.ingestion.filters.hierarchy INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    filt = HierarchyFilter()
    res = filt.apply(in_dir, out_dir)

    print(f"Files processed   : {res.input_count}")
    print(f"Files corrected   : {res.metrics['files_corrected']}")
    print(f"Files pass-through: {res.metrics['files_passthrough']}")

    for entry in res.log_entries:
        status = "CORRECTED" if entry["corrected"] else "pass-through"
        print(f"\n  {entry['file']} [{status}]")
        print(f"    before: {entry['levels_before']}")
        print(f"    after:  {entry['levels_after']}")
