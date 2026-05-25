"""Filter: relocate textbook blocks that Docling placed under the wrong header.

Docling sometimes associates the textbook-cover image with a nearby unrelated
header (typically ``Grading Policy`` or ``Submission of Assignments``) which
places a ``This material Is: Required → ISBN → Authors → Publisher`` block
under the wrong section. This filter detects those mislocated blocks and moves
them under ``Textbook and/or Resource Materials`` (creating that header if
absent).

Behaviour modes (CLI / config flag ``mode``):
  - ``"move"`` (default): cut and relocate the block.
  - ``"flag"``: print a warning per detection but leave content untouched. Use
    this when you suspect false positives on a new department's pilot files.

CLI: python -m tamubot.ingestion.filters.relocate_textbook INPUT_DIR OUTPUT_DIR [--mode move|flag]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tamubot.ingestion.filters.base import FilterResult

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BLOCK_START_RE = re.compile(r"^\s*This material Is\b", re.IGNORECASE)
BLOCK_MARKER_RE = re.compile(r"^\s*(ISBN|Authors|Publisher|Publication Date):", re.IGNORECASE)
FIELD_MARKER_RE = re.compile(
    r"^\s*(ISBN|Authors?|Publisher|Publication Date|Edition|Notes?|Material(s)? Type|Source):?\s*$",
    re.IGNORECASE,
)
TEXTBOOK_HEADER_RE = re.compile(
    r"^\s*textbook(?:\s+and/?or)?(?:\s+resource)?(?:\s+materials?)?\s*:?\s*$", re.IGNORECASE
)

LOOKAHEAD = 15  # max non-blank lines after "This material Is" to look for ISBN/Authors
PROSE_LEN_THRESHOLD = 180  # paragraph longer than this and not a field-value = end of block

# Docling sometimes emits a "Credit Hours: / <value>" block twice back-to-back
# inside the Course Information area. The second copy is verbatim and adds no
# value. Match the canonical layout (blank-separated, value on its own line).
_CREDIT_HOURS_DUP_RE = re.compile(
    r"(Credit Hours:\s*\n+\s*[^\n#]+?\s*\n+)\s*Credit Hours:\s*\n+\s*\1?[^\n#]+?\s*\n",
    re.IGNORECASE,
)
# Tighter form for the actual observed pattern (value lines repeat verbatim):
_CREDIT_HOURS_DUP_STRICT_RE = re.compile(
    r"(Credit Hours:\s*\n+\s*([^\n#]+?)\s*\n+)(?:\s*\n+)?Credit Hours:\s*\n+\s*\2\s*\n",
    re.IGNORECASE,
)

# Gemini/Docling sometimes injects a textbook-cover *description* in place of
# the image. The line adds no information beyond the structured Textbook block
# that already exists. Match conservatively: lines starting with these stems
# that describe a book cover.
_COVER_DESCRIPTION_RE = re.compile(
    r"^\s*(?:An?\s+)?(?:image\s+(?:of|shows|displays)|the\s+image\s+(?:shows|displays))[^\n]*?"
    r"(?:textbook\s+cover|cover\s+of\s+(?:the\s+)?(?:textbook\b|['\"][^'\"]+['\"]))[^\n]*\n",
    re.IGNORECASE | re.MULTILINE,
)


def _find_block_end(lines: list[str], start: int) -> int:
    """Return the index just past the end of a textbook block starting at ``start``.

    A block ends when we hit any of:
      1. The next markdown header (#).
      2. The next ``This material Is`` line (adjacent textbook block).
      3. 3+ consecutive blank lines.
      4. A long-form prose paragraph that doesn't follow a field marker
         (i.e. course-specific text that was bleeding into the misplaced block).
    """
    i = start + 1
    blank_streak = 0
    last_field_idx = start  # treat the "This material Is" header as a field anchor
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if HEADER_RE.match(stripped):
            break
        if i > start and BLOCK_START_RE.match(line):
            # Next textbook block — current block ends here.
            break
        if stripped == "":
            blank_streak += 1
            if blank_streak >= 3:
                return i - blank_streak + 1
            i += 1
            continue
        # Non-blank line.
        if FIELD_MARKER_RE.match(stripped):
            last_field_idx = i
        elif len(stripped) >= PROSE_LEN_THRESHOLD and i - last_field_idx > 3:
            # Long-form prose paragraph not anchored to a recent field marker —
            # this is course-specific text that got mixed in. End the block
            # before this paragraph.
            # Walk back to the most recent blank line so we don't truncate
            # mid-paragraph for the prior textbook content.
            j = i - 1
            while j > start and lines[j].strip() != "":
                j -= 1
            return j + 1 if j > start else i
        blank_streak = 0
        i += 1
    # Trim trailing blank lines
    while i > start + 1 and lines[i - 1].strip() == "":
        i -= 1
    return i


def _has_textbook_markers(block: list[str]) -> bool:
    """Confirm the block is genuinely a textbook block (has ISBN/Authors/Publisher within LOOKAHEAD lines)."""
    non_blank = 0
    for line in block[1 : 1 + LOOKAHEAD * 2]:
        if line.strip():
            non_blank += 1
        if BLOCK_MARKER_RE.match(line):
            return True
        if non_blank > LOOKAHEAD:
            break
    return False


def _section_for_line(headers_by_line: list[tuple[int, str]], line_no: int) -> str | None:
    """Return the most recent header text at or before ``line_no``."""
    cur = None
    for ln, text in headers_by_line:
        if ln <= line_no:
            cur = text
        else:
            break
    return cur


def cleanup_artifacts(text: str) -> tuple[str, dict[str, int]]:
    """Strip Docling layout artifacts that survive earlier silver stages.

    Currently:
      - Duplicated ``Credit Hours: / <value>`` blocks (Docling artifact in the
        Course Information block — verbatim second copy).
      - Standalone Gemini/Docling textbook-cover *description* lines (e.g.
        "An image of the textbook cover for 'Introduction to Algorithms'…").
        These add no value beyond the structured textbook fields.

    Returns ``(new_text, {"credit_hours_dedup": n, "cover_desc_stripped": n})``.
    """
    metrics = {"credit_hours_dedup": 0, "cover_desc_stripped": 0}

    new_text, n = _CREDIT_HOURS_DUP_STRICT_RE.subn(r"\1", text)
    metrics["credit_hours_dedup"] = n

    new_text, n = _COVER_DESCRIPTION_RE.subn("", new_text)
    metrics["cover_desc_stripped"] = n

    # Tidy: collapse 3+ blank lines and trim trailing whitespace per line.
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text, metrics


def relocate_in_text(text: str, mode: str = "move") -> tuple[str, list[dict[str, Any]]]:
    """Detect and (optionally) move misplaced textbook blocks.

    Returns ``(new_text, detections)`` where each detection is a dict with
    keys ``from_header``, ``preview``, ``moved`` (bool).
    """
    lines = text.splitlines()
    detections: list[dict[str, Any]] = []

    # Index headers by line number.
    headers: list[tuple[int, int, str]] = []  # (line_no, level, header_text)
    for idx, line in enumerate(lines):
        m = HEADER_RE.match(line)
        if m:
            headers.append((idx, len(m.group(1)), m.group(2).strip()))

    # Find blocks to move.
    blocks: list[tuple[int, int, str]] = []  # (start, end, parent_header)
    for idx, line in enumerate(lines):
        if BLOCK_START_RE.match(line):
            end = _find_block_end(lines, idx)
            block_lines = lines[idx:end]
            if not _has_textbook_markers(block_lines):
                continue
            # Find parent header — most recent header above this line.
            parent: str | None = None
            for h_line, _lvl, h_text in headers:
                if h_line < idx:
                    parent = h_text
                else:
                    break
            parent_norm = (parent or "").strip().lower().rstrip(":")
            if TEXTBOOK_HEADER_RE.match(parent_norm or ""):
                continue  # already under the right header
            blocks.append((idx, end, parent or "(no header)"))

    if not blocks:
        return text, detections

    if mode == "flag":
        for start, end, parent in blocks:
            preview = lines[start].strip()[:80]
            detections.append({"from_header": parent, "preview": preview, "moved": False})
        return text, detections

    # mode == "move": rebuild text.
    # 1. Mark lines to remove (the block ranges).
    remove_mask = [False] * len(lines)
    cut_blocks: list[list[str]] = []
    for start, end, parent in blocks:
        cut_blocks.append(lines[start:end])
        for i in range(start, end):
            remove_mask[i] = True
        detections.append({"from_header": parent, "preview": lines[start].strip()[:80], "moved": True})

    # 2. Determine where to splice the textbook content.
    # Prefer the existing "Textbook and/or Resource Materials" header; if absent,
    # insert before "Grading Policy" or "Grading" or near the start of the doc.
    textbook_header_line: int | None = None
    textbook_header_level: int = 2
    grading_header_line: int | None = None
    for h_line, lvl, h_text in headers:
        norm = h_text.strip().lower().rstrip(":")
        if TEXTBOOK_HEADER_RE.match(norm) and textbook_header_line is None:
            textbook_header_line = h_line
            textbook_header_level = lvl
        if grading_header_line is None and norm in {"grading policy", "grading"}:
            grading_header_line = h_line

    # Build the relocated payload (cut blocks joined with blank lines).
    payload: list[str] = []
    for block in cut_blocks:
        # Strip leading/trailing blank lines from the block.
        while block and block[0].strip() == "":
            block = block[1:]
        while block and block[-1].strip() == "":
            block = block[:-1]
        payload.extend(block)
        payload.append("")  # one blank separator between successive blocks

    # 3. Produce the new line list.
    survivors = [ln for ln, rm in zip(lines, remove_mask) if not rm]

    # Recompute header positions in survivors.
    def _find_header_idx(predicate) -> int | None:
        for i, ln in enumerate(survivors):
            m = HEADER_RE.match(ln)
            if m and predicate(m.group(2).strip().lower().rstrip(":")):
                return i
        return None

    tb_idx = _find_header_idx(lambda t: bool(TEXTBOOK_HEADER_RE.match(t)))
    gp_idx = _find_header_idx(lambda t: t in {"grading policy", "grading"})

    insert_idx: int
    if tb_idx is not None:
        # Insert payload immediately after the header line (and blank lines).
        j = tb_idx + 1
        while j < len(survivors) and survivors[j].strip() == "":
            j += 1
        insert_idx = j
        new_survivors = survivors[:insert_idx] + payload + [""] + survivors[insert_idx:]
    elif gp_idx is not None:
        # Insert a new header + payload immediately before grading policy.
        header_block = [
            "",
            f"{'#' * textbook_header_level} Textbook and/or Resource Materials",
            "",
        ]
        new_survivors = survivors[:gp_idx] + header_block + payload + survivors[gp_idx:]
    else:
        # Last resort: append at end of file.
        header_block = ["", f"{'#' * textbook_header_level} Textbook and/or Resource Materials", ""]
        new_survivors = survivors + header_block + payload

    new_text = "\n".join(new_survivors)
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, detections


class RelocateTextbookFilter:
    """Relocate textbook blocks Docling placed under the wrong header.

    Optional pipeline step. NOT yet wired into ``ALL_STEPS`` — heuristic needs
    validation against a wider corpus first. Use the standalone CLI or invoke
    explicitly via ``--step relocate_textbook`` after enabling it manually.
    """

    name: str = "relocate_textbook"

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

        mode = config.get("mode", "move")
        pattern = config.get("file_pattern", "*.md")
        md_files = sorted(input_dir.glob(pattern))
        if limit := config.get("limit"):
            md_files = md_files[:limit]

        result = FilterResult()
        result.input_count = len(md_files)
        total_moved = 0
        total_flagged = 0

        total_credit_dedup = 0
        total_cover_stripped = 0
        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            new_text, detections = relocate_in_text(text, mode=mode)
            new_text, cleanup_metrics = cleanup_artifacts(new_text)
            (output_dir / md_path.name).write_text(new_text, encoding="utf-8")

            moved = sum(1 for d in detections if d["moved"])
            flagged = sum(1 for d in detections if not d["moved"])
            total_moved += moved
            total_flagged += flagged
            total_credit_dedup += cleanup_metrics["credit_hours_dedup"]
            total_cover_stripped += cleanup_metrics["cover_desc_stripped"]
            if moved or cleanup_metrics["credit_hours_dedup"] or cleanup_metrics["cover_desc_stripped"]:
                result.modified_count += 1

            result.log_entries.append(
                {
                    "file": md_path.name,
                    "mode": mode,
                    "blocks_moved": moved,
                    "blocks_flagged": flagged,
                    "detections": detections,
                    "credit_hours_dedup": cleanup_metrics["credit_hours_dedup"],
                    "cover_desc_stripped": cleanup_metrics["cover_desc_stripped"],
                }
            )

        result.metrics = {
            "total_moved": total_moved,
            "total_flagged": total_flagged,
            "mode": mode,
            "credit_hours_dedup": total_credit_dedup,
            "cover_desc_stripped": total_cover_stripped,
        }
        return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Relocate misplaced textbook blocks")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", default="move", choices=["move", "flag"])
    parser.add_argument("--file-pattern", default="*.md")
    args = parser.parse_args()

    filt = RelocateTextbookFilter()
    res = filt.apply(
        Path(args.input_dir),
        Path(args.output_dir),
        {"mode": args.mode, "file_pattern": args.file_pattern},
    )

    print(f"Files processed : {res.input_count}")
    print(f"Files modified  : {res.modified_count}")
    print(f"Blocks moved    : {res.metrics['total_moved']}")
    print(f"Blocks flagged  : {res.metrics['total_flagged']}")
    for entry in res.log_entries:
        if entry["blocks_moved"] or entry["blocks_flagged"]:
            print(f"\n  {entry['file']}:")
            for d in entry["detections"]:
                tag = "MOVED" if d["moved"] else "FLAG "
                print(f"    [{tag}] from='{d['from_header']}' | {d['preview']}")


if __name__ == "__main__":
    main()
