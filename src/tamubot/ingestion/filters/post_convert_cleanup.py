"""Filter: deterministic post-Docling-conversion cleanups.

Runs immediately after the ``convert`` step (before ``image_recovery``) to fix
mechanical artifacts the Docling converter introduces on TAMU syllabus PDFs.

Cleanups (ported from ``scripts/fix_fall26_isen.py``):
    1. Label-value line joins  (``Email:\\n\\nvalue`` → ``Email: value``)
    2. Duplicate consecutive headers removal
    3. Duplicate ``Credit Hours`` lines removal
    4. Image description artifact removal (headshot/textbook cover / TAMU logo)
    5. Double-hyphen list-item fix  (``- -text`` → ``- text``)
    6. Leftover ``<!-- image -->`` marker removal
    7. Broken table-fragment removal (``|    |`` / ``|----|``)
    8. Excessive blank-line collapsing (3+ → 2)

Additionally (new in v4):
    9. ``Meeting Days`` back-fill — populates an empty Meeting Days field by
       parsing nearby Meeting Times / Meeting Location lines. Sets
       ``Asynchronous`` for online sections, parses ``MWF``-style weekday tokens
       from a Meeting Times line, and clears values that collided with the
       next label (``Meeting Days: Start Time: N/A``).

CLI:  python -m tamubot.ingestion.filters.post_convert_cleanup INPUT_DIR OUTPUT_DIR
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from tamubot.ingestion.filters.base import FilterResult

# ── Regexes ──────────────────────────────────────────────────────────────────

# Labels that Docling commonly splits across lines.
_LABEL_NAMES = (
    r"(?:Email|Phone|Office Location|Office Hours|Credit Hours|"
    r"Meeting Location|Meeting Days|Meeting Type|Meeting Times|"
    r"Start Time|End Time|Start Date|End Date|Authors|ISBN|Publisher|"
    r"Publication Date|Webpage|URL for Resource|Preferred Contact Method|"
    r"Prerequisite/Corequisite\(s\))"
)
# Bare label form (no value):  "Email:"
LABEL_RE = re.compile(rf"^{_LABEL_NAMES}:\s*$")
# Either bare OR populated label form:  "Email:" / "Email: foo"
LABEL_OR_VALUE_RE = re.compile(rf"^{_LABEL_NAMES}:")

# Image-description artifacts from Gemini image_recovery.
IMG_DESC_RE = re.compile(
    r"^(A headshot of .+|"
    r"An image of the (book cover|textbook cover) .+|"
    r"The Texas A&M University logo .+)$"
)

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")

BROKEN_TABLE_RE = re.compile(r"^\|\s*\|$|^\|-+\|$")

# Meeting-Days back-fill helpers
_LABEL_AS_VALUE_RE = re.compile(
    r"^Meeting Days:\s*(Start Time|End Time|Meeting Location|Meeting Type|Start Date|End Date):"
)
_WEEKDAY_TOKEN_RE = re.compile(r"\b((?:M|T|W|R|F|Sa|Su|TR|MWF|MW|TTh|TR)+)\b\s+\d{1,2}:\d{2}")
_VALID_WEEKDAY_TOKEN_RE = re.compile(r"^(?:M|T|W|R|F|MWF|MW|TR|TTh)$")


# ── File-level fix ───────────────────────────────────────────────────────────


def fix_text(text: str) -> tuple[str, dict[str, int]]:
    """Apply all post-convert cleanups to ``text``. Returns ``(new_text, counts)``."""
    lines = text.splitlines()
    counts: Counter[str] = Counter()

    # === Pass 1: Label-value joins ===
    joined: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if LABEL_RE.match(line.strip()):
            # Look ahead past blank lines for the value
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and lines[j].strip():
                value = lines[j].strip()
                # Don't join if the value is itself a header or another label
                # (bare or populated — e.g. don't pull "Start Time: N/A" up as
                # the value for an empty "Meeting Days:").
                if (
                    not HEADER_RE.match(value)
                    and not LABEL_RE.match(value)
                    and not LABEL_OR_VALUE_RE.match(value)
                ):
                    label = line.strip()
                    joined.append(f"{label} {value}")
                    counts["label_value_joined"] += 1
                    i = j + 1
                    continue
        joined.append(line)
        i += 1
    lines = joined

    # === Pass 2: Remove duplicate consecutive headers ===
    deduped: list[str] = []
    prev_header_text: str | None = None
    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            header_text = m.group(2).strip()
            if header_text == prev_header_text:
                counts["dup_header_removed"] += 1
                continue
            prev_header_text = header_text
        elif line.strip():
            prev_header_text = None
        deduped.append(line)
    lines = deduped

    # === Pass 3: Remove duplicate Credit Hours lines (keep first) ===
    credit_seen = False
    filtered: list[str] = []
    for line in lines:
        if re.match(r"^Credit Hours:?\s+\d", line.strip()):
            if credit_seen:
                counts["dup_credit_removed"] += 1
                continue
            credit_seen = True
        filtered.append(line)
    lines = filtered

    # === Pass 4: Remove image-description artifacts ===
    filtered = []
    for line in lines:
        if IMG_DESC_RE.match(line.strip()):
            counts["img_desc_removed"] += 1
            continue
        filtered.append(line)
    lines = filtered

    # === Pass 5: Fix double-hyphen list items ===
    fixed: list[str] = []
    for line in lines:
        if re.match(r"^- -\w", line):
            fixed.append(line.replace("- -", "- ", 1))
            counts["double_hyphen_fixed"] += 1
        else:
            fixed.append(line)
    lines = fixed

    # === Pass 6: Remove <!-- image --> markers ===
    filtered = []
    for line in lines:
        if re.match(r"^\s*<!--\s*image\s*-->\s*$", line):
            counts["img_marker_removed"] += 1
            continue
        filtered.append(line)
    lines = filtered

    # === Pass 7: Remove broken table fragments ===
    filtered = []
    for line in lines:
        if BROKEN_TABLE_RE.match(line.strip()):
            counts["broken_table_frag_removed"] += 1
            continue
        filtered.append(line)
    lines = filtered

    # === Pass 8: Collapse excessive blank lines (3+ → 2) ===
    result_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
            else:
                counts["excess_blank_removed"] += 1
        else:
            blank_count = 0
            result_lines.append(line)
    lines = result_lines

    # === Pass 9: Meeting Days back-fill / normalization ===
    lines, meeting_counts = _backfill_meeting_days(lines)
    for k, v in meeting_counts.items():
        counts[k] += v

    new_text = "\n".join(lines)
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, dict(counts)


# ── Meeting-Days back-fill ───────────────────────────────────────────────────


def _backfill_meeting_days(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    """Normalize ``Meeting Days:`` lines.

    Rules (priority order):
      (a) If the value collided with another label (``Meeting Days: Start Time:``),
          clear the value.
      (b) If empty and a nearby Meeting Times / Meeting Location line indicates an
          asynchronous-online section (``Online`` location, or N/A Start Time),
          set to ``Asynchronous``.
      (c) Else if empty and a ``MWF 3:00PM`` style token appears earlier under
          the same ``## Course Information`` block, parse the weekday token and
          back-fill.
    """
    counts: Counter[str] = Counter()

    # Find each Meeting Days line and decide what to do.
    n = len(lines)
    out = list(lines)

    # Build quick lookup of indices for relevant labels per Meeting-Days line.
    # We look in a small window around each Meeting Days line for siblings.
    WINDOW = 20

    for idx, line in enumerate(out):
        stripped = line.strip()

        # (a) Label-as-value collision: "Meeting Days: Start Time: N/A"
        if _LABEL_AS_VALUE_RE.match(stripped):
            out[idx] = "Meeting Days:"
            counts["meeting_days_label_collision"] += 1
            stripped = "Meeting Days:"

        # Only act on the empty form now.
        if stripped != "Meeting Days:":
            continue

        # Gather context lines in window.
        start = max(0, idx - WINDOW)
        end = min(n, idx + WINDOW + 1)
        window = out[start:end]

        # (b) Asynchronous-online detection.
        is_online = False
        is_async = False
        for w in window:
            ws = w.strip()
            if ws.lower().startswith("meeting location:"):
                tail = ws.split(":", 1)[1].strip().lower()
                if "online" in tail:
                    is_online = True
            # Standalone "ONLINE ONLINE"-style location line.
            if "online" in ws.lower() and "meeting" not in ws.lower() and len(ws) < 40:
                if re.fullmatch(r"(?i)\s*online(\s+online)*\s*", ws):
                    is_online = True
            if ws.lower().startswith("start time:"):
                tail = ws.split(":", 1)[1].strip().lower()
                if tail in ("", "n/a", "na", "none"):
                    is_async = True

        if is_online and is_async:
            out[idx] = "Meeting Days: Asynchronous"
            counts["meeting_days_async_filled"] += 1
            continue

        # (c) Parse "MWF 3:00PM" style token from a Meeting Times line.
        weekday: str | None = None
        for w in window:
            ws = w.strip()
            # Skip non-time-bearing lines.
            if not re.search(r"\d{1,2}:\d{2}", ws):
                continue
            m = _WEEKDAY_TOKEN_RE.search(ws)
            if m:
                cand = m.group(1)
                if _VALID_WEEKDAY_TOKEN_RE.match(cand):
                    weekday = cand
                    break
        if weekday:
            out[idx] = f"Meeting Days: {weekday}"
            counts["meeting_days_parsed"] += 1

    return out, dict(counts)


# ── Filter class ─────────────────────────────────────────────────────────────


class PostConvertCleanupFilter:
    """Deterministic post-Docling-conversion cleanup filter."""

    name: str = "post_convert"

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

        pattern = config.get("file_pattern", "*.md")
        md_files = sorted(input_dir.glob(pattern))
        if (limit := config.get("limit")):
            md_files = md_files[:limit]

        result = FilterResult()
        result.input_count = len(md_files)

        total_counts: Counter[str] = Counter()

        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            new_text, counts = fix_text(text)
            (output_dir / md_path.name).write_text(new_text, encoding="utf-8")

            fixes_applied = sum(counts.values())
            if fixes_applied:
                result.modified_count += 1

            for k, v in counts.items():
                total_counts[k] += v

            result.log_entries.append(
                {
                    "file": md_path.name,
                    "fixes_applied": fixes_applied,
                    "by_type": dict(counts),
                }
            )

        result.metrics = {
            "total_fixes": sum(total_counts.values()),
            "by_type": dict(total_counts),
        }
        return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tamubot.ingestion.filters.post_convert_cleanup INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    filt = PostConvertCleanupFilter()
    res = filt.apply(in_dir, out_dir)

    print(f"Files processed : {res.input_count}")
    print(f"Files modified  : {res.modified_count}")
    print(f"Total fixes     : {res.metrics['total_fixes']}")
    if res.metrics["by_type"]:
        print("\nBy type:")
        for k, v in sorted(res.metrics["by_type"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<32} {v:>5}")


if __name__ == "__main__":
    main()
