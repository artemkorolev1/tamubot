#!/usr/bin/env python3
"""Fix mechanical errors in Fall 2026 ISEN processed syllabi.

Operates on data/syllabi/silver/04_hierarchy/202641_ISEN_*.md files.
Applies these fixes:
  1. Label-value line joins (Email:\n\nvalue → Email: value)
  2. Duplicate consecutive headers removal
  3. Duplicate Credit Hours removal
  4. Image description artifact removal
  5. Double-hyphen list fix (- -text → - text)
  6. Leftover <!-- image --> marker removal
  7. Broken table fragment removal (|    | / |----|)
  8. Excessive blank line cleanup
"""

import re
from collections import Counter
from pathlib import Path

HIERARCHY_DIR = Path("data/syllabi/silver/04_hierarchy")

# Labels that Docling splits across lines
LABEL_RE = re.compile(
    r"^(Email|Phone|Office Location|Office Hours|Credit Hours|"
    r"Meeting Location|Meeting Days|Meeting Type|Start Time|End Time|"
    r"Start Date|End Date|Authors|ISBN|Publisher|Publication Date|"
    r"Webpage|URL for Resource|Preferred Contact Method|"
    r"Prerequisite/Corequisite\(s\)):\s*$"
)

# Image description artifacts from Gemini
IMG_DESC_RE = re.compile(
    r"^(A headshot of .+|"
    r"An image of the (book cover|textbook cover) .+|"
    r"The Texas A&M University logo .+)$"
)

# Heading pattern
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Broken table fragments
BROKEN_TABLE_RE = re.compile(r"^\|\s*\|$|^\|-+\|$")


def fix_file(path: Path) -> dict[str, int]:
    """Apply all fixes to a single file. Returns per-fix counts."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    counts = Counter()

    # === Pass 1: Label-value joins ===
    joined = []
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
                # Don't join if the value is a header or another label
                if not HEADER_RE.match(value) and not LABEL_RE.match(value):
                    label = line.strip()
                    joined.append(f"{label} {value}")
                    counts["label_value_joined"] += 1
                    i = j + 1
                    continue
        joined.append(line)
        i += 1
    lines = joined

    # === Pass 2: Remove duplicate consecutive headers ===
    deduped = []
    prev_header_text = None
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

    # === Pass 3: Remove duplicate Credit Hours lines ===
    # Keep only the first "Credit Hours: N" line
    credit_seen = False
    filtered = []
    for line in lines:
        if re.match(r"^Credit Hours:?\s+\d", line.strip()):
            if credit_seen:
                counts["dup_credit_removed"] += 1
                continue
            credit_seen = True
        filtered.append(line)
    lines = filtered

    # === Pass 4: Remove image description artifacts ===
    filtered = []
    for line in lines:
        if IMG_DESC_RE.match(line.strip()):
            counts["img_desc_removed"] += 1
            continue
        filtered.append(line)
    lines = filtered

    # === Pass 5: Fix double-hyphen list items ===
    fixed = []
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
    result_lines = []
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

    new_text = "\n".join(result_lines)
    if not new_text.endswith("\n"):
        new_text += "\n"

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")

    return dict(counts)


def main():
    files = sorted(HIERARCHY_DIR.glob("202641_ISEN_*.md"))
    # Skip 645_601 which was already manually fixed
    files = [f for f in files if "645_601" not in f.name]

    print(f"Processing {len(files)} Fall 2026 ISEN files (excluding 645_601 already fixed)\n")

    total_counts = Counter()
    files_changed = 0

    for f in files:
        counts = fix_file(f)
        if counts:
            files_changed += 1
            total_fixes = sum(counts.values())
            print(f"  {f.name}: {total_fixes} fixes — {dict(counts)}")
            for k, v in counts.items():
                total_counts[k] += v

    print(f"\n{'=' * 60}")
    print(f"Files changed: {files_changed}/{len(files)}")
    print(f"Total fixes: {sum(total_counts.values())}")
    print("\nBy type:")
    for fix_type, count in total_counts.most_common():
        print(f"  {fix_type}: {count}")


if __name__ == "__main__":
    main()
