#!/usr/bin/env python3
"""Fix mechanical errors in Fall 2026 ISEN processed syllabi.

Thin wrapper around ``tamubot.ingestion.filters.post_convert_cleanup`` that
preserves the original CLI behaviour: in-place rewrites of
``data/syllabi/silver/04_hierarchy/202641_ISEN_*.md`` (excluding 645_601 which
was manually fixed).

All cleanup logic now lives in the filter — see
``src/tamubot/ingestion/filters/post_convert_cleanup.py``.
"""

from collections import Counter
from pathlib import Path

from tamubot.ingestion.filters.post_convert_cleanup import fix_text

HIERARCHY_DIR = Path("data/syllabi/silver/04_hierarchy")


def fix_file(path: Path) -> dict[str, int]:
    """Apply all post-convert fixes to a single file in place."""
    text = path.read_text(encoding="utf-8")
    new_text, counts = fix_text(text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return counts


def main() -> None:
    files = sorted(HIERARCHY_DIR.glob("202641_ISEN_*.md"))
    # Skip 645_601 which was already manually fixed
    files = [f for f in files if "645_601" not in f.name]

    print(f"Processing {len(files)} Fall 2026 ISEN files (excluding 645_601 already fixed)\n")

    total_counts: Counter[str] = Counter()
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
