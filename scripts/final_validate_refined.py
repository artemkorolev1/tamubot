"""Final validate pass on the 9 refined files after header refresh.

Subagents validated their files BEFORE the header-refresh script ran, so the
stored validation JSONs reference a stale enrich state. This script re-runs
validate_directory on just those 9 files so the report reflects the current
markdown + enrich state.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from tamubot.ingestion.validators.llm_validator import validate_directory

REFINED_STEMS = [
    "202611_CSCE_636_600_42745_HP_v013",
    "202611_CSCE_635_600_46646_v013",
    "202611_CSCE_632_600_54784_v013",
    "202611_CSCE_633_600_12367_HP_v013",
    "202611_CSCE_638_600_54988_v013",
    "202611_CSCE_650_600_55689_HP_v013",
    "202611_CSCE_633_700_44010_v013",
    "202611_CSCE_648_600_60319_v013",
    "202611_CSCE_665_600_30874_v013",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument("--version", default="v016")
    args = ap.parse_args()

    base = Path(f"data/syllabi/{args.dept}/silver")
    log_path = Path(f"data/syllabi/{args.dept}/logs/validate_log.csv")
    now = datetime.now().isoformat()

    # Build a temp-style invocation: run validate_directory once per file
    # via file_pattern to avoid wasting calls on the other 21 files.
    print(f"Final re-validate of {len(REFINED_STEMS)} refined files ({args.dept})...\n")

    with log_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames

    new_rows: list[dict] = []
    for i, stem in enumerate(REFINED_STEMS, 1):
        print(f"[{i}/{len(REFINED_STEMS)}] {stem}")
        results = validate_directory(
            input_dir=base / "04_hierarchy",
            metadata_dir=base / "05_enrich",
            output_dir=base / "05_validate",
            version_label=args.version,
            file_pattern=f"{stem}.md",
        )
        for r in results:
            counts = r.issue_counts
            total = r.total_issues
            row = {fn: "" for fn in fieldnames}
            row.update(
                {
                    "version": args.version,
                    "file": r.file_stem,
                    "total_issues": total,
                    **counts,
                    "timing_s": round(r.timing_s, 2),
                    "timestamp": now,
                }
            )
            new_rows.append(row)
            summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0) or "clean"
            print(f"   {summary}  (total={total}, {r.timing_s:.1f}s)")

    # Append to log
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for row in new_rows:
            writer.writerow(row)

    print(f"\nAppended {len(new_rows)} rows to {log_path} (version={args.version})")
    grand = sum(r["total_issues"] for r in new_rows)
    print(f"Refined-files grand total: {grand}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
