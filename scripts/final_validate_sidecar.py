"""Final validate on Gemini-skipped files after sidecar refresh.

Validates only the 15 Gemini-skipped files (after their enrich JSON headers
were refreshed from the Docling sidecar). Appends fresh entries to
validate_log.csv so the pilot report picks up the new counts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from tamubot.ingestion.validators.llm_validator import validate_directory


def gemini_skipped_stems(dept: str) -> list[str]:
    log_path = Path(f"data/syllabi/{dept}/logs/filter_image_recovery_log.csv")
    skipped: dict[str, str] = {}
    with log_path.open() as fh:
        for row in csv.DictReader(fh):
            stem = (row.get("file") or "").removesuffix(".md")
            status = row.get("status", "")
            if stem:
                skipped[stem] = status
    return sorted(s for s, st in skipped.items() if st == "skipped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument("--version", default="v017")
    args = ap.parse_args()

    base = Path(f"data/syllabi/{args.dept}/silver")
    log_path = Path(f"data/syllabi/{args.dept}/logs/validate_log.csv")
    now = datetime.now().isoformat()

    stems = gemini_skipped_stems(args.dept)
    print(f"Re-validating {len(stems)} Gemini-skipped files ({args.dept}) at {args.version}...\n")

    with log_path.open("r", encoding="utf-8") as fh:
        fieldnames = next(csv.reader(fh))

    new_rows: list[dict] = []
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
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

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for row in new_rows:
            writer.writerow(row)

    grand = sum(r["total_issues"] for r in new_rows)
    clean = sum(1 for r in new_rows if r["total_issues"] == 0)
    print(f"\nAppended {len(new_rows)} rows to {log_path}")
    print(f"Clean files: {clean}/{len(new_rows)}")
    print(f"Grand total findings: {grand}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
