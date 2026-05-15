"""Sync validate_log.csv with the latest *_validation.json counts.

After manual refinement passes (where subagents called validate_directory
directly), the per-file JSONs are current but validate_log.csv is stale.
The pilot report reads counts from the log, not the JSONs.

This script reads every *_validation.json under <dept>/silver/05_validate/,
computes counts per category, and appends one row per file to validate_log.csv
with a fresh version label (latest_label) and timestamp. The report then
picks up the new entries (last-wins iteration in build_csce_pilot_report).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument("--version", default="v015")
    args = ap.parse_args()

    base = Path(f"data/syllabi/{args.dept}")
    validate_dir = base / "silver" / "05_validate"
    log_path = base / "logs" / "validate_log.csv"

    if not validate_dir.is_dir():
        print(f"ERROR: {validate_dir} not found", file=sys.stderr)
        return 1
    if not log_path.exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        return 1

    # Read existing fieldnames to preserve schema
    with log_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        existing_rows = list(reader)

    if not fieldnames:
        print(f"ERROR: {log_path} has no header", file=sys.stderr)
        return 1

    now = datetime.now().isoformat()
    new_rows: list[dict] = []
    cats = (
        "content_preservation",
        "strip_completeness",
        "structural_coherence",
        "metadata_accuracy",
    )

    for vjson in sorted(validate_dir.glob("*_validation.json")):
        stem = vjson.stem.removesuffix("_validation")
        try:
            data = json.loads(vjson.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {stem}: {e}")
            continue
        findings = data.get("findings", {})
        counts = {cat: len(findings.get(cat, [])) for cat in cats}
        total = sum(counts.values())

        row = {fn: "" for fn in fieldnames}
        row.update(
            {
                "version": args.version,
                "file": stem,
                "total_issues": total,
                **counts,
                "timing_s": data.get("timing_s", ""),
                "timestamp": now,
            }
        )
        new_rows.append(row)

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for row in new_rows:
            writer.writerow(row)

    print(f"Appended {len(new_rows)} rows to {log_path} with version={args.version}")
    by_total = sorted(new_rows, key=lambda r: -r["total_issues"])
    print("\nTop 10 by findings:")
    for r in by_total[:10]:
        print(
            f"  {r['file']}: total={r['total_issues']} ({r['content_preservation']}/{r['strip_completeness']}/{r['structural_coherence']}/{r['metadata_accuracy']})"
        )
    print(f"\nFiles with 0 findings: {sum(1 for r in new_rows if r['total_issues'] == 0)}/{len(new_rows)}")
    grand_total = sum(r["total_issues"] for r in new_rows)
    print(f"Grand total findings: {grand_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
