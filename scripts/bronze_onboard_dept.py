"""Onboard a department to the v5 Dagster pipeline through bronze.

Discovers stems for the department, registers them as syllabus_stem dynamic
partitions, then materializes raw_pdf + bronze_assets for each. Docling
converter is loaded once and reused across stems.

Usage:
    python scripts/bronze_onboard_dept.py --dept STAT
    python scripts/bronze_onboard_dept.py --dept ECEN --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DAGSTER_HOME", str(Path(__file__).resolve().parent.parent / ".dagster_home"))

from dagster import DagsterInstance, materialize

from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.assets.bronze import bronze_assets
from tamubot.ingestion.pipeline_v5.assets.raw import raw_pdf
from tamubot.ingestion.pipeline_v5.resources import DoclingConverterResource
from tamubot.ingestion.pipeline_v5.source_discovery import discover_source_pdfs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dept", required=True, help="Department code, e.g. STAT")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stems whose bronze markdown already exists.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of stems (debug).")
    args = parser.parse_args()

    dept = args.dept.upper()
    stems = list(discover_source_pdfs(dept).keys())
    if args.limit:
        stems = stems[: args.limit]

    if not stems:
        print(f"No stems discovered for {dept}.")
        return 1

    print(f"Discovered {len(stems)} stems for {dept}.")

    instance = DagsterInstance.get()
    existing = set(instance.get_dynamic_partitions("syllabus_stem"))
    to_register = [s for s in stems if s not in existing]
    if to_register:
        instance.add_dynamic_partitions("syllabus_stem", to_register)
        print(f"Registered {len(to_register)} new partitions ({len(stems) - len(to_register)} already present).")
    else:
        print("All partitions already registered.")

    resources = {"docling": DoclingConverterResource()}

    succeeded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    t0 = time.time()

    for i, stem in enumerate(stems, 1):
        if args.skip_existing and paths.bronze_md_path(stem).exists():
            skipped += 1
            print(f"[{i}/{len(stems)}] {stem}: skip (bronze exists)")
            continue

        t_stem = time.time()
        try:
            result = materialize(
                [raw_pdf, bronze_assets],
                partition_key=stem,
                instance=instance,
                resources=resources,
                raise_on_error=False,
            )
            elapsed = time.time() - t_stem
            if result.success:
                succeeded += 1
                print(f"[{i}/{len(stems)}] {stem}: ok ({elapsed:.1f}s)")
            else:
                failed.append((stem, "materialize returned success=False"))
                print(f"[{i}/{len(stems)}] {stem}: FAIL ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.time() - t_stem
            failed.append((stem, repr(exc)))
            print(f"[{i}/{len(stems)}] {stem}: ERROR ({elapsed:.1f}s) {exc!r}")

    total = time.time() - t0
    print()
    print(f"Done in {total / 60:.1f} min — ok={succeeded} skipped={skipped} failed={len(failed)}")
    if failed:
        print("Failures:")
        for stem, msg in failed:
            print(f"  {stem}: {msg}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
