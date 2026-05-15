"""Retry generate_summary_statements only on chunk files with empty statements.

Reads each chunk JSON, re-runs the LLM call (with the new retry-on-empty
logic), writes the statements back. Skips files that already have statements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tamubot.core import config
from tamubot.ingestion.filters.metadata_enrichment import generate_summary_statements


def retry_one(chunk_file: Path, client) -> tuple[int, str]:
    """Retry one file. Returns (n_statements, error)."""
    data = json.loads(chunk_file.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if not chunks:
        return 0, "no chunks"
    course_id = data.get("course_metadata", {}).get("course_id", "")
    term = data.get("course_metadata", {}).get("term", "")

    statements, error = generate_summary_statements(chunks, course_id, term, client)
    data["summary_statements"] = statements
    if error:
        data["summary_statements_error"] = error
    elif "summary_statements_error" in data:
        del data["summary_statements_error"]

    chunk_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(statements), error


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument("--dry-run", action="store_true", help="Just list what would be retried")
    ap.add_argument(
        "--only-empty",
        action="store_true",
        default=True,
        help="Skip files that already have statements (default)",
    )
    args = ap.parse_args()

    chunk_dir = Path(f"data/syllabi/{args.dept}/silver/06_chunk")
    if not chunk_dir.is_dir():
        print(f"ERROR: {chunk_dir} does not exist", file=sys.stderr)
        return 1

    targets: list[Path] = []
    for f in sorted(chunk_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        ss = data.get("summary_statements", [])
        if not ss:
            targets.append(f)

    print(f"Found {len(targets)} files with empty summary_statements:")
    for f in targets:
        print(f"  {f.stem}")
    if args.dry_run:
        return 0

    client = config.get_tamu_client()
    print()
    print(f"Retrying {len(targets)} files...")
    results: list[tuple[str, int, str]] = []
    t0 = time.monotonic()
    for i, f in enumerate(targets, 1):
        n, err = retry_one(f, client)
        results.append((f.stem, n, err))
        status = "OK" if n > 0 else f"STILL EMPTY ({err or 'no error'})"
        print(f"  [{i}/{len(targets)}] {f.stem}: {n} stmts — {status}")

    elapsed = time.monotonic() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    populated = sum(1 for _, n, _ in results if n > 0)
    print(f"Populated: {populated}/{len(targets)}")
    if populated < len(targets):
        print("Still-empty files:")
        for stem, n, err in results:
            if n == 0:
                print(f"  {stem}: {err or 'no error'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
