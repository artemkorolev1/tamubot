"""Second-pass retry for previously failed syllabus PDFs.

Scans tamu_data/processed/gemini_parsed/ for JSON files with a top-level "error"
key, re-runs parse_pdf() (with sanitization + collapse logic), and reports recovery.

Usage:
    python -m tamubot.ingestion.refine_errors
    python -m tamubot.ingestion.refine_errors --department CSCE
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tamubot.core import config
from tamubot.ingestion.process_syllabi import (
    DELAY_BETWEEN_CALLS,
    LOG_DIR,
    OUTPUT_DIR,
    SYLLABI_DIR,
    parse_pdf,
    write_per_file_report,
)


def find_error_jsons(department: str | None = None) -> list[Path]:
    """Return all JSON files in OUTPUT_DIR that have a top-level 'error' key."""
    error_files: list = []
    if not OUTPUT_DIR.exists():
        return error_files
    for json_path in sorted(OUTPUT_DIR.glob("*.json")):
        if department and f"_{department}_" not in json_path.name:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "error" in data:
                error_files.append(json_path)
        except (json.JSONDecodeError, IOError):
            pass
    return error_files


def get_pdf_path(json_path: Path) -> Path | None:
    """Resolve source PDF path from the error JSON's _source_file field."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        source = data.get("_source_file", "")
        if source:
            pdf_path = SYLLABI_DIR / source
            if pdf_path.exists():
                return pdf_path
    except (json.JSONDecodeError, IOError):
        pass
    # Fallback: JSON stem matches PDF filename
    pdf_path = SYLLABI_DIR / (json_path.stem + ".pdf")
    return pdf_path if pdf_path.exists() else None


def log_refine_error(filename: str, error: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "file": filename,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    with open(LOG_DIR / "refine_errors.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Retry failed syllabus PDFs")
    parser.add_argument("--department", type=str, help="Filter by department (e.g. CSCE)")
    args = parser.parse_args()

    if not config.TAMU_API_KEY:
        print("ERROR: Set TAMU_API_KEY environment variable")
        sys.exit(1)

    department = args.department.upper() if args.department else None
    error_jsons = find_error_jsons(department)

    if not error_jsons:
        print("No error files found to retry.")
        return

    print(f"{'='*60}")
    print(f"Refine Errors — Retrying {len(error_jsons)} failed file(s)")
    if department:
        print(f"Department filter: {department}")
    print(f"{'='*60}\n")

    client = config.get_tamu_client()

    recovered = 0
    still_failing = 0

    for i, json_path in enumerate(error_jsons):
        print(f"[{i+1}/{len(error_jsons)}] {json_path.stem}.pdf...", end=" ", flush=True)

        pdf_path = get_pdf_path(json_path)
        if pdf_path is None:
            print("SKIP — source PDF not found")
            log_refine_error(json_path.name, "Source PDF not found")
            still_failing += 1
            continue

        result = parse_pdf(client, pdf_path)
        write_per_file_report(pdf_path, result)

        if "error" in result:
            still_failing += 1
            print(f"STILL FAILING: {result['error'][:60]}")
            log_refine_error(pdf_path.name, result["error"])
        else:
            recovered += 1
            chunks = len(result.get("chunks", []))
            # Overwrite the error JSON with the recovered result
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"RECOVERED ({chunks} chunks)")

        if i < len(error_jsons) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n{'='*60}")
    print("Refine complete")
    print(f"  Recovered:     {recovered}")
    print(f"  Still failing: {still_failing}")
    if still_failing > 0:
        print(f"  See: {LOG_DIR / 'refine_errors.jsonl'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
