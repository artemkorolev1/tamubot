"""
Production pipeline: Parse all CSCE + ISEN syllabus PDFs with Gemini 2.5 Flash.

Features:
  - Resumes from where it left off (skips already-processed files)
  - Logs all errors to tamu_data/ingestion_logs/errors.jsonl
  - Saves each result immediately (no batch-at-end risk)
  - Rate-limit aware with configurable delay between calls
  - Summary report at the end

Usage:
    python -m tamubot.ingestion.process_syllabi
    python -m tamubot.ingestion.process_syllabi --department CSCE   # only CSCE
    python -m tamubot.ingestion.process_syllabi --retry-errors      # retry previously failed files
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import fitz  # PyMuPDF  # noqa: E402

from tamubot.core import config  # noqa: E402
from tamubot.ingestion.boilerplate_stripper import strip_pdf  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = config.TAMU_MODEL

SYLLABI_DIR = Path("tamu_data/raw/syllabi")
OUTPUT_DIR = Path(f"tamu_data/processed/gem_parsed_{datetime.now().strftime('%Y%m%d')}")
LOG_DIR = Path("tamu_data/logs")
REPORT_DIR = Path("tamu_data/logs/per_file")
PROGRESS_CSV = OUTPUT_DIR / "parsing_progress.csv"
PROGRESS_JSONL = OUTPUT_DIR / "parsing_progress.jsonl"

DEPARTMENTS = ["CSCE", "ISEN"]
MAX_RETRIES = 2
DELAY_BETWEEN_CALLS = 2  # seconds
DELAY_ON_RATE_LIMIT = 30  # seconds

# Token-count warning thresholds (min, max). Applied to total chunk content.
CSV_FIELDS = [
    "file",
    "course_id",
    "section",
    "crn",
    "status",
    "error_type",
    "error_detail",
    "chunk_count",
    "total_tok",
    "course_url",
    "flags",
    "parsed_at",
    "pdf_link",
    "json_link",
]

TOKEN_THRESHOLDS: dict[str, tuple[int, int]] = {
    "default": (20, 3000),
}

PROMPT = """You are a university syllabus parser. Analyze this PDF and extract ALL content into structured JSON.

OUTPUT FORMAT — return ONLY valid JSON:
{
  "course_metadata": {
    "course_id": "DEPT XXX",
    "section": "XXX",
    "term": "Spring 2026",
    "crn": "XXXXX",
    "instructor": {
      "name": "...",
      "email": "...",
      "office": "...",
      "office_hours": "..."
    },
    "teaching_assistants": [{"name": "...", "email": "..."}],
    "meeting_times": "...",
    "location": "...",
    "credit_hours": "...",
    "course_url": "<Canvas course URL or official course webpage if found, else null>"
  },
  "chunks": [
    {
      "title": "<section heading from the document>",
      "content": "<full text of this section, preserving all detail>",
      "has_table": true/false
    }
  ],
  "boilerplate_policies": [
    "<list NAMES of standard TAMU university policies found, e.g. 'ADA Policy', 'FERPA'>"
  ],
  "completeness_check": {
    "missing_sections": ["<section names that are NOT present in this syllabus, e.g. 'grading', 'schedule'>"],
    "warnings": ["<data quality issues, e.g. 'Grade weights not specified', 'Schedule has no dates'>"]
  }
}

RULES:
- Extract ALL content present in the text. Do not skip or summarize.
- Preserve tables as Markdown tables (| col1 | col2 | format). Set has_table=true.
- title field: use the actual section heading from the document, max ~80 chars. Never use the document header (college name, course name, section number) as the title.
- Each chunk should be a coherent section. Don't split mid-paragraph.
- Escape all special characters properly. Output must be valid JSON.
- course_url: extract the Canvas course URL or official course webpage URL if present in the
  syllabus. Set to null if not found.
"""


def get_pdf_list(departments: list[str], source_dir: Path | None = None) -> list[Path]:
    """Get all PDFs for the specified departments."""
    d = source_dir or SYLLABI_DIR
    pdfs = []
    for dept in departments:
        pattern = f"*_{dept}_*.pdf" if source_dir else f"202611_{dept}_*.pdf"
        pdfs.extend(sorted(d.glob(pattern)))
    return pdfs


def get_completed_files() -> set[str]:
    """Get set of filenames already successfully processed."""
    completed = set()
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.json"):
            # Verify it's a valid, non-error result
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if "error" not in data:
                    completed.add(f.stem + ".pdf")
            except json.JSONDecodeError, IOError:
                pass
    return completed


def get_error_files() -> set[str]:
    """Get set of filenames that previously errored."""
    errors = set()
    error_log = LOG_DIR / "errors.jsonl"
    if error_log.exists():
        with open(error_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    errors.add(entry.get("file", ""))
                except json.JSONDecodeError:
                    pass
    return errors


def log_error(filename: str, error: str, attempt: int):
    """Append an error entry to the error log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "file": filename,
        "error": error,
        "attempt": attempt,
        "timestamp": datetime.now().isoformat(),
    }
    with open(LOG_DIR / "errors.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_progress(completed: int, total: int, filename: str, status: str):
    """Append a progress entry."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "file": filename,
        "status": status,
        "progress": f"{completed}/{total}",
        "timestamp": datetime.now().isoformat(),
    }
    with open(LOG_DIR / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def sanitize_json(raw: str) -> str:
    """Attempt to clean common Gemini JSON output errors."""
    raw = raw.replace("\x00", "")  # null bytes
    # Fix invalid backslash escapes (not one of: " \ / b f n r t u 0-9)
    raw = re.sub(r'\\([^"\\/bfnrtu0-9])', r"\\\\\1", raw)
    # Strip bare control chars (0x01-0x1F except \t \n \r)
    raw = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    return raw


def clean_replacement_chars(obj):
    """Recursively replace Unicode replacement character (U+FFFD) with a hyphen.

    Gemini substitutes \ufffd when it encounters bytes it can't decode from the
    PDF (e.g. Windows-1252 en-dashes, ligatures). Replacing with '-' is safe
    for office hours ranges, schedules, etc.
    """
    if isinstance(obj, str):
        return obj.replace("\ufffd", "-")
    if isinstance(obj, dict):
        return {k: clean_replacement_chars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_replacement_chars(v) for v in obj]
    return obj


def collapse_chunks_by_category(chunks: list[dict]) -> list[dict]:
    """No-op: category-based collapsing is disabled. Returns chunks unchanged."""
    return chunks


def clean_template_noise(chunks: list[dict]) -> list[dict]:
    """Strip known TAMU template labels and empty-field artifacts from chunk content."""
    for chunk in chunks:
        content = chunk.get("content", "")

        # Strip template label "Prerequisite/Corequisite(s):" at start (PREREQUISITES sections)
        content = re.sub(r"^Prerequisite/Corequisite\(s\):\s*", "", content)

        # Strip "This material Is: Required/Recommended/Optional" lines (MATERIALS template)
        content = re.sub(r"^This material Is: \w+\n?", "", content, flags=re.MULTILINE)

        # Strip preamble (both "student" and "learner" variants) — LEARNING_OUTCOMES
        content = re.sub(
            r"^Upon completion of this course, the (?:student|learner) will be able to:\n?",
            "",
            content,
            flags=re.IGNORECASE,
        )

        # Strip bare "None" lines (empty template fields captured verbatim) — applied broadly
        lines = [ln for ln in content.splitlines() if ln.strip() != "None"]
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

        chunk["content"] = content.strip()
    return chunks


def dedup_course_summary(content: str) -> str:
    """Remove duplicate terms in COURSE_SUMMARY after generation.

    - Deduplicates within Topics itself (removes repeated terms).
    - Strips any Tools terms already present in Topics.
    - Drops any other legacy fields (Skills, Methods, Niche, Schedule) if the model emitted them.
    """

    def parse_terms(s: str) -> list[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    def norm(t: str) -> str:
        return t.lower().strip()

    lines = content.split("\n")
    result_lines = []
    topics_seen: set[str] = set()

    for line in lines:
        # Deduplicate within Topics
        m = re.match(r"^(\s*)(Topics):\s*(.*)", line)
        if m:
            indent, _, terms_str = m.groups()
            unique = []
            for t in parse_terms(terms_str):
                if norm(t) not in topics_seen:
                    topics_seen.add(norm(t))
                    unique.append(t)
            result_lines.append(f"{indent}Topics: {', '.join(unique)}")
            continue

        # Strip Tools terms already in Topics
        m = re.match(r"^(\s*)(Tools):\s*(.*)", line)
        if m:
            indent, _, terms_str = m.groups()
            unique = [t for t in parse_terms(terms_str) if norm(t) not in topics_seen]
            if unique:
                result_lines.append(f"{indent}Tools: {', '.join(unique)}")
            continue

        # Drop legacy fields the model might still emit
        if re.match(r"^\s*(Skills|Methods|Tools/Platforms|Niche|Schedule):", line):
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def write_per_file_report(pdf_path: Path, result: dict):
    """Write a human-readable .txt report for a processed PDF."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / (pdf_path.stem + ".txt")

    lines = [f"File: {pdf_path.name}"]

    if "error" in result:
        lines.append("Status: FAILED")
        lines.append(f"Error: {result['error']}")
        lines.append(f"Attempts: {result.get('_attempts', MAX_RETRIES + 1)}")
    else:
        chunks = result.get("chunks", [])
        parsed_at = result.get("_parsed_at", "")
        lines.append(f"Status: OK  |  Chunks: {len(chunks)}  |  Parsed: {parsed_at}")
        lines.append("")
        lines.append("Chunks:")
        for chunk in chunks:
            title = chunk.get("title", "")
            lines.append(f'  "{title}"')
        completeness = result.get("completeness_check", {})
        missing = completeness.get("missing_sections", [])
        warnings = completeness.get("warnings", [])
        if missing:
            lines.append("")
            lines.append(f"Missing sections: {', '.join(missing)}")
        if warnings:
            lines.append("Completeness warnings:")
            for w in warnings:
                lines.append(f"  - {w}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def count_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for English text)."""
    return max(0, round(len(text) / 4))


def classify_error(error_str: str) -> str:
    """Map a raw error string to a standardized error type."""
    s = error_str.lower()
    if "json parse error" in s:
        return "JSON_PARSE_ERROR"
    if "ssl" in s or "certificate" in s:
        return "SSL_ERROR"
    if "getaddrinfo" in s or "name or service not known" in s or "nodename nor servname" in s:
        return "DNS_ERROR"
    if "429" in error_str or "quota" in s or "rate" in s:
        return "RATE_LIMIT"
    if "exhausted retries" in s:
        return "EXHAUSTED_RETRIES"
    return "UNKNOWN_ERROR"


def build_progress_row(pdf_path: Path, result: dict) -> dict:
    """Build a CSV row dict for one processed PDF."""
    meta = result.get("course_metadata", {})

    # Total token count across all chunks; flag if total falls outside default thresholds
    total_content = "\n".join(c.get("content", "") for c in result.get("chunks", []))
    total_tok = count_tokens(total_content)
    flags: list[str] = []
    if total_content:
        lo, hi = TOKEN_THRESHOLDS["default"]
        if total_tok < lo:
            flags.append(f"TOTAL:TOO_SMALL({total_tok})")
        elif total_tok > hi:
            flags.append(f"TOTAL:TOO_LARGE({total_tok})")

    if "error" in result:
        error_type = classify_error(result["error"])
        error_detail = result["error"][:120]
    else:
        error_type = ""
        error_detail = ""

    pdf_abs = pdf_path.resolve()
    json_abs = (OUTPUT_DIR / f"{pdf_path.stem}.json").resolve()
    pdf_uri = pdf_abs.as_uri()
    json_uri = json_abs.as_uri()

    return {
        "file": pdf_path.name,
        "course_id": meta.get("course_id", ""),
        "section": meta.get("section", ""),
        "crn": meta.get("crn", ""),
        "status": "FAILED" if "error" in result else "OK",
        "error_type": error_type,
        "error_detail": error_detail,
        "chunk_count": len(result.get("chunks", [])),
        "total_tok": total_tok,
        "course_url": meta.get("course_url") or "",
        "flags": "; ".join(flags),
        "parsed_at": result.get("_parsed_at", datetime.now().isoformat()),
        "pdf_link": f'=HYPERLINK("{pdf_uri}","{pdf_path.name}")',
        "json_link": f'=HYPERLINK("{json_uri}","{pdf_path.stem}.json")',
    }


def load_progress_csv() -> list[dict]:
    """Load existing progress CSV rows, keyed by filename for deduplication."""
    if not PROGRESS_CSV.exists():
        return []
    try:
        with open(PROGRESS_CSV, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_progress_csv(rows: list[dict], retries: int = 5, delay: float = 2.0):
    """Rewrite the full progress CSV. Retries on PermissionError (e.g. Excel lock)."""
    import time

    if not rows:
        return
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(retries):
        try:
            tmp.replace(PROGRESS_CSV)
            return
        except PermissionError:
            if attempt < retries - 1:
                print(f"  CSV locked, retrying in {delay}s... (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                print(f"  WARNING: could not write CSV after {retries} attempts — saved to {tmp}")
                raise


def append_progress_jsonl(row: dict):
    """Append one row to the live-tail-able JSONL sidecar (append-only, no lock conflicts)."""
    PROGRESS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def parse_pdf(client, pdf_path: Path) -> dict:
    """Extract PDF text, pre-strip boilerplate, then parse structured JSON via TAMU API."""
    pdf_text, boilerplate_log = strip_pdf(pdf_path)
    if not pdf_text.strip():
        # Fallback: if stripper returns nothing, use raw text
        pdf_text = extract_pdf_text(pdf_path)
        boilerplate_log = []
    user_message = f"{PROMPT}\n\n---\n\nSYLLABUS TEXT:\n{pdf_text}"

    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.1,
                max_tokens=65536,
                response_format={"type": "json_object"},
                stream=True,  # TAMU gateway always returns SSE
            )
            raw = "".join(chunk.choices[0].delta.content or "" for chunk in stream).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = json.loads(sanitize_json(raw))

            # Clean up Unicode replacement chars (e.g. en-dashes in office hours)
            parsed = clean_replacement_chars(parsed)

            # collapse_chunks_by_category is now a no-op; kept for back-compat
            parsed["chunks"] = collapse_chunks_by_category(parsed.get("chunks", []))

            # Strip TAMU template labels and empty-field artifacts
            parsed["chunks"] = clean_template_noise(parsed["chunks"])

            # Strip has_table when false — only keep the field when a table is present
            for chunk in parsed["chunks"]:
                if not chunk.get("has_table"):
                    chunk.pop("has_table", None)

            # Deterministically remove cross-field duplicates from any COURSE_SUMMARY-style
            # block. With no category field we apply dedup uniformly — the function is a
            # no-op for content that lacks Topics:/Tools: lines.
            for chunk in parsed["chunks"]:
                chunk["content"] = dedup_course_summary(chunk["content"])

            # Trim bloated titles — take first segment before "/" and cap at 80 chars
            for chunk in parsed["chunks"]:
                title = chunk.get("title", "") or ""
                title = title.split("/")[0].strip().rstrip(":").strip()
                chunk["title"] = title[:80] if title else "Chunk"

            # Inject source filename, parse time, and boilerplate strip log
            parsed["_source_file"] = pdf_path.name
            parsed["_parsed_at"] = datetime.now().isoformat()
            parsed["_boilerplate_stripped"] = boilerplate_log

            return parsed

        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"
            log_error(pdf_path.name, error_msg, attempt)
            if attempt < MAX_RETRIES:
                time.sleep(DELAY_BETWEEN_CALLS)
            else:
                return {"error": error_msg, "_source_file": pdf_path.name, "_attempts": attempt + 1}

        except Exception as e:
            error_str = str(e)
            log_error(pdf_path.name, error_str, attempt)

            # Rate limit detection
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                print(f"    Rate limited. Waiting {DELAY_ON_RATE_LIMIT}s...")
                time.sleep(DELAY_ON_RATE_LIMIT)
            elif attempt < MAX_RETRIES:
                time.sleep(DELAY_BETWEEN_CALLS * (attempt + 1))
            else:
                return {"error": error_str, "_source_file": pdf_path.name, "_attempts": attempt + 1}

    return {"error": "Exhausted retries", "_source_file": pdf_path.name, "_attempts": MAX_RETRIES + 1}


def main():
    parser = argparse.ArgumentParser(description="Parse syllabus PDFs with Gemini 2.5 Flash")
    parser.add_argument("--department", type=str, help="Process only this department (e.g., CSCE)")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously failed files")
    parser.add_argument("--input-dir", type=str, help="Custom PDF input directory (overrides default SYLLABI_DIR)")
    args = parser.parse_args()

    if not config.TAMU_API_KEY:
        print("ERROR: Set TAMU_API_KEY environment variable")
        sys.exit(1)

    input_dir = Path(args.input_dir) if args.input_dir else None
    departments = [args.department.upper()] if args.department else DEPARTMENTS
    all_pdfs = get_pdf_list(departments, input_dir)
    completed = get_completed_files()

    if args.retry_errors:
        error_files = get_error_files()
        to_process = [p for p in all_pdfs if p.name in error_files]
        print(f"Retrying {len(to_process)} previously failed files")
    else:
        to_process = [p for p in all_pdfs if p.name not in completed]

    total = len(all_pdfs)
    already_done = len(completed)
    remaining = len(to_process)

    print(f"{'=' * 60}")
    print("Syllabus Processing Pipeline")
    print(f"{'=' * 60}")
    print(f"  Departments:  {', '.join(departments)}")
    print(f"  Total PDFs:   {total}")
    print(f"  Already done: {already_done}")
    print(f"  To process:   {remaining}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  Progress CSV: {PROGRESS_CSV}")
    print(f"  Reports dir:  {REPORT_DIR}")
    print(f"  Error log:    {LOG_DIR / 'errors.jsonl'}")
    print(f"{'=' * 60}\n")

    if remaining == 0:
        print("Nothing to process. All files already completed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    client = config.get_tamu_client()

    # Load existing progress rows; index by filename for upsert behaviour
    progress_rows = load_progress_csv()
    progress_index = {r["file"]: i for i, r in enumerate(progress_rows)}

    ok_count = 0
    fail_count = 0
    start_time = time.time()

    for i, pdf_path in enumerate(to_process):
        n = already_done + i + 1
        print(f"[{n}/{total}] {pdf_path.name} ({pdf_path.stat().st_size / 1024:.0f} KB)...", end=" ", flush=True)

        result = parse_pdf(client, pdf_path)
        write_per_file_report(pdf_path, result)

        # Update realtime progress CSV
        row = build_progress_row(pdf_path, result)
        if pdf_path.name in progress_index:
            progress_rows[progress_index[pdf_path.name]] = row
        else:
            progress_index[pdf_path.name] = len(progress_rows)
            progress_rows.append(row)
        # Save JSON first so it's never lost due to CSV write failure
        out_path = OUTPUT_DIR / pdf_path.name.replace(".pdf", ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        write_progress_csv(progress_rows)
        append_progress_jsonl(row)

        if "error" in result:
            fail_count += 1
            print(f"FAIL: {result['error'][:60]}")
            log_progress(n, total, pdf_path.name, "error")
        else:
            ok_count += 1
            chunks = len(result.get("chunks", []))
            missing = result.get("completeness_check", {}).get("missing_sections", [])
            warnings = result.get("completeness_check", {}).get("warnings", [])
            status_parts = [f"{chunks} chunks"]
            if missing:
                status_parts.append(f"missing: {','.join(missing)}")
            if warnings:
                status_parts.append(f"{len(warnings)} warnings")
            print(f"OK ({', '.join(status_parts)})")
            log_progress(n, total, pdf_path.name, "ok")

        time.sleep(DELAY_BETWEEN_CALLS)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"  Succeeded: {ok_count}")
    print(f"  Failed:    {fail_count}")
    print(f"  Total processed this run: {ok_count + fail_count}")
    print(f"  Overall progress: {already_done + ok_count}/{total}")
    if fail_count > 0:
        print("  Run with --retry-errors to retry failed files")
    print(f"{'=' * 60}")

    # Write summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "departments": departments,
        "total_pdfs": total,
        "previously_completed": already_done,
        "processed_this_run": ok_count + fail_count,
        "succeeded": ok_count,
        "failed": fail_count,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(LOG_DIR / "last_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
