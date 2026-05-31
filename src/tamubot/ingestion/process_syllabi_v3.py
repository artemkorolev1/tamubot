"""
tamubot.ingestion.process_syllabi_v3

V3 syllabus pipeline — flat token-based chunking, no header parsing.

Steps:
  0. Copy source PDF  → v3_step0_source/{stem}_vNNN.pdf
  1. PDF → Markdown   → v3_step1_markdown/{stem}_vNNN.md
  2. Strip boilerplate→ v3_step2_boilerplate/{stem}_vNNN.md
  3. Flat chunk + LLM → v3_step3_flat/{stem}_vNNN.json

Usage:
    # Single PDF, new run:
    python -m tamubot.ingestion.process_syllabi_v3 \\
        --pdf tamu_data/raw/simple_syllabus_20260305/202611_CSCE_670_600_46627.pdf \\
        --new-run

    # Pilot run with custom chunk params:
    python -m tamubot.ingestion.process_syllabi_v3 --pilot --chunk-size 400 --overlap 50
    python -m tamubot.ingestion.process_syllabi_v3 --pilot --chunk-size 800 --overlap 150 --new-run

    # Full department:
    python -m tamubot.ingestion.process_syllabi_v3 --department CSCE --new-run

    # Individual step:
    python -m tamubot.ingestion.process_syllabi_v3 --pilot --step 3 --force
"""

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Keywords that flag a non-stripped header as a potential new boilerplate candidate
_BP_KEYWORDS = frozenset([
    "policy", "policies", "privacy", "ferpa", "ada", "disability",
    "wellness", "mental health", "nondiscrimination", "civil rights",
    "title ix", "copyright", "attendance", "makeup", "writing center",
    "technical support", "canvas", "perusall", "peerceptiv",
    "learning resources", "statement on", "notice of", "accommodation",
    "safety", "harassment", "discrimination",
    "evaluation",    # catches "course evaluation" variants
    "accessibility", # catches "accessibility statement" variants
    "honor",         # catches "aggie honor code" variants
    "pronouns",      # catches preferred name/pronouns sections
])

from tamubot.core import config  # noqa: E402
from tamubot.ingestion.boilerplate_stripper import (  # noqa: E402
    annotated_to_clean_markdown,
    pdf_to_annotated_markdown,
    render_stripped_markdown,
    strip_font_annotated_boilerplate,
)
from tamubot.ingestion.chunker_v3 import chunk_text  # noqa: E402
from tamubot.ingestion.pipeline_logger import (  # noqa: E402
    STEP0_ROOT,
    STEP1_ROOT,
    STEP2_ROOT,
    STEP3_V3_ROOT,
    StepLogger,
    resolve_latest_file,
    resolve_version,
    write_run_meta,
)

# ── Constants ──────────────────────────────────────────────────────────────────

RAW_ROOT = Path("tamu_data/raw")
PILOT_COURSES = ["CSCE_638", "CSCE_670", "CSCE_605", "ISEN_620", "ISEN_637"]

MODEL = config.TAMU_MODEL
MAX_RETRIES = 2

METADATA_PROMPT = """You are a university syllabus parser. Extract ONLY the course metadata fields from the text below.

Return ONLY valid JSON with this exact structure (omit fields you cannot find):
{
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
  "course_url": null
}

The input text uses font annotations like [13.0pt bold] to mark non-body text. Ignore these annotations for metadata extraction — just read the values.

Rules:
- Extract only what is explicitly stated. Do not infer or hallucinate.
- Return null for fields not found.
- Output must be valid JSON only — no markdown fences, no explanation.
"""


# ── Raw metadata lookup ────────────────────────────────────────────────────────

def _load_syllabus_metadata() -> dict[str, dict]:
    """Build {stem: {syllabus_url, doc_id}} from all simple_syllabus_*/simple_syllabus_metadata.json.

    Later directories (higher date) take precedence when stems overlap.
    """
    lookup: dict[str, dict] = {}
    for d in sorted(RAW_ROOT.glob("simple_syllabus_*")):
        meta_file = d / "simple_syllabus_metadata.json"
        if not meta_file.exists():
            continue
        try:
            entries = json.loads(meta_file.read_text(encoding="utf-8"))
            for filename, info in entries.items():
                stem = filename.removesuffix(".pdf")
                lookup[stem] = info
        except Exception as e:
            print(f"  WARN: Could not load {meta_file}: {e}")
    return lookup


# ── PDF source discovery ───────────────────────────────────────────────────────

def _simple_syllabus_dirs() -> list[Path]:
    return sorted(RAW_ROOT.glob("simple_syllabus_*"), reverse=True)


def find_pilot_pdfs() -> list[Path]:
    dirs = _simple_syllabus_dirs()
    pdfs: list[Path] = []
    for course_key in PILOT_COURSES:
        found = None
        for d in dirs:
            matches = sorted(d.glob(f"*_{course_key}_*.pdf"))
            if matches:
                found = matches[0]
                break
        if found:
            pdfs.append(found)
        else:
            print(f"  WARN: No PDF found for pilot course {course_key}")
    return pdfs


def find_department_pdfs(department: str) -> list[Path]:
    seen: dict[str, Path] = {}
    dept = department.upper()
    for d in _simple_syllabus_dirs():
        for p in sorted(d.glob(f"*_{dept}_*.pdf")):
            seen[p.stem] = p
    return sorted(seen.values(), key=lambda p: p.stem)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _sanitize_json(raw: str) -> str:
    raw = raw.replace("\x00", "")
    raw = re.sub(r"\\([^\"\\/bfnrtu0-9])", r"\\\\\1", raw)
    raw = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    return raw


def _clean_replacement_chars(obj):
    if isinstance(obj, str):
        return obj.replace("\ufffd", "-")
    if isinstance(obj, dict):
        return {k: _clean_replacement_chars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_replacement_chars(v) for v in obj]
    return obj


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw.strip()


def _new_bp_candidates(filtered_text: str) -> list[str]:
    """Return headers in the KEPT text that contain boilerplate-sounding keywords.

    These are headers that weren't matched by the registry but look like they
    might be institutional boilerplate — useful for expanding the registry.
    """
    font_prefix = re.compile(r"^\[[\d.]+pt(?:\s+bold)?\]\s+(.+)$")
    candidates: list[str] = []
    for line in filtered_text.splitlines():
        m = font_prefix.match(line)
        if m:
            header_text = m.group(1).strip()
            lower_h = header_text.lower()
            if any(kw in lower_h for kw in _BP_KEYWORDS):
                candidates.append(header_text)
    return candidates


# ── Step 0: Copy source PDF ───────────────────────────────────────────────────

def step0_copy_source(
    pdf_path: Path,
    version: str,
    logger: StepLogger,
    skip_existing: bool,
) -> dict:
    versioned_name = f"{pdf_path.stem}_{version}.pdf"
    out_file = STEP0_ROOT / versioned_name

    if skip_existing and any(STEP0_ROOT.glob(f"{pdf_path.stem}_v*.pdf")):
        return {"file": versioned_name, "status": "skipped"}

    try:
        shutil.copy2(str(pdf_path), str(out_file))
        size_kb = round(out_file.stat().st_size / 1024, 1)
        row = {
            "version": version,
            "file": versioned_name,
            "source_file": StepLogger.hyperlink(pdf_path, pdf_path.name),
            "output_file": StepLogger.hyperlink(out_file, versioned_name),
            "file_size_kb": size_kb,
            "status": "ok",
            "error": "",
            "copied_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {"file": versioned_name, "status": "ok", "out_file": out_file}
    except Exception as e:
        row = {
            "version": version,
            "file": versioned_name,
            "source_file": StepLogger.hyperlink(pdf_path, pdf_path.name),
            "output_file": "",
            "file_size_kb": 0,
            "status": "error",
            "error": str(e),
            "copied_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {"file": versioned_name, "status": "error", "error": str(e)}


# ── Step 1: PDF → Annotated Markdown ─────────────────────────────────────────

def step1_to_markdown(
    pdf_path: Path,
    version: str,
    logger: StepLogger,
    skip_existing: bool,
) -> dict:
    stem = pdf_path.stem
    stem = re.sub(r"_v\d{3}$", "", stem)
    versioned_name = f"{stem}_{version}.md"
    out_file = STEP1_ROOT / versioned_name

    if skip_existing and out_file.exists():
        return {
            "file": versioned_name,
            "status": "skipped",
            "md_text": out_file.read_text(encoding="utf-8"),
            "out_file": out_file,
            "stem": stem,
        }

    try:
        md_text, stats = pdf_to_annotated_markdown(pdf_path)
        out_file.write_text(md_text, encoding="utf-8")

        output_tokens = len(md_text) // 4
        row = {
            "version": version,
            "file": versioned_name,
            "source_pdf": StepLogger.hyperlink(pdf_path, pdf_path.name),
            "output_file": StepLogger.hyperlink(out_file, versioned_name),
            "body_font_size_pt": stats["body_size"],
            "annotated_lines": stats["annotated_lines"],
            "plain_lines": stats["plain_lines"],
            "output_tokens": output_tokens,
            "page_count": stats["page_count"],
            "status": "ok",
            "error": "",
            "processed_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {
            "file": versioned_name,
            "status": "ok",
            "md_text": md_text,
            "out_file": out_file,
            "stem": stem,
        }
    except Exception as e:
        row = {
            "version": version,
            "file": versioned_name,
            "source_pdf": StepLogger.hyperlink(pdf_path, pdf_path.name),
            "output_file": "",
            "body_font_size_pt": 0,
            "annotated_lines": 0,
            "plain_lines": 0,
            "output_tokens": 0,
            "page_count": 0,
            "status": "error",
            "error": str(e),
            "processed_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {"file": versioned_name, "status": "error", "error": str(e), "stem": stem}


# ── Step 2: Strip Boilerplate ─────────────────────────────────────────────────

def step2_strip_boilerplate(
    stem: str,
    md_text: str,
    input_file: "Path | None",
    version: str,
    logger: StepLogger,
    skip_existing: bool,
) -> dict:
    versioned_name = f"{stem}_{version}.md"
    stripped_name = f"{stem}_{version}_stripped.md"
    out_file = STEP2_ROOT / versioned_name
    stripped_file = STEP2_ROOT / stripped_name

    if skip_existing and out_file.exists():
        return {
            "file": versioned_name,
            "status": "skipped",
            "filtered_md": out_file.read_text(encoding="utf-8"),
            "out_file": out_file,
        }

    try:
        filtered_md, strip_log = strip_font_annotated_boilerplate(md_text)
        stripped_md = render_stripped_markdown(stem, strip_log)

        out_file.write_text(filtered_md, encoding="utf-8")
        stripped_file.write_text(stripped_md, encoding="utf-8")

        chars_stripped = sum(e.get("chars", 0) for e in strip_log)
        chars_remaining = len(filtered_md)
        chars_total = chars_stripped + chars_remaining
        reduction_pct = round(chars_stripped / max(chars_total, 1) * 100, 1)

        type_counter = Counter(e["type"] for e in strip_log)
        strip_type_counts = ",".join(
            f"{t}:{n}" for t, n in type_counter.most_common()
        )

        sorted_by_chars = sorted(strip_log, key=lambda e: e.get("chars", 0))
        largest = sorted_by_chars[-1] if sorted_by_chars else {}
        smallest = sorted_by_chars[0] if sorted_by_chars else {}

        # Full list of stripped headers: "[TYPE] Header" separated by " | "
        stripped_headers = " | ".join(
            f"[{e['type']}] {e['header']}" for e in strip_log
        )
        # Headers in kept text that look like unreported boilerplate
        new_bp = _new_bp_candidates(filtered_md)
        new_bp_str = " | ".join(new_bp) if new_bp else ""

        input_link = (
            StepLogger.hyperlink(input_file, input_file.name)
            if input_file and input_file.exists()
            else (str(input_file) if input_file else "")
        )
        row = {
            "version": version,
            "file": versioned_name,
            "input_file": input_link,
            "output_file": StepLogger.hyperlink(out_file, versioned_name),
            "stripped_file": StepLogger.hyperlink(stripped_file, stripped_name),
            "sections_stripped": len(strip_log),
            "tokens_stripped": chars_stripped // 4,
            "tokens_remaining": chars_remaining // 4,
            "reduction_pct": reduction_pct,
            "stripped_headers": stripped_headers,
            "new_bp_candidates": new_bp_str,
            "largest_strip_tokens": largest.get("chars", 0) // 4,
            "largest_strip_header": largest.get("header", ""),
            "smallest_strip_tokens": smallest.get("chars", 0) // 4,
            "smallest_strip_header": smallest.get("header", ""),
            "strip_type_counts": strip_type_counts,
            "status": "ok",
            "processed_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {
            "file": versioned_name,
            "status": "ok",
            "filtered_md": filtered_md,
            "strip_log": strip_log,
            "out_file": out_file,
        }
    except Exception as e:
        row = {
            "version": version,
            "file": versioned_name,
            "input_file": "",
            "output_file": "",
            "stripped_file": "",
            "sections_stripped": 0,
            "tokens_stripped": 0,
            "tokens_remaining": 0,
            "reduction_pct": 0,
            "largest_strip_tokens": 0,
            "largest_strip_header": "",
            "smallest_strip_tokens": 0,
            "smallest_strip_header": "",
            "strip_type_counts": "",
            "status": "error",
            "processed_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {"file": versioned_name, "status": "error", "error": str(e)}


# ── Step 3: Flat Chunk + LLM Metadata ────────────────────────────────────────

def _extract_metadata(client, annotated_text: str) -> tuple[dict, str]:
    """LLM call: extract course metadata from font-annotated text."""
    user_message = f"{METADATA_PROMPT}\n\n---\n\nSYLLABUS TEXT:\n{annotated_text}"
    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.0,
                max_tokens=4096,
                stream=True,
            )
            raw = "".join(c.choices[0].delta.content or "" for c in stream).strip()
            raw = _strip_fences(raw)
            try:
                meta = json.loads(raw)
            except json.JSONDecodeError:
                meta = json.loads(_sanitize_json(raw))
            return _clean_replacement_chars(meta), ""
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {}, str(e)
            print(f"    WARN: LLM attempt {attempt + 1} failed ({e}), retrying...")
    return {}, "unknown error"


def step3_flat_chunk_and_save(
    client,
    stem: str,
    filtered_text: str,
    input_file: "Path | None",
    version: str,
    logger: StepLogger,
    skip_existing: bool,
    chunk_size: int,
    overlap: int,
    syllabus_meta: "dict | None" = None,
) -> dict:
    """Flat chunk + LLM metadata → v3_step3_flat/{stem}_vNNN.json."""
    versioned_name = f"{stem}_{version}.json"
    out_file = STEP3_V3_ROOT / versioned_name

    if skip_existing and out_file.exists():
        return {"file": versioned_name, "status": "skipped"}

    try:
        clean_markdown = annotated_to_clean_markdown(filtered_text)
        meta, llm_error = _extract_metadata(client, filtered_text)
        # Inject syllabus URL + doc_id from raw metadata (overrides LLM course_url if present)
        if syllabus_meta:
            meta["syllabus_url"] = syllabus_meta.get("syllabus_url")
            meta["doc_id"] = syllabus_meta.get("doc_id")
        chunks = chunk_text(clean_markdown, chunk_size=chunk_size, overlap=overlap)

        out = {
            "course_metadata": meta,
            "chunks": chunks,
            "pipeline_version": "v3",
            "chunk_config": {"chunk_size": chunk_size, "overlap": overlap},
            "_parsed_at": datetime.now().isoformat(),
        }
        out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        input_tokens = len(clean_markdown) // 4
        avg_chunk_tokens = (
            round(sum(len(c["content"]) // 4 for c in chunks) / len(chunks))
            if chunks else 0
        )

        input_link = (
            StepLogger.hyperlink(input_file, input_file.name)
            if input_file and input_file.exists()
            else (str(input_file) if input_file else "")
        )
        row = {
            "version": version,
            "file": versioned_name,
            "input_file": input_link,
            "output_file": StepLogger.hyperlink(out_file, versioned_name),
            "course_id": meta.get("course_id", ""),
            "section": meta.get("section", ""),
            "crn": meta.get("crn", ""),
            "chunk_count": len(chunks),
            "chunk_size_setting": chunk_size,
            "overlap_setting": overlap,
            "input_tokens": input_tokens,
            "avg_chunk_tokens": avg_chunk_tokens,
            "llm_status": "error" if llm_error else "ok",
            "llm_error": llm_error,
            "status": "ok",
            "processed_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {
            "file": versioned_name,
            "status": "ok",
            "chunk_count": len(chunks),
            "out_file": out_file,
            "llm_error": llm_error,
        }
    except Exception as e:
        row = {
            "version": version,
            "file": versioned_name,
            "input_file": "",
            "output_file": "",
            "course_id": "",
            "section": "",
            "crn": "",
            "chunk_count": 0,
            "chunk_size_setting": chunk_size,
            "overlap_setting": overlap,
            "input_tokens": 0,
            "avg_chunk_tokens": 0,
            "llm_status": "error",
            "llm_error": str(e),
            "status": "error",
            "processed_at": datetime.now().isoformat(),
        }
        logger.log(row)
        return {"file": versioned_name, "status": "error", "error": str(e)}


# ── Per-PDF orchestrator ───────────────────────────────────────────────────────

def process_pdf(
    client,
    pdf_path: Path,
    version: str,
    loggers: dict,
    skip_existing: bool,
    run_steps: list[int],
    chunk_size: int,
    overlap: int,
    url_lookup: "dict | None" = None,
) -> dict:
    stem = re.sub(r"_v\d{3}$", "", pdf_path.stem)
    print(f"  {pdf_path.name}")
    result: dict = {"file": pdf_path.name, "status": "ok"}

    md_text: str = ""
    filtered_text: str = ""

    # ── Step 0 ──
    if 0 in run_steps:
        r = step0_copy_source(pdf_path, version, loggers[0], skip_existing)
        if r["status"] == "error":
            result.update({"status": "error", "error": r.get("error", "")})
            print(f"    ERROR step0: {r.get('error', '')}")
            return result
        print(f"    step0: {'skipped' if r['status'] == 'skipped' else 'copied'}")

    # ── Step 1 ──
    if 1 in run_steps:
        r = step1_to_markdown(pdf_path, version, loggers[1], skip_existing)
        if r["status"] == "error":
            result.update({"status": "error", "error": r.get("error", "")})
            print(f"    ERROR step1: {r.get('error', '')}")
            return result
        md_text = r.get("md_text", "")
        tok = len(md_text) // 4
        print(f"    step1: {'skipped' if r['status'] == 'skipped' else f'{tok} tokens'}")
    elif 2 in run_steps or 3 in run_steps:
        latest = resolve_latest_file(STEP1_ROOT, stem, ".md")
        if latest:
            md_text = latest.read_text(encoding="utf-8")
        else:
            print(f"    WARN: No step1 output found for {stem}, step2/3 will use empty text")

    # ── Step 2 ──
    if 2 in run_steps:
        input_file = resolve_latest_file(STEP1_ROOT, stem, ".md")
        r = step2_strip_boilerplate(stem, md_text, input_file, version, loggers[2], skip_existing)
        if r["status"] == "error":
            result.update({"status": "error", "error": r.get("error", "")})
            print(f"    ERROR step2: {r.get('error', '')}")
            return result
        filtered_text = r.get("filtered_md", "")
        n_stripped = len(r.get("strip_log", []))
        print(f"    step2: {'skipped' if r['status'] == 'skipped' else f'{n_stripped} sections stripped'}")
    elif 3 in run_steps:
        latest = resolve_latest_file(STEP2_ROOT, stem, ".md")
        if latest:
            filtered_text = latest.read_text(encoding="utf-8")
        else:
            filtered_text = md_text

    # ── Step 3 ──
    if 3 in run_steps:
        if client is None:
            print("    WARN: No LLM client — skipping step3")
            return result
        input_file = resolve_latest_file(STEP2_ROOT, stem, ".md")
        syllabus_meta = url_lookup.get(stem) if url_lookup else None
        r = step3_flat_chunk_and_save(
            client, stem, filtered_text, input_file, version, loggers[3],
            skip_existing, chunk_size, overlap, syllabus_meta,
        )
        if r["status"] == "error":
            result.update({"status": "error", "error": r.get("error", "")})
            print(f"    ERROR step3: {r.get('error', '')}")
            return result
        result["chunk_count"] = r.get("chunk_count", 0)
        llm_err = r.get("llm_error", "")
        step3_msg = "skipped" if r["status"] == "skipped" else f"{r.get('chunk_count', 0)} chunks"
        llm_note = f" [LLM ERROR: {llm_err}]" if llm_err else ""
        print(f"    step3: {step3_msg}{llm_note}")

    return result


# ── Combined preprocessing log ────────────────────────────────────────────────

_STEM_RE = re.compile(r"_v\d{3}\.\w+$")


def _read_log_all_versions(log_path: Path) -> dict[str, dict[str, dict]]:
    """Read a step-log CSV and return {base_stem: {version: row}} for all versions."""
    if not log_path.exists():
        return {}
    result: dict[str, dict[str, dict]] = {}
    with log_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fname = row.get("file", "")
            stem = _STEM_RE.sub("", fname)
            version = row.get("version", "")
            result.setdefault(stem, {})[version] = row
    return result


def _read_log_latest(log_path: Path) -> dict[str, dict]:
    """Read a step-log CSV and return {base_stem: row} keeping the latest version per stem."""
    all_versions = _read_log_all_versions(log_path)
    return {
        stem: versions[max(versions)]
        for stem, versions in all_versions.items()
    }


def _reduction_notes(reduction_pct: object, body_font_size_pt: object,
                     tokens_remaining: object = "") -> str:
    """Auto-computed QA note for extreme reduction_pct outliers.

    Manual fix history is stored in _annotations.json and takes precedence
    over this auto-computed value in generate_combined_log.
    """
    try:
        pct = float(reduction_pct)  # type: ignore[arg-type]
        bfs = float(body_font_size_pt) if body_font_size_pt != "" else None  # type: ignore[arg-type]
        rem = int(tokens_remaining) if tokens_remaining != "" else None  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return ""
    if pct < 5 and bfs is not None and bfs < 11.0:
        return f"issue: body_font_misdetect (detected {bfs}pt, content 11pt); {pct}% stripped"
    if pct > 75:
        rem_str = f", {rem} tokens remaining" if rem is not None else ""
        return f"thin_syllabus: {pct}% stripped{rem_str}"
    return ""


def generate_combined_log() -> Path:
    """Join all four step logs on base stem and write _full_preprocess_log.csv.

    Anchored on step3: only stems that completed step3 appear as rows.
    For each such stem, the step3 version is used to look up the matching
    step0/1/2 rows exactly — no cross-version bleed.

    Output: tamu_data/processed/v3_step3_flat/_full_preprocess_log.csv
    """
    s0_all = _read_log_all_versions(STEP0_ROOT / "step0_source_copy_log.csv")
    s1_all = _read_log_all_versions(STEP1_ROOT / "step1_pdf_to_markdown_log.csv")
    s2_all = _read_log_all_versions(STEP2_ROOT / "step2_boilerplate_strip_log.csv")
    s3 = _read_log_latest(STEP3_V3_ROOT / "step3_flat_chunk_log.csv")

    all_stems = sorted(s3.keys())

    FIELDS = [
        # Identity
        "stem", "version", "course_id", "crn", "section",
        # Step 0 — source
        "file_size_kb", "s0_status", "source_pdf",
        # Step 1 — PDF → Markdown
        "page_count", "body_font_size_pt", "annotated_lines", "plain_lines",
        "annotated_tokens", "s1_status", "s1_md_file",
        # Step 2 — boilerplate
        "sections_stripped", "tokens_stripped", "tokens_remaining",
        "reduction_pct", "reduction_notes", "strip_type_counts",
        "stripped_headers", "new_bp_candidates",
        "s2_status", "s2_md_file",
        # Step 3 — chunks
        "input_tokens", "chunk_count", "avg_chunk_tokens",
        "chunk_size_setting", "overlap_setting",
        "llm_status", "llm_error", "s3_status", "s3_json_file", "processed_at",
        "validation_feedback",
    ]

    rows: list[dict] = []
    for stem in all_stems:
        r3 = s3[stem]
        version = r3.get("version", "")
        # Look up matching rows for this exact version in earlier steps
        r0 = s0_all.get(stem, {}).get(version, {})
        r1 = s1_all.get(stem, {}).get(version, {})
        r2 = s2_all.get(stem, {}).get(version, {})

        new_bp = r2.get("new_bp_candidates", "")
        new_bp_count = f"{len(new_bp.split(' | '))} candidates" if new_bp else ""

        rows.append({
            "stem":               stem,
            "version":            version,
            "course_id":          r3.get("course_id", ""),
            "crn":                r3.get("crn", ""),
            "section":            r3.get("section", ""),
            # step 0
            "file_size_kb":       r0.get("file_size_kb", ""),
            "s0_status":          r0.get("status", ""),
            "source_pdf":         r0.get("source_file", ""),
            # step 1
            "page_count":         r1.get("page_count", ""),
            "body_font_size_pt":  r1.get("body_font_size_pt", ""),
            "annotated_lines":    r1.get("annotated_lines", ""),
            "plain_lines":        r1.get("plain_lines", ""),
            "annotated_tokens":   r1.get("output_tokens", ""),
            "s1_status":          r1.get("status", ""),
            "s1_md_file":         r1.get("output_file", ""),
            # step 2
            "sections_stripped":  r2.get("sections_stripped", ""),
            "tokens_stripped":    r2.get("tokens_stripped", ""),
            "tokens_remaining":   r2.get("tokens_remaining", ""),
            "reduction_pct":      r2.get("reduction_pct", ""),
            "reduction_notes":    _reduction_notes(
                r2.get("reduction_pct", ""),
                r1.get("body_font_size_pt", ""),
                r2.get("tokens_remaining", ""),
            ),
            "strip_type_counts":  r2.get("strip_type_counts", ""),
            "stripped_headers":   r2.get("stripped_headers", ""),
            "new_bp_candidates":  new_bp_count,
            "s2_status":          r2.get("status", ""),
            "s2_md_file":         r2.get("output_file", ""),
            # step 3
            "input_tokens":       r3.get("input_tokens", ""),
            "chunk_count":        r3.get("chunk_count", ""),
            "avg_chunk_tokens":   r3.get("avg_chunk_tokens", ""),
            "chunk_size_setting": r3.get("chunk_size_setting", ""),
            "overlap_setting":    r3.get("overlap_setting", ""),
            "llm_status":         r3.get("llm_status", ""),
            "llm_error":          r3.get("llm_error", ""),
            "s3_status":          r3.get("status", ""),
            "s3_json_file":       r3.get("output_file", ""),
            "processed_at":       r3.get("processed_at", ""),
            "validation_feedback": "",
        })

    rows.sort(key=lambda r: r.get("processed_at", ""), reverse=True)

    # Merge manual annotations — override auto-computed reduction_notes.
    # Edit tamu_data/processed/v3_step3_flat/_annotations.json to add fix history.
    # Format: {"stem": "issue: ...; solution: ...; result: ..."}
    annotations_path = STEP3_V3_ROOT / "_annotations.json"
    annotations: dict[str, str] = {}
    if annotations_path.exists():
        with annotations_path.open(encoding="utf-8") as f:
            annotations = json.load(f)
    for row in rows:
        note = annotations.get(row["stem"])
        if note:
            row["reduction_notes"] = note

    out_path = STEP3_V3_ROOT / "_full_preprocess_log.csv"
    tmp_path = STEP3_V3_ROOT / "_full_preprocess_log.csv.tmp"

    for attempt in range(5):
        try:
            with tmp_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(rows)
            tmp_path.replace(out_path)
            return out_path
        except PermissionError:
            if attempt == 4:
                import warnings
                warnings.warn(f"Combined log locked after 5 retries: {out_path}", stacklevel=2)
                return out_path
            time.sleep(1.0)

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V3 syllabus pipeline — flat token-based chunking"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pilot", action="store_true", help="Process the 5 pilot courses")
    group.add_argument("--pdf", type=str, help="Process a single PDF file")
    group.add_argument("--department", type=str, help="Process all PDFs for a department")
    parser.add_argument("--new-run", action="store_true",
                        help="Increment version counter (vNNN+1).")
    parser.add_argument("--step", type=int, choices=[0, 1, 2, 3],
                        help="Run only a specific step (default: all steps 0-3)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if versioned output file already exists")
    parser.add_argument("--chunk-size", type=int, default=600,
                        help="Target tokens per chunk (default: 600)")
    parser.add_argument("--overlap", type=int, default=100,
                        help="Overlap tokens between chunks (default: 100)")
    args = parser.parse_args()

    run_steps = [args.step] if args.step is not None else [0, 1, 2, 3]
    force_new = args.new_run and (0 in run_steps or args.step is None)
    version = resolve_version(force_new=force_new)
    print(f"Pipeline version: {version}  (steps: {run_steps}, chunk_size={args.chunk_size}, overlap={args.overlap})")

    step_roots = {0: STEP0_ROOT, 1: STEP1_ROOT, 2: STEP2_ROOT, 3: STEP3_V3_ROOT}
    cli_args = vars(args)
    for s in run_steps:
        step_roots[s].mkdir(parents=True, exist_ok=True)
        write_run_meta(step_roots[s], version, cli_args)

    log_names = {
        0: "step0_source_copy_log",
        1: "step1_pdf_to_markdown_log",
        2: "step2_boilerplate_strip_log",
        3: "step3_flat_chunk_log",
    }
    loggers = {
        s: StepLogger(step_roots[s] / log_names[s])
        for s in run_steps
    }

    client = None
    if 3 in run_steps:
        client = config.get_tamu_client()

    url_lookup = _load_syllabus_metadata()
    print(f"URL lookup  : {len(url_lookup)} entries from raw metadata")

    if args.pdf:
        pdfs = [Path(args.pdf)]
    elif args.pilot:
        pdfs = find_pilot_pdfs()
        print(f"Pilot mode: {len(pdfs)} PDFs")
    elif args.department:
        pdfs = find_department_pdfs(args.department)
        print(f"Department {args.department.upper()}: {len(pdfs)} PDFs")
    else:
        parser.print_help()
        sys.exit(0)

    results = []
    skip_existing = not args.force
    for pdf in pdfs:
        r = process_pdf(
            client, pdf, version, loggers, skip_existing, run_steps,
            args.chunk_size, args.overlap, url_lookup,
        )
        results.append(r)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"Version    : {version}")
    print(f"Steps      : {run_steps}")
    print(f"Chunk cfg  : size={args.chunk_size}, overlap={args.overlap}")
    print(f"Done       : {len(ok)} ok, {len(skipped)} skipped, {len(errors)} errors")
    for s in run_steps:
        print(f"Step {s} dir : {step_roots[s]}/")
    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  {r['file']}: {r.get('error', '?')}")

    # Always regenerate the combined log, even for partial step runs
    combined = generate_combined_log()
    print(f"Combined log: {combined}")


if __name__ == "__main__":
    main()
