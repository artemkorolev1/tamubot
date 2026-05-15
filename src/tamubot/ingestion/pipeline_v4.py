"""
tamubot.ingestion.pipeline_v4

V4 syllabus pipeline — Docling-based conversion with modular filters.

Medallion data layout: data/syllabi/{raw,bronze,silver,gold}

Usage:
    # Full pipeline for a department:
    python -m tamubot.ingestion.pipeline_v4 --department ISEN --term "Fall 2026"

    # Specific steps:
    python -m tamubot.ingestion.pipeline_v4 --department ISEN --term "Fall 2025" \\
        --from convert --to boilerplate

    # Single step:
    python -m tamubot.ingestion.pipeline_v4 --department ISEN --term "Fall 2025" \\
        --step false_positive

    # Skip validation, use PyMuPDF:
    python -m tamubot.ingestion.pipeline_v4 --department ISEN --term "Fall 2025" \\
        --converter pymupdf --skip validate
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from tamubot.ingestion.pipeline_logger import StepLogger

# ── Constants ────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data/syllabi")
RAW_ROOT = DATA_ROOT / "raw"
BRONZE_ROOT = DATA_ROOT / "bronze"
SILVER_ROOT = DATA_ROOT / "silver"
GOLD_ROOT = DATA_ROOT / "gold"
LOGS_ROOT = DATA_ROOT / "logs"

RAW_SOURCE = Path("tamu_data/raw/simple_syllabus")


def setup_paths(dept: str | None) -> None:
    """Re-point all medallion roots under data/syllabi/<DEPT>/ when a dept is given.

    Leaves the shared data/syllabi/{raw,bronze,silver,gold,logs} layout in place
    for backwards compatibility (existing ISEN files were ingested there).

    Also propagates the new raw root to filters/image_recovery so its
    _find_raw_pdf() lookup resolves under the per-dept tree.
    """
    global RAW_ROOT, BRONZE_ROOT, SILVER_ROOT, GOLD_ROOT, LOGS_ROOT, SILVER_DIRS
    if not dept:
        return
    base = DATA_ROOT / dept.upper()
    RAW_ROOT = base / "raw"
    BRONZE_ROOT = base / "bronze"
    SILVER_ROOT = base / "silver"
    GOLD_ROOT = base / "gold"
    LOGS_ROOT = base / "logs"
    SILVER_DIRS = {
        "post_convert": SILVER_ROOT / "00_post_convert",
        "image_recovery": SILVER_ROOT / "01_image_recovery",
        "false_positive": SILVER_ROOT / "02_false_positive",
        "boilerplate": SILVER_ROOT / "03_boilerplate",
        "hierarchy": SILVER_ROOT / "04_hierarchy",
        "validate": SILVER_ROOT / "05_validate",
        "chunk": SILVER_ROOT / "06_chunk",
    }
    from tamubot.ingestion.filters import image_recovery as _ir

    _ir.RAW_ROOT = RAW_ROOT


# Ordered list of pipeline step names
ALL_STEPS = [
    "copy",
    "convert",
    "post_convert",
    "image_recovery",
    "false_positive",
    "boilerplate",
    "hierarchy",
    "validate",
    "chunk",
    "gold",
]

# Term code mapping: "Fall 2025" <-> "202541"
_SEASON_TO_CODE = {"spring": "11", "summer": "21", "fall": "41"}
_CODE_TO_SEASON = {"11": "Spring", "21": "Summer", "41": "Fall"}

# Silver sub-step directories
SILVER_DIRS = {
    "post_convert": SILVER_ROOT / "00_post_convert",
    "image_recovery": SILVER_ROOT / "01_image_recovery",
    "false_positive": SILVER_ROOT / "02_false_positive",
    "boilerplate": SILVER_ROOT / "03_boilerplate",
    "hierarchy": SILVER_ROOT / "04_hierarchy",
    "validate": SILVER_ROOT / "05_validate",
    "chunk": SILVER_ROOT / "06_chunk",
}


# ── Term helpers ─────────────────────────────────────────────────────────────


def term_to_code(term: str) -> str:
    """Convert 'Fall 2025' to '202541'."""
    parts = term.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid term format: {term!r}. Expected 'Season YYYY' (e.g., 'Fall 2025').")
    season, year = parts[0].lower(), parts[1]
    if season not in _SEASON_TO_CODE:
        raise ValueError(f"Unknown season: {parts[0]!r}. Use Spring, Summer, or Fall.")
    if not year.isdigit() or len(year) != 4:
        raise ValueError(f"Invalid year: {year!r}.")
    return f"{year}{_SEASON_TO_CODE[season]}"


def code_to_term(code: str) -> str:
    """Convert '202541' to 'Fall 2025'."""
    if len(code) != 6 or not code.isdigit():
        return code
    year = code[:4]
    season_code = code[4:]
    season = _CODE_TO_SEASON.get(season_code, f"?{season_code}")
    return f"{season} {year}"


# ── PDF discovery ────────────────────────────────────────────────────────────


def find_pdfs(department: str, term_code: str | None = None) -> list[Path]:
    """Find source PDFs for a department, optionally filtered by term code."""
    dept = department.upper()
    seen: dict[str, Path] = {}

    for dept_dir in RAW_SOURCE.glob(f"{dept}/*"):
        if not dept_dir.is_dir():
            continue
        for season_dir in sorted(dept_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            for pdf in sorted(season_dir.glob("*.pdf")):
                if term_code and not pdf.stem.startswith(term_code):
                    continue
                seen[pdf.stem] = pdf

    return sorted(seen.values(), key=lambda p: p.stem)


# ── Version resolution ────────────────────────────────────────��──────────────

_VERSION_RE = re.compile(r"_v(\d{3})\.")


def resolve_version(force_new: bool = True) -> str:
    """Scan data/syllabi/ for highest _vNNN, return next version."""
    max_n = 0
    for root in [RAW_ROOT, BRONZE_ROOT, GOLD_ROOT] + list(SILVER_DIRS.values()):
        if not root.exists():
            continue
        for f in root.iterdir():
            m = _VERSION_RE.search(f.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    if force_new or max_n == 0:
        return f"v{max_n + 1:03d}"
    return f"v{max_n:03d}"


# ── Step selection ───────────────────────────────────────────────────────────


def resolve_steps(args: argparse.Namespace) -> list[str]:
    """Determine which steps to run from CLI args."""
    if args.step:
        if args.step not in ALL_STEPS:
            print(f"Error: Unknown step {args.step!r}. Choose from: {', '.join(ALL_STEPS)}")
            sys.exit(1)
        return [args.step]

    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
        for s in steps:
            if s not in ALL_STEPS:
                print(f"Error: Unknown step {s!r}. Choose from: {', '.join(ALL_STEPS)}")
                sys.exit(1)
        return steps

    start = ALL_STEPS.index(args.from_step) if args.from_step else 0
    end = ALL_STEPS.index(args.to_step) + 1 if args.to_step else len(ALL_STEPS)

    steps = ALL_STEPS[start:end]

    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        steps = [s for s in steps if s not in skip]

    return steps


# ── Pipeline steps ───────────────────────────────────────────────────────────


def _ensure_dirs() -> None:
    """Create all output directories."""
    for d in [RAW_ROOT, BRONZE_ROOT, GOLD_ROOT, LOGS_ROOT] + list(SILVER_DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)


def step_copy(pdf_paths: list[Path], version: str) -> list[Path]:
    """Copy source PDFs to data/syllabi/raw/ with version suffix."""
    logger = StepLogger(LOGS_ROOT / "copy_log")
    copied = []
    for pdf in pdf_paths:
        out_name = f"{pdf.stem}_{version}.pdf"
        out_path = RAW_ROOT / out_name
        shutil.copy2(str(pdf), str(out_path))
        size_kb = round(out_path.stat().st_size / 1024, 1)
        logger.log(
            {
                "version": version,
                "file": out_name,
                "source": str(pdf),
                "size_kb": size_kb,
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
            }
        )
        copied.append(out_path)
        print(f"  Copied {pdf.name} ({size_kb} KB)")
    return copied


def step_convert(
    pdf_paths: list[Path],
    version: str,
    converter_name: str,
) -> list[Path]:
    """Convert PDFs to markdown using specified converter."""
    logger = StepLogger(LOGS_ROOT / "convert_log")
    outputs = []

    if converter_name == "docling":
        from tamubot.ingestion.converters.docling_converter import convert, create_converter

        print("  Loading Docling model (this takes a moment)...")
        dc = create_converter()
        for i, pdf in enumerate(pdf_paths, 1):
            stem = re.sub(r"_v\d{3}$", "", pdf.stem)
            out_name = f"{stem}_{version}.md"
            print(f"  [{i}/{len(pdf_paths)}] {pdf.name}...", end=" ", flush=True)
            result = convert(pdf, BRONZE_ROOT, converter=dc)
            # Rename output to versioned name
            versioned_path = BRONZE_ROOT / out_name
            if result.output_path != versioned_path:
                result.output_path.rename(versioned_path)
            print(f"{result.header_count} headers, {result.timing_s:.1f}s")
            logger.log(
                {
                    "version": version,
                    "file": out_name,
                    "converter": "docling",
                    "header_count": result.header_count,
                    "hierarchy_depth": json.dumps(result.hierarchy_depth),
                    "timing_s": round(result.timing_s, 2),
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            outputs.append(versioned_path)
    else:
        from tamubot.ingestion.converters.pymupdf_converter import convert

        for i, pdf in enumerate(pdf_paths, 1):
            stem = re.sub(r"_v\d{3}$", "", pdf.stem)
            out_name = f"{stem}_{version}.md"
            print(f"  [{i}/{len(pdf_paths)}] {pdf.name}...", end=" ", flush=True)
            result = convert(pdf, BRONZE_ROOT)
            versioned_path = BRONZE_ROOT / out_name
            if result.output_path != versioned_path:
                result.output_path.rename(versioned_path)
            print(f"{result.header_count} headers, {result.timing_s:.1f}s")
            logger.log(
                {
                    "version": version,
                    "file": out_name,
                    "converter": "pymupdf",
                    "header_count": result.header_count,
                    "hierarchy_depth": json.dumps(result.hierarchy_depth),
                    "timing_s": round(result.timing_s, 2),
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            outputs.append(versioned_path)

    return outputs


def step_filter(
    filter_name: str,
    input_dir: Path,
    output_dir: Path,
    version: str,
    filter_config: dict | None = None,
) -> Path:
    """Run a filter on all markdown files in input_dir matching the configured pattern."""
    from tamubot.ingestion.filters import (
        BoilerplateFilter,
        FalsePositiveFilter,
        HierarchyFilter,
        ImageRecoveryFilter,
        PostConvertCleanupFilter,
    )

    filter_map = {
        "post_convert": PostConvertCleanupFilter,
        "image_recovery": ImageRecoveryFilter,
        "false_positive": FalsePositiveFilter,
        "boilerplate": BoilerplateFilter,
        "hierarchy": HierarchyFilter,
    }

    filt = filter_map[filter_name]()
    print(f"  Running {filt.name} filter...")
    result = filt.apply(input_dir, output_dir, filter_config or {})

    logger = StepLogger(LOGS_ROOT / f"filter_{filter_name}_log")
    for entry in result.log_entries:
        entry["version"] = version
        entry["timestamp"] = datetime.now().isoformat()
        logger.log(entry)

    print(f"  {result.input_count} files processed, {result.modified_count} modified")
    if result.metrics:
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v}")

    return output_dir


def step_validate(
    input_dir: Path,
    version: str,
    version_label: str | None = None,
    file_pattern: str = "*.md",
    limit: int | None = None,
) -> Path:
    """Run LLM validation on filtered markdown files matching the configured pattern."""
    from tamubot.ingestion.validators.llm_validator import validate_directory

    output_dir = SILVER_DIRS["validate"]
    print("  Running LLM validation (4 checks per file)...")
    results = validate_directory(
        input_dir,
        None,
        output_dir,
        version_label=version_label,
        file_pattern=file_pattern,
        limit=limit,
    )

    logger = StepLogger(LOGS_ROOT / "validate_log")
    for r in results:
        logger.log(
            {
                "version": version,
                "file": r.file_stem,
                "total_issues": r.total_issues,
                **r.issue_counts,
                "timing_s": round(r.timing_s, 2),
                "timestamp": datetime.now().isoformat(),
            }
        )
        counts = r.issue_counts
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0) or "clean"
        print(f"    {r.file_stem}: {summary}")

    return output_dir


def _load_enrichment(enrichment_dir: Path, stem: str) -> dict:
    """Load enrichment JSON for a file stem, return empty dict if missing."""
    path = enrichment_dir / f"{stem}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_LEADING_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\s+")


def _build_page_lookup(enrichment: dict) -> dict[str, int]:
    """Build a lowercase header-text → page mapping from enrichment headers."""
    lookup: dict[str, int] = {}
    for h in enrichment.get("headers", []):
        text = h.get("text", "").strip().lower()
        page = h.get("page")
        if text and page is not None:
            lookup[text] = page
            # Also index without leading section numbers
            stripped = _LEADING_NUM_RE.sub("", text)
            if stripped != text:
                lookup[stripped] = page
    return lookup


def _resolve_page(chunk: dict, page_lookup: dict[str, int]) -> int | None:
    """Find the page number for a chunk by matching its header against enrichment."""
    hp = chunk.get("header_path", "")
    if not hp:
        return None
    # The chunk's own header is the last segment of header_path
    own_header = hp.split(" > ")[-1].strip().lower()
    page = page_lookup.get(own_header)
    if page is None:
        # Try stripping leading numbers
        page = page_lookup.get(_LEADING_NUM_RE.sub("", own_header))
    return page


def step_chunk(
    input_dir: Path,
    version: str,
    max_chunk_tokens: int,
    min_chunk_tokens: int,
    split_level: int,
    file_pattern: str = "*.md",
    limit: int | None = None,
) -> Path:
    """Semantic chunking: one chunk per section, enriched with course metadata.

    Produces ingestion-ready JSON documents in 06_chunk/.
    """
    from tamubot.core import config
    from tamubot.ingestion.chunk_report import generate_chunk_report
    from tamubot.ingestion.chunker_v4 import (
        chunk_semantic,
    )
    from tamubot.ingestion.filters.metadata_enrichment import generate_summary_statements

    output_dir = SILVER_DIRS["chunk"]
    enrichment_dir = SILVER_ROOT / "05_enrich"
    logger = StepLogger(LOGS_ROOT / "chunk_log")
    llm_client = config.get_tamu_client()

    md_files = sorted(input_dir.glob(file_pattern))
    if limit:
        md_files = md_files[:limit]
    print(f"  Chunking {len(md_files)} files (semantic mode)...")

    errors: list[dict] = []

    for md_file in md_files:
        stem = md_file.stem
        try:
            markdown = md_file.read_text(encoding="utf-8")
            chunks, log_info = chunk_semantic(
                markdown,
                flag_threshold=max_chunk_tokens,
                min_chunk_tokens=min_chunk_tokens,
            )

            # Load enrichment metadata and annotate chunks with page numbers
            enrichment = _load_enrichment(enrichment_dir, stem)
            page_lookup = _build_page_lookup(enrichment)
            for chunk in chunks:
                chunk["page"] = _resolve_page(chunk, page_lookup)
                # Root body (preamble before any header) is always page 1
                if chunk["page"] is None and not chunk["header_path"]:
                    chunk["page"] = 1

            # Extract prerequisites from markdown and add to metadata
            course_metadata = enrichment.get("course_metadata", {})
            prereq_match = re.search(
                r"^#{1,6}\s+(?:\d+(?:\.\d+)*\s+)?Course\s+Prerequisites\s*\n+(.*?)(?=\n#{1,6}\s|\Z)",
                markdown,
                re.MULTILINE | re.DOTALL,
            )
            if prereq_match:
                course_metadata["prerequisites"] = prereq_match.group(1).strip()

            # Generate page-anchored summary statements from the chunks. One LLM
            # call per file. Statements drive [Source N, p.X] citations on the
            # summary path in the same way chunks do on the detailed path.
            course_id = course_metadata.get("course_id", "")
            term = course_metadata.get("term", "")
            statements, stmt_error = generate_summary_statements(chunks, course_id, term, llm_client)
            if stmt_error:
                print(f"    WARN: summary_statements for {stem} failed: {stmt_error}")

            out_data = {
                "source_file": stem,
                "pipeline_version": "v4",
                "source": enrichment.get("source", ""),
                "course_type": enrichment.get("course_type", ""),
                "course_metadata": course_metadata,
                "course_summary": enrichment.get("course_summary", ""),
                "summary_statements": statements,
                "chunk_config": {
                    "strategy": "semantic",
                    "flag_threshold": max_chunk_tokens,
                    "min_chunk_tokens": min_chunk_tokens,
                },
                "total_chunks": len(chunks),
                "chunks": chunks,
                "_parsed_at": datetime.now().isoformat(),
            }

            out_path = output_dir / f"{stem}.json"
            out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

            flagged = sum(1 for c in chunks if c["flags"])
            enriched = "+" if enrichment else "-"
            print(f"    {stem}: {len(chunks)} chunks, {flagged} flagged, enrichment={enriched}")

            logger.log(
                {
                    "version": version,
                    "file": stem,
                    "chunk_count": len(chunks),
                    "has_enrichment": bool(enrichment),
                    **log_info,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as exc:
            errors.append({"file": stem, "error": str(exc)})
            print(f"    ERROR {stem}: {exc}")
            logger.log(
                {
                    "version": version,
                    "file": stem,
                    "chunk_count": 0,
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors:
            print(f"    {e['file']}: {e['error']}")

    # Generate chunking report
    report_path = output_dir / "chunk_report.xlsx"
    generate_chunk_report(
        input_dir=input_dir,
        enrichment_dir=enrichment_dir,
        output_path=report_path,
        flag_threshold=max_chunk_tokens,
        min_chunk_tokens=min_chunk_tokens,
        file_stems=[f.stem for f in md_files],
    )

    return output_dir


def step_gold(input_dir: Path, version: str) -> Path:
    """Copy final JSON files to gold directory."""
    json_files = sorted(input_dir.glob("*.json"))
    for jf in json_files:
        shutil.copy2(str(jf), str(GOLD_ROOT / jf.name))
    print(f"  {len(json_files)} files copied to gold/")
    return GOLD_ROOT


# ── Orchestrator ─────────────────────────────────────────────────────────────


def _input_dir_for_step(step: str, steps: list[str], file_pattern: str = "*.md") -> Path:
    """Determine the input directory for a step based on what ran before.

    For a step run alone (no filters earlier in the same invocation), fall back to
    the deepest silver dir that actually has files matching file_pattern, instead
    of going all the way back to bronze.
    """
    if step == "copy":
        return RAW_SOURCE  # not used directly
    if step == "convert":
        return RAW_ROOT

    filter_order = ["post_convert", "image_recovery", "false_positive", "boilerplate", "hierarchy"]

    def _deepest_silver_with_files(upto_idx: int) -> Path:
        """Return the latest silver dir (up to filter_order[upto_idx]) that has matching files."""
        for prev in reversed(filter_order[: upto_idx + 1]):
            sdir = SILVER_DIRS[prev]
            if sdir.exists() and any(sdir.glob(file_pattern)):
                return sdir
        return BRONZE_ROOT

    if step == "post_convert":
        return BRONZE_ROOT
    if step == "image_recovery":
        if "post_convert" in steps:
            return SILVER_DIRS["post_convert"]
        return _deepest_silver_with_files(0)
    if step == "false_positive":
        for prev in ("image_recovery", "post_convert"):
            if prev in steps:
                return SILVER_DIRS[prev]
        return _deepest_silver_with_files(1)
    if step == "boilerplate":
        for prev in ("false_positive", "image_recovery", "post_convert"):
            if prev in steps:
                return SILVER_DIRS[prev]
        return _deepest_silver_with_files(2)
    if step == "hierarchy":
        for prev in ("boilerplate", "false_positive", "image_recovery", "post_convert"):
            if prev in steps:
                return SILVER_DIRS[prev]
        return _deepest_silver_with_files(3)
    if step in ("validate", "chunk"):
        for prev in reversed(filter_order):
            if prev in steps:
                return SILVER_DIRS[prev]
        return _deepest_silver_with_files(4)
    if step == "gold":
        return SILVER_DIRS["chunk"]
    return BRONZE_ROOT


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the pipeline with the given configuration."""
    setup_paths(args.department)
    _ensure_dirs()
    steps = resolve_steps(args)
    version = resolve_version(force_new=True)
    term_code = term_to_code(args.term) if args.term else None

    # Display confirmation
    print()
    print("=" * 50)
    print("  Docling Pipeline v4")
    print("=" * 50)
    print(f"  Source:      {RAW_SOURCE}/")
    print(f"  Department:  {args.department}")
    if args.term:
        print(f"  Term:        {args.term} ({term_code})")
    print(f"  Converter:   {args.converter}")
    print(f"  Steps:       {' > '.join(steps)}")
    print(f"  Version:     {version}")

    if "chunk" in steps:
        print(
            f"  Chunk config: max={args.max_chunk_tokens}, min={args.min_chunk_tokens}, split_level={args.split_level}"
        )
    if "validate" in steps:
        print("  LLM validate: yes")
    elif "validate" not in steps and "validate" in ALL_STEPS:
        print("  LLM validate: skipped")

    # Find PDFs
    pdf_paths = find_pdfs(args.department, term_code)
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        print(f"\n  No PDFs found for {args.department}" + (f" term {args.term}" if args.term else ""))
        sys.exit(1)

    # Build a glob pattern that scopes every downstream step (filters, validate, chunk)
    # to the same dept/term as copy/convert. Filenames look like:
    #   <term_code>_<DEPT>_<course>_<section>_<crn>[_HP]_v<NNN>.md
    dept = args.department.upper()
    if term_code:
        file_pattern = f"{term_code}_{dept}_*.md"
    else:
        file_pattern = f"*_{dept}_*.md"
    filter_config = {"file_pattern": file_pattern}
    if args.limit:
        filter_config["limit"] = args.limit

    print(f"\n  Found {len(pdf_paths)} PDFs.")
    print(f"  Filter scope: glob={file_pattern!r}" + (f", limit={args.limit}" if args.limit else ""))
    print("=" * 50)

    if not args.yes:
        confirm = input("  Proceed? [Y/n] ").strip().lower()
        if confirm and confirm != "y":
            print("  Aborted.")
            sys.exit(0)

    print()
    t0 = time.monotonic()

    # Execute steps
    for step in steps:
        print(f"\n--- Step: {step} ---")

        if step == "copy":
            step_copy(pdf_paths, version)

        elif step == "convert":
            step_convert(
                # Use raw copies if copy step ran, otherwise use source PDFs
                list(RAW_ROOT.glob(f"*_{version}.pdf")) if "copy" in steps else pdf_paths,
                version,
                args.converter,
            )

        elif step in ("post_convert", "image_recovery", "false_positive", "boilerplate", "hierarchy"):
            input_dir = _input_dir_for_step(step, steps, file_pattern)
            step_filter(step, input_dir, SILVER_DIRS[step], version, filter_config)

        elif step == "validate":
            input_dir = _input_dir_for_step(step, steps, file_pattern)
            step_validate(input_dir, version, file_pattern=file_pattern, limit=args.limit)

        elif step == "chunk":
            input_dir = _input_dir_for_step(step, steps, file_pattern)
            step_chunk(
                input_dir,
                version,
                args.max_chunk_tokens,
                args.min_chunk_tokens,
                args.split_level,
                file_pattern=file_pattern,
                limit=args.limit,
            )

        elif step == "gold":
            step_gold(SILVER_DIRS["chunk"], version)

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 50}")
    print(f"  Pipeline complete. {len(steps)} steps in {elapsed:.1f}s")
    print(f"  Version: {version}")
    print(f"  Output:  {DATA_ROOT}/")
    print(f"{'=' * 50}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V4 Docling syllabus pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--department", required=True, help="Department code (e.g., ISEN, CSCE)")
    parser.add_argument("--term", default=None, help='Term filter (e.g., "Fall 2025", "Spring 2026")')
    parser.add_argument(
        "--converter", default="docling", choices=["docling", "pymupdf"], help="PDF converter (default: docling)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N PDFs")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    # Step selection
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument("--step", default=None, help="Run a single step")
    step_group.add_argument("--steps", default=None, help="Comma-separated steps to run")
    parser.add_argument("--from", dest="from_step", default=None, help="Start from this step")
    parser.add_argument("--to", dest="to_step", default=None, help="End at this step (inclusive)")
    parser.add_argument("--skip", default=None, help="Comma-separated steps to skip")

    # Chunk config
    parser.add_argument("--max-chunk-tokens", type=int, default=600, help="Max tokens per chunk (default: 600)")
    parser.add_argument("--min-chunk-tokens", type=int, default=50, help="Min tokens per chunk (default: 50)")
    parser.add_argument("--split-level", type=int, default=3, help="Deepest header level to split on (default: 3)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
