"""After manual refinement of 04_hierarchy markdown, refresh downstream artifacts.

Steps per refined file:
  1. Re-run map_headers_to_pages on the refined markdown (no LLM call).
     Update only the `headers` field in 05_enrich/<stem>.json — preserves
     course_metadata and course_summary (which the manual edits didn't invalidate).
  2. Re-run step_chunk equivalent: regenerate chunks from refined markdown
     using the freshened enrichment, including a new summary_statements call.

Budget: 1 LLM call per file (generate_summary_statements).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tamubot.core import config
from tamubot.ingestion.chunker_v4 import chunk_semantic
from tamubot.ingestion.filters.metadata_enrichment import (
    find_source_pdf,
    generate_summary_statements,
    map_headers_to_pages,
)

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

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_LEADING_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\s+")


def _build_page_lookup(enrichment: dict) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for h in enrichment.get("headers", []):
        text = h.get("text", "").strip().lower()
        page = h.get("page")
        if text and page is not None:
            lookup[text] = page
            stripped = _LEADING_NUM_RE.sub("", text)
            if stripped != text:
                lookup[stripped] = page
    return lookup


def _resolve_page(chunk: dict, page_lookup: dict[str, int]) -> int | None:
    hp = chunk.get("header_path", "")
    if not hp:
        return None
    own_header = hp.split(" > ")[-1].strip().lower()
    page = page_lookup.get(own_header)
    if page is None:
        page = page_lookup.get(_LEADING_NUM_RE.sub("", own_header))
    return page


def refresh_headers(stem: str, hierarchy_dir: Path, enrich_dir: Path) -> dict:
    """Refresh the `headers` field in 05_enrich/<stem>.json from current markdown."""
    md_path = hierarchy_dir / f"{stem}.md"
    enrich_path = enrich_dir / f"{stem}.json"

    if not md_path.exists():
        raise FileNotFoundError(f"Missing markdown: {md_path}")
    if not enrich_path.exists():
        raise FileNotFoundError(f"Missing enrich JSON: {enrich_path}")

    markdown = md_path.read_text(encoding="utf-8")
    enrichment = json.loads(enrich_path.read_text(encoding="utf-8"))

    pdf_path = find_source_pdf(stem)
    if pdf_path:
        headers = map_headers_to_pages(markdown, pdf_path)
    else:
        headers = [
            {"text": m.group(2).strip(), "level": len(m.group(1)), "page": None} for m in _HEADER_RE.finditer(markdown)
        ]

    enrichment["headers"] = headers
    enrich_path.write_text(json.dumps(enrichment, indent=2, ensure_ascii=False), encoding="utf-8")
    return enrichment


def rechunk_one(
    stem: str,
    hierarchy_dir: Path,
    enrich_dir: Path,
    chunk_dir: Path,
    flag_threshold: int,
    min_chunk_tokens: int,
    client,
) -> dict:
    """Regenerate 06_chunk/<stem>.json from refined markdown + freshened enrichment."""
    md_path = hierarchy_dir / f"{stem}.md"
    enrich_path = enrich_dir / f"{stem}.json"
    chunk_path = chunk_dir / f"{stem}.json"

    markdown = md_path.read_text(encoding="utf-8")
    enrichment = json.loads(enrich_path.read_text(encoding="utf-8"))

    chunks, log_info = chunk_semantic(
        markdown,
        flag_threshold=flag_threshold,
        min_chunk_tokens=min_chunk_tokens,
    )

    page_lookup = _build_page_lookup(enrichment)
    for chunk in chunks:
        chunk["page"] = _resolve_page(chunk, page_lookup)
        if chunk["page"] is None and not chunk["header_path"]:
            chunk["page"] = 1

    course_metadata = enrichment.get("course_metadata", {})
    prereq_match = re.search(
        r"^#{1,6}\s+(?:\d+(?:\.\d+)*\s+)?Course\s+Prerequisites\s*\n+(.*?)(?=\n#{1,6}\s|\Z)",
        markdown,
        re.MULTILINE | re.DOTALL,
    )
    if prereq_match:
        course_metadata["prerequisites"] = prereq_match.group(1).strip()

    course_id = course_metadata.get("course_id", "")
    term = course_metadata.get("term", "")
    statements, stmt_error = generate_summary_statements(chunks, course_id, term, client)
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
            "flag_threshold": flag_threshold,
            "min_chunk_tokens": min_chunk_tokens,
        },
        "total_chunks": len(chunks),
        "chunks": chunks,
        "_parsed_at": datetime.now().isoformat(),
    }
    chunk_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "chunks": len(chunks),
        "statements": len(statements),
        "flagged": sum(1 for c in chunks if c["flags"]),
        **log_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument("--flag-threshold", type=int, default=600)
    ap.add_argument("--min-chunk-tokens", type=int, default=50)
    args = ap.parse_args()

    base = Path(f"data/syllabi/{args.dept}/silver")
    hierarchy_dir = base / "04_hierarchy"
    enrich_dir = base / "05_enrich"
    chunk_dir = base / "06_chunk"

    client = config.get_tamu_client()
    t0 = time.monotonic()
    print(f"Refreshing {len(REFINED_STEMS)} files in {args.dept}...\n")

    for i, stem in enumerate(REFINED_STEMS, 1):
        print(f"[{i}/{len(REFINED_STEMS)}] {stem}")
        try:
            enrichment = refresh_headers(stem, hierarchy_dir, enrich_dir)
            n_headers = len(enrichment["headers"])
            print(f"   headers refreshed: {n_headers}")
        except FileNotFoundError as e:
            print(f"   SKIP: {e}")
            continue

        try:
            res = rechunk_one(
                stem,
                hierarchy_dir,
                enrich_dir,
                chunk_dir,
                args.flag_threshold,
                args.min_chunk_tokens,
                client,
            )
            print(f"   chunks={res['chunks']} stmts={res['statements']} flagged={res['flagged']}")
        except Exception as exc:
            print(f"   ERROR: {exc}")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
