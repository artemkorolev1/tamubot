"""Refresh enrich `headers` + re-chunk using Docling sidecar.

For each Gemini-skipped file in <dept>:
  1. Read current 04_hierarchy/<stem>.md and bronze/<stem>.headers.json sidecar.
  2. Rebuild headers field in 05_enrich/<stem>.json from sidecar (no LLM).
  3. Re-run chunk (no LLM call).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tamubot.ingestion.chunker_v4 import chunk_semantic
from tamubot.ingestion.filters.metadata_enrichment import (
    _build_headers_from_sidecar,
    _load_headers_sidecar,
)

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


def gemini_skipped_stems(dept: str) -> list[str]:
    log_path = Path(f"data/syllabi/{dept}/logs/filter_image_recovery_log.csv")
    if not log_path.exists():
        return []
    skipped: dict[str, str] = {}
    with log_path.open() as fh:
        for row in csv.DictReader(fh):
            stem = (row.get("file") or "").removesuffix(".md")
            status = row.get("status", "")
            if stem:
                skipped[stem] = status
    return sorted(s for s, st in skipped.items() if st == "skipped")


def refresh_headers_via_sidecar(stem: str, hierarchy_dir: Path, enrich_dir: Path) -> tuple[int, int]:
    """Update 05_enrich/<stem>.json headers field using the sidecar.

    Returns (header_count, with_page_count).
    """
    md_path = hierarchy_dir / f"{stem}.md"
    enrich_path = enrich_dir / f"{stem}.json"
    if not md_path.exists():
        raise FileNotFoundError(f"Missing markdown: {md_path}")
    if not enrich_path.exists():
        raise FileNotFoundError(f"Missing enrich JSON: {enrich_path}")

    sidecar = _load_headers_sidecar(stem, sidecar_dir=None)
    if sidecar is None:
        raise FileNotFoundError(f"No sidecar for {stem}")

    markdown = md_path.read_text(encoding="utf-8")
    headers = _build_headers_from_sidecar(markdown, sidecar)

    enrichment = json.loads(enrich_path.read_text(encoding="utf-8"))
    enrichment["headers"] = headers
    enrich_path.write_text(json.dumps(enrichment, indent=2, ensure_ascii=False), encoding="utf-8")

    with_page = sum(1 for h in headers if h["page"] is not None)
    return len(headers), with_page


def rechunk_one(
    stem: str,
    hierarchy_dir: Path,
    enrich_dir: Path,
    chunk_dir: Path,
    flag_threshold: int,
    min_chunk_tokens: int,
) -> dict:
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

    chunks_with_page = sum(1 for c in chunks if c.get("page") is not None)

    out_data = {
        "source_file": stem,
        "pipeline_version": "v4",
        "source": enrichment.get("source", ""),
        "course_type": enrichment.get("course_type", ""),
        "course_metadata": course_metadata,
        "course_summary": enrichment.get("course_summary", ""),
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
        "chunks_with_page": chunks_with_page,
        "flagged": sum(1 for c in chunks if c["flags"]),
        **log_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument("--flag-threshold", type=int, default=600)
    ap.add_argument("--min-chunk-tokens", type=int, default=50)
    ap.add_argument("--stem", action="append", default=None, help="Override stem list (repeatable)")
    args = ap.parse_args()

    base = Path(f"data/syllabi/{args.dept}/silver")
    hierarchy_dir = base / "04_hierarchy"
    enrich_dir = base / "05_enrich"
    chunk_dir = base / "06_chunk"

    stems = args.stem or gemini_skipped_stems(args.dept)
    if not stems:
        print("No stems to process.", file=sys.stderr)
        return 1

    print(f"Refreshing {len(stems)} files using Docling sidecars...\n")
    t0 = time.monotonic()

    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
        try:
            n, wp = refresh_headers_via_sidecar(stem, hierarchy_dir, enrich_dir)
            print(f"   headers (sidecar): {n} total, {wp} with page")
        except FileNotFoundError as e:
            print(f"   SKIP enrich-refresh: {e}")
            continue

        try:
            res = rechunk_one(
                stem,
                hierarchy_dir,
                enrich_dir,
                chunk_dir,
                args.flag_threshold,
                args.min_chunk_tokens,
            )
            print(f"   chunks={res['chunks']} (with page={res['chunks_with_page']}) flagged={res['flagged']}")
        except Exception as exc:
            print(f"   ERROR rechunk: {exc}")

    print(f"\nDone in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
