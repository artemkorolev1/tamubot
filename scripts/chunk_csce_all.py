"""Batch chunk for all CSCE files (no LLM cost).

For each 03b_relocate_textbook .md whose 07_chunk JSON does not exist or
is older than its source (md or enrich), re-chunk using chunk_semantic +
page resolution from the enrichment JSON. Mirrors silver_chunk's logic.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tamubot.ingestion.chunker_v4 import chunk_semantic
from tamubot.ingestion.pipeline_v4 import _build_page_lookup, _load_enrichment, _resolve_page

DEPT = "CSCE"
ROOT = Path("data/syllabi") / DEPT / "v5"
SRC_DIR = ROOT / "silver" / "03b_relocate_textbook"
ENRICH_DIR = ROOT / "silver" / "05_enrich"
DST_DIR = ROOT / "silver" / "07_chunk"

FLAG_THRESHOLD = 600
MIN_CHUNK_TOKENS = 50

_PREREQ_RE = re.compile(
    r"^#{1,6}\s+(?:\d+(?:\.\d+)*\s+)?Course\s+Prerequisites\s*\n+(.*?)(?=\n#{1,6}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)


def chunk_one(stem: str, md_path: Path) -> dict:
    markdown = md_path.read_text(encoding="utf-8")
    chunks, log_info = chunk_semantic(markdown, flag_threshold=FLAG_THRESHOLD, min_chunk_tokens=MIN_CHUNK_TOKENS)
    enrichment = _load_enrichment(ENRICH_DIR, stem)
    page_lookup = _build_page_lookup(enrichment)
    for chunk in chunks:
        chunk["page"] = _resolve_page(chunk, page_lookup)
        if chunk["page"] is None and not chunk["header_path"]:
            chunk["page"] = 1
    course_metadata = enrichment.get("course_metadata", {})
    prereq_match = _PREREQ_RE.search(markdown)
    if prereq_match:
        course_metadata["prerequisites"] = prereq_match.group(1).strip()
    return {
        "source_file": stem,
        "pipeline_version": "v5",
        "source": enrichment.get("source", ""),
        "course_type": enrichment.get("course_type", ""),
        "course_metadata": course_metadata,
        "course_summary": enrichment.get("course_summary", ""),
        "chunk_config": {
            "strategy": "semantic",
            "flag_threshold": FLAG_THRESHOLD,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
        },
        "total_chunks": len(chunks),
        "chunks": chunks,
        "_parsed_at": datetime.now().isoformat(),
        "_log": log_info,
    }


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    all_md = sorted(SRC_DIR.glob("*.md"))
    todo: list[Path] = []
    for md in all_md:
        out = DST_DIR / f"{md.stem}.json"
        enrich = ENRICH_DIR / f"{md.stem}.json"
        # rerun if missing OR stale vs md OR stale vs enrich
        stale = (
            not out.exists()
            or out.stat().st_mtime < md.stat().st_mtime
            or (enrich.exists() and out.stat().st_mtime < enrich.stat().st_mtime)
        )
        if stale:
            todo.append(md)

    print(f"Total md: {len(all_md)} | needs chunk: {len(todo)}", flush=True)
    t0 = time.perf_counter()
    ok = err = 0
    for i, md in enumerate(todo, 1):
        try:
            data = chunk_one(md.stem, md)
            (DST_DIR / f"{md.stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            n_chunks = data["total_chunks"]
            n_flagged = sum(1 for c in data["chunks"] if c.get("flags"))
            print(f"[{i:>3}/{len(todo)}] {md.stem}  chunks={n_chunks} flagged={n_flagged}", flush=True)
            ok += 1
        except Exception as e:
            print(f"[{i:>3}/{len(todo)}] {md.stem}  ERROR: {e}", flush=True)
            err += 1
    print(f"\nDone. ok={ok} err={err} total_time={time.perf_counter() - t0:.1f}s", flush=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
