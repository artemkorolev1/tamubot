"""One-shot batch enrich for the remaining CSCE files.

Runs enrich_file() with Docling sidecar (not PyMuPDF fallback) on every
3b_relocate_textbook .md whose 05_enrich JSON does not yet exist or is
older than the markdown. Skips already-fresh outputs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from tamubot.core import config
from tamubot.ingestion.filters.metadata_enrichment import enrich_file, load_scraped_metadata

DEPT = "CSCE"
ROOT = Path("data/syllabi") / DEPT / "v5"
SRC_DIR = ROOT / "silver" / "03b_relocate_textbook"
DST_DIR = ROOT / "silver" / "05_enrich"
SIDECAR_DIR = ROOT / "bronze"


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    client = config.get_tamu_client()
    scraped_meta = load_scraped_metadata()

    all_md = sorted(SRC_DIR.glob("*.md"))
    todo: list[Path] = []
    for md in all_md:
        out = DST_DIR / f"{md.stem}.json"
        if not out.exists() or out.stat().st_mtime < md.stat().st_mtime:
            todo.append(md)

    print(f"Total md: {len(all_md)} | needs enrich: {len(todo)}", flush=True)
    if not todo:
        return 0

    t0 = time.perf_counter()
    ok = err = 0
    for i, md in enumerate(todo, 1):
        ts = time.perf_counter() - t0
        try:
            result = enrich_file(md, DST_DIR, client, scraped_meta, sidecar_dir=SIDECAR_DIR)
            log = result.get("_log", {})
            print(
                f"[{i:>3}/{len(todo)}] {md.stem}  "
                f"meta={log.get('llm_metadata_status', '?')} "
                f"summary={log.get('llm_summary_status', '?')} "
                f"headers={log.get('header_count', 0)}(p={log.get('headers_with_page', 0)})  "
                f"elapsed={ts:.1f}s",
                flush=True,
            )
            ok += 1
        except Exception as e:
            print(f"[{i:>3}/{len(todo)}] {md.stem}  ERROR: {e}", flush=True)
            err += 1

    total = time.perf_counter() - t0
    print(f"\nDone. ok={ok} err={err} total_time={total:.1f}s", flush=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
