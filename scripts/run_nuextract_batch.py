"""Batch runner for the v6b silver_structured stage.

Drives ``process_stems`` outside Dagster for one-shot department runs. The text
path goes through TAMU Gemini (no GPU needed); NuExtract3 is lazy-loaded only
if a scanned-PDF vision fallback is required for some stem.

Usage:
    python scripts/run_nuextract_batch.py --dept STAT
    python scripts/run_nuextract_batch.py --dept STAT --limit 10 --force
    python scripts/run_nuextract_batch.py --stems 202611_STAT_608_600_12115_HP

The script name is kept for compatibility — the underlying extractor moved from
NuExtract (text) to TAMU Gemini (text) + NuExtract (scanned-PDF vision only).
"""

import argparse

from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.assets.silver_structured import process_stems


def discover_stems(depts: list[str]) -> list[str]:
    """Stems with v6b bronze markdown on disk, for the given depts (sorted)."""
    stems: list[str] = []
    for dept in depts:
        bronze_dir = paths.v6b_root(dept) / "bronze"
        if not bronze_dir.is_dir():
            continue
        stems.extend(sorted(p.stem for p in bronze_dir.glob("*.md")))
    return stems


def _lazy_nuextract():
    """Build the NuExtract extractor on first call. Wrapped in a closure so the
    ~3-minute model load only happens if some stem actually triggers the vision
    fallback path."""
    cache: dict = {}

    def get():
        if "x" not in cache:
            from tamubot.ingestion.clients.nuextract_client import get_extractor

            cache["x"] = get_extractor(quantize=True)
        return cache["x"]

    return get


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dept", nargs="+", help="Department code(s), e.g. STAT CSCE")
    g.add_argument("--stems", nargs="+", help="Explicit stem(s) to process")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of stems")
    ap.add_argument("--force", action="store_true", help="Re-extract even if output JSON exists")
    args = ap.parse_args()

    stems = args.stems or discover_stems([d.upper() for d in args.dept])
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        print("No stems found — nothing to do.")
        return

    process_stems(
        stems,
        nuextract_getter=_lazy_nuextract(),
        force=args.force,
        log=lambda m: print(m, flush=True),
    )


if __name__ == "__main__":
    main()
