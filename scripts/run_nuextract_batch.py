"""Load-once batch runner for the v6b NuExtract structured stage.

Loads the NuExtract3 model a single time, then extracts every requested stem
sequentially, writing the same output as the Dagster ``v6b_silver_structured``
asset (``silver/05_structured/<stem>.json``).

Why this exists: under Dagster's default per-partition execution the ~115 s model
load repeats for every file. Looping in one process amortizes the load across the
whole batch (e.g. 66 STAT files: ~3.2 h -> ~70 min). Batching multiple docs per
forward pass is NOT possible today — NuExtract3's gated-delta-rule layers fall back
to a torch kernel that crashes on padded batches (needs flash-linear-attention +
causal-conv1d). So this runs strictly sequential.

Usage:
    python scripts/run_nuextract_batch.py --dept STAT
    python scripts/run_nuextract_batch.py --dept STAT --limit 10 --force
    python scripts/run_nuextract_batch.py --stems 202611_STAT_608_600_12115_HP
"""

from __future__ import annotations

import argparse
import time

from tamubot.ingestion.clients.nuextract_client import NuExtractExtractor
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.assets.silver_structured import extract_stem


def discover_stems(depts: list[str]) -> list[str]:
    """Stems with v6b bronze markdown on disk, for the given depts (sorted)."""
    stems: list[str] = []
    for dept in depts:
        bronze_dir = paths.v6b_root(dept) / "bronze"
        if not bronze_dir.is_dir():
            continue
        stems.extend(sorted(p.stem for p in bronze_dir.glob("*.md")))
    return stems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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

    pending = [s for s in stems if args.force or not paths.silver_structured_path(s).exists()]
    skipped = len(stems) - len(pending)
    print(f"{len(stems)} stems ({skipped} already done, {len(pending)} to run)", flush=True)
    if not pending:
        return

    t_load = time.time()
    extractor = NuExtractExtractor.from_pretrained(quantize=True)
    print(f"[load] {time.time() - t_load:.1f}s — model loaded once for the batch\n", flush=True)

    ok = errors = vision = 0
    t_batch = time.time()
    for i, stem in enumerate(pending, 1):
        t0 = time.time()
        try:
            extract, used_vision = extract_stem(extractor, stem)
        except Exception as exc:  # noqa: BLE001 — log + continue so one bad file doesn't abort the batch
            errors += 1
            print(f"[{i}/{len(pending)}] {stem}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        out = paths.silver_structured_path(stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(extract.model_dump_json(indent=2), encoding="utf-8")
        ok += 1
        vision += int(used_vision)
        tag = " [vision]" if used_vision else ""
        print(
            f"[{i}/{len(pending)}] {stem}: {time.time() - t0:.1f}s  code={extract.course_code or '-'}{tag}",
            flush=True,
        )

    elapsed = time.time() - t_batch
    print(
        f"\n[done] {ok} ok, {errors} errors, {vision} via vision  "
        f"in {elapsed / 60:.1f} min  ({elapsed / max(ok, 1):.1f}s/file)",
        flush=True,
    )


if __name__ == "__main__":
    main()
