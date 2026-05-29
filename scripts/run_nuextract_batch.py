"""Load-once batch runner for the v6b NuExtract structured stage.

Loads the NuExtract3 model a single time, then extracts every requested stem,
writing the same output as the Dagster ``v6b_silver_structured`` asset
(``silver/05_structured/<stem>.json``).

Two amortizations over Dagster's per-partition execution:
  * the ~100 s model load happens once for the whole batch, and
  * text-mode docs are grouped into padded multi-doc forward passes.
    flash-linear-attention lets NuExtract3's gated-delta-rule layers handle
    padded batches; greedy output is byte-identical to single-doc (parity
    verified), so batching is a pure throughput win.

Batches are length-bucketed (see ``plan_batches``) so a few long syllabi don't
blow the ~8 GB VRAM budget. Vision-fallback stems (scanned/image-only PDFs) are
rare and per-page, so they stay sequential.

Usage:
    python scripts/run_nuextract_batch.py --dept STAT
    python scripts/run_nuextract_batch.py --dept STAT --limit 10 --force
    python scripts/run_nuextract_batch.py --stems 202611_STAT_608_600_12115_HP
    python scripts/run_nuextract_batch.py --dept STAT --batch-tokens 16000 --max-batch 8
"""

from __future__ import annotations

import argparse
import time

from tamubot.core import config
from tamubot.ingestion.clients.nuextract_client import get_extractor, plan_batches
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.assets.silver_structured import (
    apply_course_code_fix,
    extract_stem,
    needs_vision,
)

# Per-doc tokens added on top of the raw markdown: the JSON schema template +
# chat scaffolding. Used only for length-bucketing (a safety margin, not exact).
_TEMPLATE_OVERHEAD_TOKENS = 400


def _write_extract(stem: str, extract) -> None:
    out = paths.silver_structured_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(extract.model_dump_json(indent=2), encoding="utf-8")


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dept", nargs="+", help="Department code(s), e.g. STAT CSCE")
    g.add_argument("--stems", nargs="+", help="Explicit stem(s) to process")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of stems")
    ap.add_argument("--force", action="store_true", help="Re-extract even if output JSON exists")
    ap.add_argument(
        "--batch-tokens",
        type=int,
        default=16000,
        help="Max padded tokens per forward pass (batch_size * longest-doc); caps VRAM. Default 16000 (~8GB).",
    )
    ap.add_argument("--max-batch", type=int, default=8, help="Hard cap on docs per batch. Default 8.")
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
    extractor = get_extractor(quantize=True)
    http_backend = config.NUEXTRACT_BACKEND == "http"
    if http_backend:
        print(f"[backend] http — vLLM sidecar at {config.NUEXTRACT_SERVER_URL}\n", flush=True)
    else:
        print(f"[load] {time.time() - t_load:.1f}s — model loaded once for the batch\n", flush=True)

    # Classify pending stems: batchable text docs vs sequential vision fallbacks.
    text_items: list[tuple[str, str]] = []  # (stem, markdown)
    vision_stems: list[str] = []
    ok = errors = vision = 0
    for stem in pending:
        md_path = paths.bronze_md_path(stem)
        if not md_path.exists():
            errors += 1
            print(f"  {stem}: ERROR bronze md missing", flush=True)
            continue
        md = md_path.read_text(encoding="utf-8")
        if needs_vision(md) and paths.raw_path(stem).exists():
            vision_stems.append(stem)
        else:
            text_items.append((stem, md))

    t_batch = time.time()

    # --- batched text path ---
    if text_items:
        if http_backend:
            # The sidecar continuous-batches server-side; skip local token-bucketing
            # and fan out every doc at once via the client's bounded concurrency.
            t0 = time.time()
            extracts = extractor.extract_text_batch([md for _, md in text_items])
            for (stem, _), ex in zip(text_items, extracts):
                _write_extract(stem, apply_course_code_fix(ex, stem))
                ok += 1
            dt = time.time() - t0
            print(
                f"{len(text_items)} text docs via http (server-batched) in {dt:.1f}s "
                f"({dt / len(text_items):.1f}s/doc)",
                flush=True,
            )
        else:
            tok = extractor.processor.tokenizer
            lengths = [
                len(tok(md, add_special_tokens=False)["input_ids"]) + _TEMPLATE_OVERHEAD_TOKENS
                for _, md in text_items
            ]
            batches = plan_batches(lengths, max_batch_tokens=args.batch_tokens, max_batch_size=args.max_batch)
            print(
                f"{len(text_items)} text docs -> {len(batches)} batches "
                f"(<={args.max_batch} docs, <={args.batch_tokens} tok/batch)",
                flush=True,
            )
            for b, idxs in enumerate(batches, 1):
                group = [text_items[i] for i in idxs]
                t0 = time.time()
                try:
                    extracts = extractor.extract_text_batch([md for _, md in group])
                except Exception as exc:  # noqa: BLE001 — fall back so one bad batch doesn't lose the rest
                    print(
                        f"  [batch {b}/{len(batches)}] {len(group)} docs FAILED "
                        f"({type(exc).__name__}: {exc}); retrying individually",
                        flush=True,
                    )
                    for stem, md in group:
                        try:
                            ex = extractor.extract_text(md)
                        except Exception as exc2:  # noqa: BLE001
                            errors += 1
                            print(f"    {stem}: ERROR {type(exc2).__name__}: {exc2}", flush=True)
                            continue
                        _write_extract(stem, apply_course_code_fix(ex, stem))
                        ok += 1
                    continue
                for (stem, _), ex in zip(group, extracts):
                    _write_extract(stem, apply_course_code_fix(ex, stem))
                    ok += 1
                dt = time.time() - t0
                codes = ", ".join((ex.course_code or "-") for ex in extracts)
                print(
                    f"  [batch {b}/{len(batches)}] {len(group)} docs in {dt:.1f}s "
                    f"({dt / len(group):.1f}s/doc)  {codes}",
                    flush=True,
                )

    # --- sequential vision path ---
    for j, stem in enumerate(vision_stems, 1):
        t0 = time.time()
        try:
            extract, _ = extract_stem(extractor, stem)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [vision {j}/{len(vision_stems)}] {stem}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        _write_extract(stem, extract)
        ok += 1
        vision += 1
        print(
            f"  [vision {j}/{len(vision_stems)}] {stem}: {time.time() - t0:.1f}s  code={extract.course_code or '-'}",
            flush=True,
        )

    elapsed = time.time() - t_batch
    print(
        f"\n[done] {ok} ok, {errors} errors, {vision} via vision  in {elapsed / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
