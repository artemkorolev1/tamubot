"""Asset checks for v6b_silver_tag_semantic.

Blocking:
  - chunk_count_preserved: enforces the never-drop invariant.

Non-blocking (range warnings) — Phase 2:
  - boilerplate_rate_in_band: rate must sit in [BP_RATE_MIN, BP_RATE_MAX].
  - duplicate_rate_in_band: rate must sit at or below DUP_RATE_MAX.
"""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths

# Phase 2 calibration — see plan §2f.
BP_RATE_MIN = 0.05
BP_RATE_MAX = 0.45
DUP_RATE_MAX = 0.20


@asset_check(asset="v6b_silver_tag_semantic", blocking=True)
def v6b_silver_tag_chunk_count_preserved(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    src = json.loads(paths.silver_chunk_semantic_path(stem).read_text(encoding="utf-8"))
    dst = json.loads(paths.silver_tag_path(stem, "semantic").read_text(encoding="utf-8"))
    n_in = len(src["chunks"])
    n_out = len(dst["chunks"])
    return AssetCheckResult(
        passed=n_in == n_out,
        severity=AssetCheckSeverity.ERROR,
        metadata={"chunks_in": n_in, "chunks_out": n_out, "delta": n_out - n_in},
    )


@asset_check(asset="v6b_silver_tag_semantic", blocking=False)
def v6b_silver_tag_boilerplate_rate_in_band(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    dst = json.loads(paths.silver_tag_path(stem, "semantic").read_text(encoding="utf-8"))
    chunks = dst["chunks"]
    total = len(chunks)
    bp = sum(1 for c in chunks if c.get("is_boilerplate"))
    rate = bp / total if total else 0.0
    in_band = BP_RATE_MIN <= rate <= BP_RATE_MAX
    return AssetCheckResult(
        passed=in_band,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "boilerplate_count": bp,
            "total_chunks": total,
            "boilerplate_rate": round(rate, 4),
            "band_min": BP_RATE_MIN,
            "band_max": BP_RATE_MAX,
        },
    )


@asset_check(asset="v6b_silver_tag_semantic", blocking=False)
def v6b_silver_tag_duplicate_rate_in_band(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    dst = json.loads(paths.silver_tag_path(stem, "semantic").read_text(encoding="utf-8"))
    chunks = dst["chunks"]
    total = len(chunks)
    dup = sum(1 for c in chunks if c.get("is_duplicate"))
    rate = dup / total if total else 0.0
    in_band = rate <= DUP_RATE_MAX
    return AssetCheckResult(
        passed=in_band,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "duplicate_count": dup,
            "total_chunks": total,
            "duplicate_rate": round(rate, 4),
            "band_max": DUP_RATE_MAX,
        },
    )
