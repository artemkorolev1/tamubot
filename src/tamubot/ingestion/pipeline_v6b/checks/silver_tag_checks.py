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
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.validation.boilerplate_quality import check_no_boilerplate_overhide

# Phase 2 calibration — see plan §2f. Band widened to 35%–65% after the
# corpus retag (2026-06-08): grad syllabi share a near-constant ~11–13 chunks of
# university-mandated boilerplate (academic integrity, disability services,
# Title IX, attendance), so a healthy boilerplate_rate sits ~50%, not <45%.
BP_RATE_MIN = 0.35
BP_RATE_MAX = 0.65
DUP_RATE_MAX = 0.20


@asset_check(asset="v6b_silver_tag_semantic", blocking=True, partitions_def=stem_partitions)
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
        description=(
            f"chunk count preserved ({n_out})"
            if n_in == n_out
            else f"chunk count changed {n_in} → {n_out} (Δ {n_out - n_in:+d}) — tagging must never drop chunks"
        ),
        metadata={"chunks_in": n_in, "chunks_out": n_out, "delta": n_out - n_in},
    )


@asset_check(asset="v6b_silver_tag_semantic", blocking=False, partitions_def=stem_partitions)
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
        description=(
            f"boilerplate_rate {rate:.0%} within band {BP_RATE_MIN:.0%}–{BP_RATE_MAX:.0%}"
            if in_band
            else f"boilerplate_rate {rate:.0%} ({bp}/{total}) outside band {BP_RATE_MIN:.0%}–{BP_RATE_MAX:.0%}"
        ),
        metadata={
            "boilerplate_count": bp,
            "total_chunks": total,
            "boilerplate_rate": round(rate, 4),
            "band_min": BP_RATE_MIN,
            "band_max": BP_RATE_MAX,
        },
    )


@asset_check(asset="v6b_silver_tag_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_tag_no_boilerplate_overhide(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Boilerplate over-hide gate (ECEN_749 class): a chunk flagged
    ``is_boilerplate=True`` whose body carries a strong course-specific signal (a grade
    weight / penalty percentage or a concrete due date) — a customized section the
    header-anchored matcher wrongly suppressed. Tuned (corpus-calibrated 2026-06-10, 0/324
    standard policies flag) so genuine university policies don't trigger. WARN for now;
    promote to blocking once the pass rate is confirmed across the golden set."""
    stem = context.partition_key
    dst = json.loads(paths.silver_tag_path(stem, "semantic").read_text(encoding="utf-8"))
    outcome = check_no_boilerplate_overhide(dst["chunks"])
    n = outcome.metadata["overhide_count"]
    ha = outcome.metadata["header_anchored_overhide_count"]
    offenders = outcome.metadata["offenders"]
    detail = f"; e.g. {offenders[0]['header_path']} ({offenders[0]['signal']})" if n and offenders else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            "no course-specific signals in boilerplate-flagged chunks"
            if outcome.passed
            else f"{n} boilerplate chunk(s) ({ha} header-anchored) carry course-specific signals{detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_tag_semantic", blocking=False, partitions_def=stem_partitions)
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
        description=(
            f"duplicate_rate {rate:.0%} within max {DUP_RATE_MAX:.0%}"
            if in_band
            else f"duplicate_rate {rate:.0%} ({dup}/{total}) exceeds max {DUP_RATE_MAX:.0%}"
        ),
        metadata={
            "duplicate_count": dup,
            "total_chunks": total,
            "duplicate_rate": round(rate, 4),
            "band_max": DUP_RATE_MAX,
        },
    )
