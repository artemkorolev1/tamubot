"""Asset checks for v6b_silver_chunk_semantic."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.validation.baseline_diff import (
    compute_baseline_delta,
    describe_delta,
    read_metadata_history,
)
from tamubot.ingestion.validation.schema_validation import check_chunks_schema_valid
from tamubot.ingestion.validation.table_quality import check_no_grading_weight_drift
from tamubot.ingestion.validation.token_distribution import (
    check_chunk_count_nonzero,
    check_low_no_header_rate,
    check_no_header_only_chunks,
    check_no_oversized_chunks,
)


def _load_chunks(stem: str) -> list[dict]:
    data = json.loads(paths.silver_chunk_semantic_path(stem).read_text(encoding="utf-8"))
    return data["chunks"]


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_count_nonzero(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_chunk_count_nonzero(chunks)
    n = outcome.metadata.get("chunk_count", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=f"{n} chunk(s)" if outcome.passed else "0 chunks — chunking produced nothing",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_oversized(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_oversized_chunks(chunks)
    over = outcome.metadata.get("oversized_count", 0)
    total = outcome.metadata.get("total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"no oversized chunks ({total} total)"
            if outcome.passed
            else f"{over} of {total} chunk(s) exceed the token cap"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_low_no_header_rate(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_low_no_header_rate(chunks, max_rate=0.10)
    rate = outcome.metadata.get("no_header_rate", 0.0)
    mx = outcome.metadata.get("max_rate", 0.10)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=(
            f"no-header rate {rate:.0%} within max {mx:.0%}"
            if outcome.passed
            else f"no-header rate {rate:.0%} exceeds max {mx:.0%} — too many headerless chunks"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_header_only(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """CHUNK_ORPHAN_HEADER gate (STAT_620 class): a chunk whose content is ONLY header
    lines (every non-blank line starts with ``#``) — a body-less orphan header. The
    INVERSE of low_no_header_rate (which flags header-LESS chunks). WARN for now;
    promote to blocking once the pass rate is confirmed across the golden set."""
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_header_only_chunks(chunks)
    n = outcome.metadata["header_only_count"]
    total = outcome.metadata["total"]
    paths_sample = outcome.metadata["offending_header_paths"]
    detail = f"; e.g. {', '.join(p for p in paths_sample[:3] if p)}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{n}/{total} chunk(s) are header-only orphans{detail}",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_grading_weights_sum_100(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_TABLE_LOST domain gate (ECEN_688 / CSCE_689 class): a grading-weight table
    whose row percentages don't sum to ~100% (a clean ``95`` = a dropped 5% row).
    Conservative — only evaluates chunks that are clearly grading breakdowns (grading/
    weight context AND >=3 percentage rows). WARN for now; promote to blocking once the
    pass rate is confirmed across the golden set."""
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_grading_weight_drift(chunks)
    n = outcome.metadata["offending_count"]
    evaluated = outcome.metadata["evaluated_grading_tables"]
    offenders = outcome.metadata["offenders"]
    detail = (
        f"; e.g. {offenders[0]['header_path']} sums to {offenders[0]['grading_sum']}%" if n and offenders else ""
    )
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"all {evaluated} grading table(s) sum to ~100%"
            if outcome.passed
            else f"{n}/{evaluated} grading table(s) miss 100±{outcome.metadata['tolerance']:.0f}%{detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_schema_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_chunks_schema_valid(chunks)
    invalid = outcome.metadata.get("invalid_count", 0)
    total = outcome.metadata.get("total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=(
            f"all {total} chunk(s) schema-valid"
            if outcome.passed
            else f"{invalid} of {total} chunk(s) fail schema validation"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_total_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    chunks = _load_chunks(stem)
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_chunk_semantic",
        partition_key=stem,
        metadata_key="total_chunks",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=len(chunks), history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="chunk_count " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_flagged_rate_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    chunks = _load_chunks(stem)
    current_rate = sum(1 for c in chunks if c.get("flags")) / len(chunks) if chunks else 0.0
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_chunk_semantic",
        partition_key=stem,
        metadata_key="flagged_chunks",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=current_rate, history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="flagged_rate " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )
