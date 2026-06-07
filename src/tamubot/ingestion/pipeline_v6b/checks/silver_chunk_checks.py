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
    read_metadata_history,
)
from tamubot.ingestion.validation.schema_validation import check_chunks_schema_valid
from tamubot.ingestion.validation.token_distribution import (
    check_chunk_count_nonzero,
    check_low_no_header_rate,
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
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_oversized(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_oversized_chunks(chunks)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_low_no_header_rate(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_low_no_header_rate(chunks, max_rate=0.10)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_schema_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_chunks_schema_valid(chunks)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
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
        metadata=outcome.metadata,
    )
