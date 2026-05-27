"""Asset checks for v6b_silver_chunk_semantic."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.validation.schema_validation import check_chunks_schema_valid
from tamubot.ingestion.validation.token_distribution import (
    check_chunk_count_nonzero,
    check_low_no_header_rate,
    check_no_oversized_chunks,
)


def _load_chunks(stem: str) -> list[dict]:
    data = json.loads(paths.silver_chunk_semantic_path(stem).read_text(encoding="utf-8"))
    return data["chunks"]


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True)
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


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True)
def v6b_silver_chunk_no_oversized(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_oversized_chunks(chunks)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True)
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


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True)
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
