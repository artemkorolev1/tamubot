"""Asset checks for v6b_bronze_blocks. Moved out of the asset file so all v6b
checks live in one place."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.validation.baseline_diff import (
    compute_baseline_delta,
    read_metadata_history,
)
from tamubot.ingestion.validation.header_hierarchy import check_header_hierarchy_valid
from tamubot.ingestion.validation.text_quality import check_no_replacement_chars


@asset_check(asset="v6b_bronze_blocks", blocking=True)
def v6b_bronze_blocks_nonempty(context: AssetCheckExecutionContext) -> AssetCheckResult:
    stem = context.partition_key
    p = paths.bronze_blocks_path(stem)
    if not p.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "blocks.json missing"},
        )
    blocks = json.loads(p.read_text(encoding="utf-8"))
    return AssetCheckResult(
        passed=len(blocks) > 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"block_count": len(blocks)},
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False)
def v6b_bronze_blocks_has_text(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """A bronze block list with zero text blocks almost certainly failed parsing."""
    stem = context.partition_key
    p = paths.bronze_blocks_path(stem)
    blocks = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    text_blocks = sum(1 for b in blocks if b.get("type") == "text")
    return AssetCheckResult(
        passed=text_blocks >= 5,
        severity=AssetCheckSeverity.WARN,
        metadata={"text_block_count": text_blocks, "min_recommended": 5},
    )


@asset_check(asset="v6b_bronze_blocks", blocking=True)
def v6b_bronze_blocks_no_replacement_chars(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    text = "\n".join(b.get("text", "") for b in blocks if isinstance(b.get("text"), str))
    outcome = check_no_replacement_chars(text)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=True)
def v6b_bronze_blocks_header_hierarchy_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    headers = [{"level": b.get("level", 1), "text": b.get("text", "")} for b in blocks if b.get("type") == "heading"]
    outcome = check_header_hierarchy_valid(headers)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False)
def v6b_bronze_blocks_block_count_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_bronze_blocks",
        partition_key=stem,
        metadata_key="block_count",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=len(blocks), history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        metadata=outcome.metadata,
    )
