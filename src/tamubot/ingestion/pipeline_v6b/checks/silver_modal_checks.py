"""Asset checks for v6b_silver_modal."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions

_REQUIRED_MODAL_RESULT_KEYS = ("confidence",)


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_budget_not_exceeded(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """WARN if modal was enabled and the run hit the budget — likely incomplete."""
    stem = context.partition_key
    modal_blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    unprocessed = sum(1 for b in modal_blocks if b.get("type") in ("image", "table") and not b.get("modal_result"))
    return AssetCheckResult(
        passed=unprocessed == 0,
        severity=AssetCheckSeverity.WARN,
        metadata={"unprocessed_image_or_table_blocks": unprocessed},
    )


@asset_check(asset="v6b_silver_modal", blocking=True, partitions_def=stem_partitions)
def v6b_silver_modal_result_schema_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """For every block with a modal_result, verify required keys are present."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    invalid: list[str] = []
    checked = 0
    for i, b in enumerate(blocks):
        mr = b.get("modal_result")
        if mr is None:
            continue
        checked += 1
        missing = [k for k in _REQUIRED_MODAL_RESULT_KEYS if k not in mr]
        if missing:
            invalid.append(f"block[{i}] missing: {','.join(missing)}")
    return AssetCheckResult(
        passed=len(invalid) == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "blocks_with_modal_result": checked,
            "invalid_count": len(invalid),
            "sample_errors": invalid[:10],
        },
    )
