"""Asset checks for v6b_silver_modal."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths


@asset_check(asset="v6b_silver_modal", blocking=False)
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
