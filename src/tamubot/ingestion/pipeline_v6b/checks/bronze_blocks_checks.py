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
