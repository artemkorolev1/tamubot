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
from tamubot.ingestion.validation.table_quality import (
    check_no_degenerate_tables,
    check_no_table_lost,
)

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
        description=(
            "all image/table blocks processed (or modal disabled)"
            if unprocessed == 0
            else f"{unprocessed} image/table block(s) left unprocessed — modal budget likely hit"
        ),
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
        description=(
            f"all {checked} modal_result(s) valid"
            if not invalid
            else f"{len(invalid)} of {checked} modal_result(s) missing required keys ({_REQUIRED_MODAL_RESULT_KEYS})"
        ),
        metadata={
            "blocks_with_modal_result": checked,
            "invalid_count": len(invalid),
            "sample_errors": invalid[:10],
        },
    )


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_no_table_lost(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_TABLE_LOST gate: every table block must contribute >=1 row to the
    merged markdown via some tier of the degradation ladder (VLM → grid →
    partial). WARN for now — promote to BLOCK after corpus calibration."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    outcome = check_no_table_lost(blocks)
    lost = outcome.metadata.get("lost", 0)
    total = outcome.metadata.get("table_blocks_total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"all {total} table(s) survived into markdown"
            if outcome.passed
            else f"{lost} of {total} table(s) lost — no VLM/grid/partial rows survived"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_no_degenerate_tables(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """WARN when a table's VLM markdown shows repetition-loop degeneration —
    signals the presence_penalty lever didn't fully tame this table."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    outcome = check_no_degenerate_tables(blocks)
    degenerate = outcome.metadata.get("degenerate_tables", 0)
    total = outcome.metadata.get("table_blocks_total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"no degenerate tables across {total} table(s)"
            if outcome.passed
            else f"{degenerate} of {total} table(s) show repetition-loop degeneration"
        ),
        metadata=outcome.metadata,
    )
