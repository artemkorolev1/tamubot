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
from tamubot.ingestion.validation.baseline_diff import (
    compute_baseline_delta,
    describe_delta,
    read_metadata_history,
)
from tamubot.ingestion.validation.table_quality import (
    check_confidence_floor,
    check_no_degenerate_tables,
    check_no_failure_markers,
    check_no_table_lost,
    check_no_truncated_tables,
    pct_tables_transcribed,
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


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_no_truncated_output(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """G2: the VLM produced a clean-but-short table the Docling grid had to rescue
    from truncation. A non-zero count means the decode is dropping rows on long
    tables (candidate for tiling / higher max_tokens). WARN for now."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    outcome = check_no_truncated_tables(blocks)
    truncated = outcome.metadata.get("vlm_truncated_tables", 0)
    total = outcome.metadata.get("table_blocks_total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"no truncated VLM tables across {total} table(s)"
            if outcome.passed
            else f"{truncated} of {total} table(s) truncated by the VLM — grid rescued the rows"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_no_failure_markers(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """G5: no ``[table/image processing failed]`` marker should reach the merged
    markdown RAG sees. WARN when one leaks but the table still survived via a
    fallback tier; ERROR-severity (still non-blocking during calibration) when a
    marker coincides with an actual unrecoverable table loss."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    md_path = paths.silver_modal_markdown_path(stem)
    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    outcome = check_no_failure_markers(markdown, blocks)
    markers = outcome.metadata["failure_markers"]
    lost = outcome.metadata["tables_lost"]
    severity = AssetCheckSeverity.ERROR if lost > 0 else AssetCheckSeverity.WARN
    return AssetCheckResult(
        passed=outcome.passed,
        severity=severity,
        description=(
            "no failure markers in merged markdown"
            if outcome.passed
            else f"{markers} failure marker(s) leaked into markdown ({lost} table(s) lost with no fallback)"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_confidence_floor(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """G4: fraction of modal results below the confidence floor. A high rate means
    vision transcription is broadly weak on this doc, not just one bad table.
    WARN above 25%."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    outcome = check_confidence_floor(blocks)
    low = outcome.metadata["low_confidence_blocks"]
    total = outcome.metadata["total_modal_blocks"]
    rate = outcome.metadata["low_confidence_rate"]
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"{low}/{total} modal result(s) low-confidence ({rate:.0%}, threshold "
            f"{outcome.metadata['max_low_confidence_rate']:.0%})"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_modal", blocking=False, partitions_def=stem_partitions)
def v6b_silver_modal_quality_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """G6 corpus regression: ``pct_tables_transcribed`` (share of tables accepted
    at the best VLM tier) vs the run-over-run baseline median. WARN on drift."""
    stem = context.partition_key
    blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))
    pct = pct_tables_transcribed(blocks)
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_modal",
        partition_key=stem,
        metadata_key="pct_tables_transcribed",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=pct, history=history, max_drift_pct=0.15)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="pct_tables_transcribed " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )
