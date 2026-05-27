"""Asset checks for v6b_silver_embed."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetKey,
    DagsterEventType,
    EventRecordsFilter,
    asset_check,
)

from tamubot.ingestion.ingest import EMBEDDING_MODEL
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.validation.baseline_diff import (
    compute_baseline_delta,
    read_metadata_history,
)


def _load_embed(stem: str) -> dict:
    return json.loads(paths.silver_embed_path(stem).read_text(encoding="utf-8"))


@asset_check(asset="v6b_silver_embed", blocking=True)
def v6b_silver_embed_count_matches_chunks(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    data = _load_embed(stem)
    chunks = data["chunks"]
    embedded = sum(1 for c in chunks if c.get("embedding") is not None)
    total = len(chunks)
    return AssetCheckResult(
        passed=embedded == total and total > 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"embedded": embedded, "total": total},
    )


@asset_check(asset="v6b_silver_embed", blocking=True)
def v6b_silver_embed_model_field_present(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    data = _load_embed(stem)
    chunks = data["chunks"]
    missing = sum(1 for c in chunks if c.get("embedding_model") != EMBEDDING_MODEL)
    return AssetCheckResult(
        passed=missing == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "expected_model": EMBEDDING_MODEL,
            "missing_or_wrong_model": missing,
            "total": len(chunks),
        },
    )


@asset_check(asset="v6b_silver_embed", blocking=False)
def v6b_silver_embed_voyage_calls_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_embed",
        partition_key=stem,
        metadata_key="voyage_calls",
        last_n=5,
    )
    recent = context.instance.get_event_records(
        EventRecordsFilter(
            event_type=DagsterEventType.ASSET_MATERIALIZATION,
            asset_key=AssetKey("v6b_silver_embed"),
            asset_partitions=[stem],
        ),
        limit=1,
        ascending=False,
    )
    if not recent:
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            metadata={"skipped": "no materialization found"},
        )
    mat = recent[0].event_log_entry.dagster_event.event_specific_data.materialization
    voyage_calls = float(mat.metadata["voyage_calls"].value)
    outcome = compute_baseline_delta(current=voyage_calls, history=history, max_drift_pct=0.50)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        metadata=outcome.metadata,
    )
