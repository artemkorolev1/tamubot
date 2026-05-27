"""Asset checks for v6b_silver_tag_semantic. Today the asset is a no-op
pass-through; the chunk_count_preserved check catches silent loss when it
is no longer a pass-through."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths


@asset_check(asset="v6b_silver_tag_semantic", blocking=True)
def v6b_silver_tag_chunk_count_preserved(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    src = json.loads(paths.silver_chunk_semantic_path(stem).read_text(encoding="utf-8"))
    dst = json.loads(paths.silver_tag_path(stem, "semantic").read_text(encoding="utf-8"))
    n_in = len(src["chunks"])
    n_out = len(dst["chunks"])
    return AssetCheckResult(
        passed=n_in == n_out,
        severity=AssetCheckSeverity.ERROR,
        metadata={"chunks_in": n_in, "chunks_out": n_out, "delta": n_out - n_in},
    )
