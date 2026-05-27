"""Asset checks for v6b_silver_embed."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.ingest import EMBEDDING_MODEL
from tamubot.ingestion.pipeline_v6b import paths


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
