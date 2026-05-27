"""Asset checks for v6b_silver_atlas_upsert.

Skipped (passed=True with metadata flag) when V6B_INGEST_ENABLED=false — dry-run
materializations should not fail just because Atlas wasn't touched.
"""

import json
import os

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)
from pymongo import MongoClient

from tamubot.core import config
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.validation.index_health import (
    check_atlas_index_ready,
    check_vector_count_matches_chunks,
)

INDEX_NAME = "vector_index_v4"


def _atlas_collection():
    uri = os.getenv("MONGODB_URI") or config.MONGODB_URI
    db_name = os.getenv("MONGODB_DB") or config.MONGODB_DB
    return MongoClient(uri)[db_name]["chunks_v4"]


def _chunks_for(stem: str) -> list:
    return json.loads(paths.silver_embed_path(stem).read_text(encoding="utf-8"))["chunks"]


@asset_check(asset="v6b_silver_atlas_upsert", blocking=True)
def v6b_silver_atlas_vector_count_matches_chunks(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    if not config.V6B_INGEST_ENABLED:
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            metadata={"skipped_reason": "V6B_INGEST_ENABLED=false (dry-run)"},
        )
    stem = context.partition_key
    chunks = _chunks_for(stem)
    coll = _atlas_collection()
    outcome = check_vector_count_matches_chunks(
        coll,
        filter_query={"source_file": stem, "chunk_tag": "v6b_semantic"},
        expected_count=len(chunks),
    )
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_atlas_upsert", blocking=False)
def v6b_silver_atlas_index_status_ready(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """WARN (not ERROR) — Atlas index can briefly be BUILDING after upsert."""
    coll = _atlas_collection()
    outcome = check_atlas_index_ready(coll, index_name=INDEX_NAME)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        metadata=outcome.metadata,
    )
