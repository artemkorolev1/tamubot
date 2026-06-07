"""Asset checks for v6b_silver_atlas_upsert.

Skipped (passed=True with metadata flag) when V6B_INGEST_ENABLED=false — dry-run
materializations should not fail just because Atlas wasn't touched.
"""

import json
import os
from pathlib import Path

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)
from pymongo import MongoClient

from tamubot.core import config
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.validation.baseline_diff import (
    compute_baseline_delta,
    read_metadata_history,
)
from tamubot.ingestion.validation.golden_recall import compute_recall_at_k
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


def _load_curated20() -> list[dict]:
    """Load the canonical curated20 golden set. Returns [] if file is absent."""
    p = Path("data/_meta/curated20.jsonl")
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atlas_retrieve(question: str, k: int) -> list[str]:
    """Embed question via Voyage, query Atlas vector_search, return chunk ids."""
    from tamubot.ingestion.ingest import EMBEDDING_MODEL, get_voyage_client

    coll = _atlas_collection()
    voyage = get_voyage_client()
    q_vec = voyage.embed([question], model=EMBEDDING_MODEL, input_type="query").embeddings[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": q_vec,
                "numCandidates": k * 10,
                "limit": k,
            }
        },
        {"$project": {"_id": 0, "crn": 1, "chunk_index": 1, "chunk_tag": 1}},
    ]
    return [f"{d['crn']}::{d['chunk_index']}::{d['chunk_tag']}" for d in coll.aggregate(pipeline)]


@asset_check(asset="v6b_silver_atlas_upsert", blocking=True, partitions_def=stem_partitions)
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


@asset_check(asset="v6b_silver_atlas_upsert", blocking=False, partitions_def=stem_partitions)
def v6b_silver_atlas_index_size_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    if not config.V6B_INGEST_ENABLED:
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            metadata={"skipped_reason": "V6B_INGEST_ENABLED=false (dry-run)"},
        )
    stem = context.partition_key
    coll = _atlas_collection()
    current = coll.count_documents({"source_file": stem, "chunk_tag": "v6b_semantic"})
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_atlas_upsert",
        partition_key=stem,
        metadata_key="atlas_upserted",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=current, history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_atlas_upsert", blocking=False)
def v6b_silver_atlas_golden_recall_at_5(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """L3 semantic check — opt-in via RUN_GOLDEN_RECALL_CHECK=true.

    Costs: 1 Voyage embed + 1 Atlas vector_search per golden query (~20 of each
    per run). Off by default to keep CI free.
    """
    if os.getenv("RUN_GOLDEN_RECALL_CHECK", "").lower() != "true":
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            metadata={"skipped_reason": "RUN_GOLDEN_RECALL_CHECK not set to true"},
        )
    if not config.V6B_INGEST_ENABLED:
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            metadata={"skipped_reason": "V6B_INGEST_ENABLED=false (dry-run)"},
        )
    queries = _load_curated20()
    outcome = compute_recall_at_k(queries, retrieve_fn=_atlas_retrieve, k=5, min_recall=0.80)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        metadata=outcome.metadata,
    )
