"""silver_atlas_upsert: upsert embedded chunks into Atlas chunks_v4.

Reads from silver_embed (not silver_tag). Dry-run default unchanged
(V6B_INGEST_ENABLED=false skips the Atlas write but still materializes).
"""

import json
import os

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    asset,
)

from tamubot.core import config
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions


def _upsert_atlas(chunks: list[dict], chunk_tag: str) -> int:
    from pymongo import MongoClient, UpdateOne

    uri = os.getenv("MONGODB_URI") or config.MONGODB_URI
    db_name = os.getenv("MONGODB_DB") or config.MONGODB_DB
    client: MongoClient = MongoClient(uri)
    db = client[db_name]

    ops = []
    for doc in chunks:
        filt = {"crn": doc["crn"], "chunk_index": doc["chunk_index"], "chunk_tag": chunk_tag}
        ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))
    if not ops:
        return 0
    res = db["chunks_v4"].bulk_write(ops, ordered=False)
    return (res.modified_count or 0) + (res.upserted_count or 0)


def _compute_upsert(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src = paths.silver_embed_path(stem)

    data = json.loads(src.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    dry_run = not config.V6B_INGEST_ENABLED
    written = 0 if dry_run else _upsert_atlas(chunks, "v6b_semantic")

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "chunk_tag": "v6b_semantic",
            "chunk_count": len(chunks),
            "atlas_upserted": written,
            "dry_run": dry_run,
        }
    )


silver_atlas_upsert = asset(
    name="v6b_silver_atlas_upsert",
    deps=[AssetKey("v6b_silver_embed")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_upsert),
    group_name="v6b_gold",
    description="Upsert embedded v6b_semantic chunks into Atlas chunks_v4. Dry-run default.",
)(_compute_upsert)
