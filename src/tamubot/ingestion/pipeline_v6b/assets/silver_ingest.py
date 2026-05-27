"""silver_ingest: embed v6b_semantic chunks via Voyage and upsert into chunks_v4.

Default V6B_INGEST_ENABLED=false: chunks are still embedded to disk-cache
JSON via Voyage (to make dry-run repeatable) and the Atlas write is skipped.
Set V6B_INGEST_ENABLED=true to actually upsert.
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


def _embed_chunks_disk_cached(chunks: list[dict]) -> int:
    """Embed every chunk in-place via Voyage. Returns number of voyage calls made.

    Skips embedding when chunks already have embeddings (re-run after a prior
    pass is free).
    """
    needs = [c for c in chunks if c.get("embedding") is None]
    if not needs:
        return 0

    from tamubot.ingestion.ingest import embed_chunks, get_voyage_client

    voyage = get_voyage_client()
    embed_chunks(voyage, needs)
    return 1


def _upsert_atlas(chunks: list[dict], chunk_tag: str) -> int:
    """Bulk-upsert v6b_semantic chunks. Filter on (crn, chunk_index, chunk_tag)."""
    from pymongo import MongoClient, UpdateOne

    uri = os.getenv("MONGO_URI") or config.MONGO_URI
    db_name = os.getenv("MONGO_DB") or config.MONGO_DB
    client = MongoClient(uri)
    db = client[db_name]

    ops = []
    for doc in chunks:
        filt = {"crn": doc["crn"], "chunk_index": doc["chunk_index"], "chunk_tag": chunk_tag}
        ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))

    if not ops:
        return 0
    res = db["chunks_v4"].bulk_write(ops, ordered=False)
    return (res.modified_count or 0) + (res.upserted_count or 0)


def _compute_ingest(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src = paths.silver_tag_path(stem, "semantic")

    data = json.loads(src.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    voyage_calls = _embed_chunks_disk_cached(chunks)
    src.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    dry_run = not config.V6B_INGEST_ENABLED
    written = 0
    if not dry_run:
        written = _upsert_atlas(chunks, "v6b_semantic")

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "chunk_tag": "v6b_semantic",
            "chunk_count": len(chunks),
            "voyage_calls": voyage_calls,
            "atlas_upserted": written,
            "dry_run": dry_run,
        }
    )


silver_ingest = asset(
    name="v6b_silver_ingest",
    deps=[AssetKey("v6b_silver_tag_semantic")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_ingest),
    group_name="v6b_silver",
    description="Embed + upsert v6b_semantic chunks into chunks_v4. Dry-run by default.",
)(_compute_ingest)
