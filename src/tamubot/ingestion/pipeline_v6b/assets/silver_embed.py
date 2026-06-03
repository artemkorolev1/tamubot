"""silver_embed: embed v6b_semantic chunks via Voyage. Writes own output.

Split from the previous monolithic silver_ingest so embed and Atlas upsert
fail independently. Stamps `embedding_model` on each chunk so downstream
checks can verify the model version that produced each vector.
"""

import json

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.ingest import EMBEDDING_MODEL
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions


def _embed_chunks(chunks: list[dict]) -> int:
    """Embed every chunk in-place via Voyage; stamp embedding_model.

    Returns the number of chunks embedded this run (0 if all were already
    embedded) — a real count, so `voyage_calls_vs_baseline` is meaningful.
    """
    needs = [c for c in chunks if c.get("embedding") is None]
    if not needs:
        for c in chunks:
            c.setdefault("embedding_model", EMBEDDING_MODEL)
        return 0

    from tamubot.ingestion.ingest import embed_chunks, get_voyage_client

    voyage = get_voyage_client()
    embed_chunks(voyage, needs)
    for c in chunks:
        c["embedding_model"] = EMBEDDING_MODEL
    return len(needs)


def _compute_embed(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)

    src = paths.silver_tag_path(stem, "semantic")
    data = json.loads(src.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    voyage_calls = _embed_chunks(chunks)
    data["chunks"] = chunks

    out = paths.silver_embed_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "chunk_count": len(chunks),
            "voyage_calls": voyage_calls,
            "embedding_model": EMBEDDING_MODEL,
            "output_path": MetadataValue.path(str(out)),
            "sha256": hash_file(out),
        }
    )


silver_embed = asset(
    name="v6b_silver_embed",
    deps=[AssetKey("v6b_silver_tag_semantic")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_embed),
    group_name="v6b_silver",
    description="Embed v6b_semantic chunks via Voyage. Writes to silver/04_embed/.",
)(_compute_embed)
