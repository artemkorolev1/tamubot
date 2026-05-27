"""silver_tag: boilerplate tagging via cosine sim against the v6 reference set.

Phase 1 default: reference parquet is not yet built (v6 plan Step 3 pending).
In that case this asset is a no-op pass-through — every chunk keeps
is_boilerplate=False. The asset materializes anyway so the downstream ingest
asset has a stable dependency, and metadata flags the deferred state.
"""

import json

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions


def _tag_chunks(chunks: list[dict], reference_available: bool) -> tuple[list[dict], dict]:
    """Returns (tagged_chunks, stats). Pure function — easy to test."""
    tagged_true = 0
    for c in chunks:
        if not reference_available:
            c["is_boilerplate"] = False
            continue
        # TODO Phase 1+: cosine sim against boilerplate_reference.parquet centroids.
        c["is_boilerplate"] = False
    return chunks, {"tagged_boilerplate": tagged_true}


def _compute_silver_tag(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)

    src_path = paths.silver_chunk_semantic_path(stem)
    dst_path = paths.silver_tag_path(stem, "semantic")

    data = json.loads(src_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    reference_parquet = paths.boilerplate_reference_path()
    reference_available = reference_parquet.exists()

    tagged, stats = _tag_chunks(chunks, reference_available=reference_available)
    data["chunks"] = tagged

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "reference_set_available": reference_available,
            "tagging_deferred": not reference_available,
            "tagged_boilerplate": stats["tagged_boilerplate"],
            "total_chunks": len(tagged),
            "output_path": MetadataValue.path(str(dst_path)),
            "sha256": hash_file(dst_path),
        }
    )


silver_tag_semantic = asset(
    name="v6b_silver_tag_semantic",
    deps=[AssetKey("v6b_silver_chunk_semantic")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_tag),
    group_name="v6b_silver",
    description="Boilerplate tag for semantic chunks. No-op when reference set missing.",
)(_compute_silver_tag)
