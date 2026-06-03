"""v6b_meta_boilerplate_reference: corpus-scanning asset that builds the
shared boilerplate reference parquet. Not partitioned. Materialize manually:

    dagster asset materialize --select v6b_meta_boilerplate_reference \\
        -f src/tamubot/ingestion/pipeline_v6b/definitions.py

Pure clustering logic lives in util/boilerplate_clustering.py.
"""

from pathlib import Path

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.util.boilerplate_clustering import (
    LSH_THRESHOLD,
    MIN_DISTINCT_DEPTS,
    MIN_DOC_FREQUENCY,
    build_reference_df,
    cluster_chunks,
    scan_corpus_chunks,
)


def _compute(context: AssetExecutionContext) -> MaterializeResult:
    rows = scan_corpus_chunks(Path(DATA_ROOT))
    clusters = cluster_chunks(rows)
    df = build_reference_df(clusters)

    out_path = paths.boilerplate_reference_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    context.log.info(
        "v6b_meta_boilerplate_reference: %d chunks scanned, %d clusters total, %d kept",
        len(rows),
        len(clusters),
        len(df),
    )

    return MaterializeResult(
        metadata={
            "chunks_scanned": len(rows),
            "clusters_total": len(clusters),
            "clusters_kept": len(df),
            "min_doc_frequency": MIN_DOC_FREQUENCY,
            "min_distinct_depts": MIN_DISTINCT_DEPTS,
            "lsh_threshold": LSH_THRESHOLD,
            "output_path": str(out_path),
        }
    )


v6b_meta_boilerplate_reference = asset(
    name="v6b_meta_boilerplate_reference",
    # Corpus scan over all chunk partitions — declare the real upstream so the
    # graph shows this asset between chunk_semantic and silver_tag (not as an
    # orphan island). Non-loadable dep: scan_corpus_chunks reads the chunk
    # outputs off disk, so no I/O-manager wiring is needed.
    deps=[AssetKey("v6b_silver_chunk_semantic")],
    group_name="v6b_meta",
    description=(
        "Corpus-scanning: builds the shared boilerplate reference parquet "
        f"(doc_frequency >= {MIN_DOC_FREQUENCY} AND distinct_depts >= {MIN_DISTINCT_DEPTS}). "
        "Not partitioned. Materialize manually."
    ),
)(_compute)
