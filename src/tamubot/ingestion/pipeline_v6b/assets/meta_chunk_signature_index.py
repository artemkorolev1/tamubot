"""v6b_meta_chunk_signature_index: corpus-scanning asset that writes a parquet
of per-chunk MinHash signatures, consumed by silver_tag's cross-syllabus dedup
pass. Not partitioned. Materialize manually:

    dagster asset materialize --select v6b_meta_chunk_signature_index \\
        -f src/tamubot/ingestion/pipeline_v6b/definitions.py

Pure scan/build logic lives in util/signature_index.py.
"""

from pathlib import Path

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.util.signature_index import (
    DEFAULT_CROSS_SYL_THRESHOLD,
    NUM_PERM,
    scan_corpus_signatures,
)


def _compute(context: AssetExecutionContext) -> MaterializeResult:
    df = scan_corpus_signatures(Path(DATA_ROOT))

    out_path = paths.chunk_signature_index_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    distinct_stems = df["stem"].nunique() if len(df) else 0
    context.log.info(
        "v6b_meta_chunk_signature_index: %d chunks indexed across %d stems",
        len(df),
        distinct_stems,
    )

    return MaterializeResult(
        metadata={
            "chunks_indexed": len(df),
            "distinct_stems": distinct_stems,
            "num_perm": NUM_PERM,
            "cross_syllabus_threshold": DEFAULT_CROSS_SYL_THRESHOLD,
            "output_path": str(out_path),
        }
    )


v6b_meta_chunk_signature_index = asset(
    name="v6b_meta_chunk_signature_index",
    # Corpus scan over all chunk partitions — same rationale as
    # v6b_meta_boilerplate_reference: declare the real upstream so lineage is
    # correct. Non-loadable dep (scan_corpus_signatures reads off disk).
    deps=[AssetKey("v6b_silver_chunk_semantic")],
    group_name="v6b_meta",
    description=(
        "Corpus-scanning: per-chunk MinHash signature index for cross-syllabus dedup. "
        "Not partitioned. Materialize manually."
    ),
)(_compute)
