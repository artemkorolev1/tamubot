"""Dagster Definitions entrypoint for pipeline_v6b.

Loaded by `dagster dev` and `dagster asset materialize -f
src/tamubot/ingestion/pipeline_v6b/definitions.py`.

v6b reads raw PDFs from v5's raw/ tree on disk — there's no v6b raw_pdf
asset. The bronze_blocks asset reads through `paths.raw_path(stem)` (which
resolves under v5/raw/) so v6b never copies PDFs.
"""

from dagster import Definitions, in_process_executor

from tamubot.ingestion.pipeline_v6b.assets.bronze_blocks import bronze_blocks
from tamubot.ingestion.pipeline_v6b.assets.corpus_report import v6b_corpus_report
from tamubot.ingestion.pipeline_v6b.assets.silver_atlas_upsert import silver_atlas_upsert
from tamubot.ingestion.pipeline_v6b.assets.silver_chunk_semantic import silver_chunk_semantic
from tamubot.ingestion.pipeline_v6b.assets.silver_embed import silver_embed
from tamubot.ingestion.pipeline_v6b.assets.silver_modal import silver_modal
from tamubot.ingestion.pipeline_v6b.assets.silver_structured import silver_structured
from tamubot.ingestion.pipeline_v6b.assets.silver_tag import silver_tag_semantic
from tamubot.ingestion.pipeline_v6b.checks.bronze_blocks_checks import (
    v6b_bronze_blocks_block_count_vs_baseline,
    v6b_bronze_blocks_has_text,
    v6b_bronze_blocks_header_hierarchy_valid,
    v6b_bronze_blocks_no_replacement_chars,
    v6b_bronze_blocks_nonempty,
    v6b_bronze_blocks_source_integrity,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_atlas_upsert_checks import (
    v6b_silver_atlas_golden_recall_at_5,
    v6b_silver_atlas_index_size_vs_baseline,
    v6b_silver_atlas_index_status_ready,
    v6b_silver_atlas_vector_count_matches_chunks,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_chunk_checks import (
    v6b_silver_chunk_count_nonzero,
    v6b_silver_chunk_flagged_rate_vs_baseline,
    v6b_silver_chunk_low_no_header_rate,
    v6b_silver_chunk_no_oversized,
    v6b_silver_chunk_schema_valid,
    v6b_silver_chunk_total_vs_baseline,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_embed_checks import (
    v6b_silver_embed_count_matches_chunks,
    v6b_silver_embed_model_field_present,
    v6b_silver_embed_voyage_calls_vs_baseline,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_modal_checks import (
    v6b_silver_modal_budget_not_exceeded,
    v6b_silver_modal_result_schema_valid,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_tag_checks import (
    v6b_silver_tag_chunk_count_preserved,
)
from tamubot.ingestion.pipeline_v6b.resources import DoclingConverterResource, NuExtractResource
from tamubot.ingestion.pipeline_v6b.sensors import v6b_check_failure_alert

defs = Definitions(
    assets=[
        bronze_blocks,
        silver_modal,
        silver_chunk_semantic,
        silver_tag_semantic,
        silver_embed,
        silver_atlas_upsert,
        silver_structured,
        v6b_corpus_report,
    ],
    asset_checks=[
        v6b_bronze_blocks_nonempty,
        v6b_bronze_blocks_has_text,
        v6b_bronze_blocks_no_replacement_chars,
        v6b_bronze_blocks_header_hierarchy_valid,
        v6b_bronze_blocks_source_integrity,
        v6b_silver_modal_budget_not_exceeded,
        v6b_silver_modal_result_schema_valid,
        v6b_silver_chunk_count_nonzero,
        v6b_silver_chunk_low_no_header_rate,
        v6b_silver_chunk_no_oversized,
        v6b_silver_chunk_schema_valid,
        v6b_silver_tag_chunk_count_preserved,
        v6b_silver_embed_count_matches_chunks,
        v6b_silver_embed_model_field_present,
        v6b_silver_atlas_vector_count_matches_chunks,
        v6b_silver_atlas_index_status_ready,
        v6b_bronze_blocks_block_count_vs_baseline,
        v6b_silver_chunk_total_vs_baseline,
        v6b_silver_chunk_flagged_rate_vs_baseline,
        v6b_silver_embed_voyage_calls_vs_baseline,
        v6b_silver_atlas_index_size_vs_baseline,
        v6b_silver_atlas_golden_recall_at_5,
    ],
    sensors=[v6b_check_failure_alert],
    resources={
        "docling": DoclingConverterResource(),
        "nuextract": NuExtractResource(),
    },
    # GPU-bound stages (NuExtract, modal vision) must not run as parallel
    # subprocesses on a single 8 GB card — they'd each load a model copy and OOM.
    executor=in_process_executor,
)
