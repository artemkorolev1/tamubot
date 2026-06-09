"""Dagster Definitions entrypoint for pipeline_v6b.

Loaded by `dagster dev` and `dagster asset materialize -f
src/tamubot/ingestion/pipeline_v6b/definitions.py`.

v6b reads raw PDFs from v5's raw/ tree on disk — there's no v6b raw_pdf
asset. The bronze_blocks asset reads through `paths.raw_path(stem)` (which
resolves under v5/raw/) so v6b never copies PDFs.
"""

from dagster import AssetSelection, Definitions, define_asset_job, in_process_executor

from tamubot.ingestion.pipeline_v6b.assets.bronze_blocks import bronze_blocks
from tamubot.ingestion.pipeline_v6b.assets.corpus_report import v6b_corpus_report
from tamubot.ingestion.pipeline_v6b.assets.meta_boilerplate_reference import (
    v6b_meta_boilerplate_reference,
)
from tamubot.ingestion.pipeline_v6b.assets.meta_chunk_signature_index import (
    v6b_meta_chunk_signature_index,
)
from tamubot.ingestion.pipeline_v6b.assets.pipeline_ledger import v6b_pipeline_ledger
from tamubot.ingestion.pipeline_v6b.assets.preprocessing_judge_report import (
    v6b_preprocessing_judge_report,
)
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
    v6b_bronze_blocks_header_levels_normalized,
    v6b_bronze_blocks_heading_repair_vs_baseline,
    v6b_bronze_blocks_min_headers,
    v6b_bronze_blocks_no_orphaned_labels,
    v6b_bronze_blocks_no_replacement_chars,
    v6b_bronze_blocks_nonempty,
    v6b_bronze_blocks_source_integrity,
    v6b_bronze_blocks_suspicious_heading_rate,
    v6b_bronze_blocks_table_cell_capture,
    v6b_bronze_blocks_text_coverage,
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
    v6b_silver_modal_confidence_floor,
    v6b_silver_modal_no_degenerate_tables,
    v6b_silver_modal_no_failure_markers,
    v6b_silver_modal_no_table_lost,
    v6b_silver_modal_no_truncated_output,
    v6b_silver_modal_quality_vs_baseline,
    v6b_silver_modal_result_schema_valid,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_structured_checks import (
    v6b_silver_structured_not_degenerate,
)
from tamubot.ingestion.pipeline_v6b.checks.silver_tag_checks import (
    v6b_silver_tag_boilerplate_rate_in_band,
    v6b_silver_tag_chunk_count_preserved,
    v6b_silver_tag_duplicate_rate_in_band,
)
from tamubot.ingestion.pipeline_v6b.resources import DoclingConverterResource, NuExtractResource
from tamubot.ingestion.pipeline_v6b.sensors import v6b_check_failure_alert

v6b_pilot_job = define_asset_job(
    "v6b_pilot_job",
    selection=AssetSelection.assets(
        "v6b_bronze_blocks",
        "v6b_silver_modal",
        "v6b_silver_chunk_semantic",
        "v6b_silver_tag_semantic",
        "v6b_silver_embed",
        "v6b_silver_atlas_upsert",
        "v6b_silver_structured",
    ),
    description="Per-syllabus partitioned pipeline. Backfill this to materialize all 7 stages in one launch.",
)

# Tag-only job so PASS 2 (boilerplate + cross-syllabus dedup) can be launched as ONE
# backfill over a cohort — giving a native Backfills page that groups every file's
# tag run + check status in one view (vs. per-stem `asset materialize`, which scatters
# them into separate "jobless" runs). Tag is CPU-only, so a backfill is GPU-safe.
# MUST run AFTER the corpus meta indexes are (re)built, or dedup sees an empty index
# (project-v6b-dedup-two-phase). See scripts/v6b_tag_backfill.sh.
v6b_tag_job = define_asset_job(
    "v6b_tag_job",
    selection=AssetSelection.assets("v6b_silver_tag_semantic"),
    description="PASS 2 — boilerplate + cross-syllabus dedup tagging. Backfill a cohort AFTER meta indexes build.",
)

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
        v6b_meta_boilerplate_reference,
        v6b_meta_chunk_signature_index,
        v6b_preprocessing_judge_report,
        v6b_pipeline_ledger,
    ],
    asset_checks=[
        v6b_bronze_blocks_nonempty,
        v6b_bronze_blocks_has_text,
        v6b_bronze_blocks_no_replacement_chars,
        v6b_bronze_blocks_header_hierarchy_valid,
        v6b_bronze_blocks_header_levels_normalized,
        v6b_bronze_blocks_min_headers,
        v6b_bronze_blocks_source_integrity,
        v6b_bronze_blocks_suspicious_heading_rate,
        v6b_bronze_blocks_table_cell_capture,
        v6b_bronze_blocks_no_orphaned_labels,
        v6b_bronze_blocks_text_coverage,
        v6b_silver_modal_budget_not_exceeded,
        v6b_silver_modal_result_schema_valid,
        v6b_silver_modal_no_table_lost,
        v6b_silver_modal_no_degenerate_tables,
        v6b_silver_modal_no_truncated_output,
        v6b_silver_modal_no_failure_markers,
        v6b_silver_modal_confidence_floor,
        v6b_silver_chunk_count_nonzero,
        v6b_silver_chunk_low_no_header_rate,
        v6b_silver_chunk_no_oversized,
        v6b_silver_chunk_schema_valid,
        v6b_silver_structured_not_degenerate,
        v6b_silver_tag_chunk_count_preserved,
        v6b_silver_tag_boilerplate_rate_in_band,
        v6b_silver_tag_duplicate_rate_in_band,
        v6b_silver_embed_count_matches_chunks,
        v6b_silver_embed_model_field_present,
        v6b_silver_atlas_vector_count_matches_chunks,
        v6b_silver_atlas_index_status_ready,
        v6b_bronze_blocks_block_count_vs_baseline,
        v6b_bronze_blocks_heading_repair_vs_baseline,
        v6b_silver_modal_quality_vs_baseline,
        v6b_silver_chunk_total_vs_baseline,
        v6b_silver_chunk_flagged_rate_vs_baseline,
        v6b_silver_embed_voyage_calls_vs_baseline,
        v6b_silver_atlas_index_size_vs_baseline,
        v6b_silver_atlas_golden_recall_at_5,
    ],
    jobs=[v6b_pilot_job, v6b_tag_job],
    sensors=[v6b_check_failure_alert],
    resources={
        "docling": DoclingConverterResource(),
        "nuextract": NuExtractResource(),
    },
    # GPU-bound stages (NuExtract, modal vision) must not run as parallel
    # subprocesses on a single 8 GB card — they'd each load a model copy and OOM.
    executor=in_process_executor,
)
