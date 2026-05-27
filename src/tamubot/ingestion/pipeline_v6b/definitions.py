"""Dagster Definitions entrypoint for pipeline_v6b.

Loaded by `dagster dev` and `dagster asset materialize -f
src/tamubot/ingestion/pipeline_v6b/definitions.py`.

v6b reads raw PDFs from v5's raw/ tree on disk — there's no v6b raw_pdf
asset. The bronze_blocks asset reads through `paths.raw_path(stem)` (which
resolves under v5/raw/) so v6b never copies PDFs.
"""

from dagster import Definitions

from tamubot.ingestion.pipeline_v6b.assets.bronze_blocks import (
    bronze_blocks,
    v6b_bronze_blocks_has_text,
    v6b_bronze_blocks_nonempty,
)
from tamubot.ingestion.pipeline_v6b.assets.silver_chunk_semantic import silver_chunk_semantic
from tamubot.ingestion.pipeline_v6b.assets.silver_ingest import silver_ingest
from tamubot.ingestion.pipeline_v6b.assets.silver_modal import (
    silver_modal,
    v6b_silver_modal_budget_not_exceeded,
)
from tamubot.ingestion.pipeline_v6b.assets.silver_tag import silver_tag_semantic
from tamubot.ingestion.pipeline_v6b.resources import DoclingConverterResource

defs = Definitions(
    assets=[
        bronze_blocks,
        silver_modal,
        silver_chunk_semantic,
        silver_tag_semantic,
        silver_ingest,
    ],
    asset_checks=[
        v6b_bronze_blocks_nonempty,
        v6b_bronze_blocks_has_text,
        v6b_silver_modal_budget_not_exceeded,
    ],
    resources={
        "docling": DoclingConverterResource(),
    },
)
