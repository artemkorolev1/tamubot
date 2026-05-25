"""Dagster Definitions entrypoint. Loaded by `dagster dev` and
`dagster asset materialize`."""

from __future__ import annotations

from dagster import Definitions

from tamubot.ingestion.pipeline_v5.assets.bronze import (
    bronze_assets,
    bronze_headers_sidecar_nonempty,
    bronze_markdown_min_headers,
    bronze_markdown_min_tokens,
    bronze_markdown_no_replacement_chars,
)
from tamubot.ingestion.pipeline_v5.assets.raw import raw_pdf
from tamubot.ingestion.pipeline_v5.assets.silver_boilerplate import silver_boilerplate
from tamubot.ingestion.pipeline_v5.assets.silver_chunk import silver_chunk
from tamubot.ingestion.pipeline_v5.assets.silver_enrich import silver_enrich
from tamubot.ingestion.pipeline_v5.assets.silver_false_positive import silver_false_positive
from tamubot.ingestion.pipeline_v5.assets.silver_image_recovery import (
    silver_image_recovery,
    silver_image_recovery_no_hallucination,
)
from tamubot.ingestion.pipeline_v5.assets.silver_relocate_textbook import silver_relocate_textbook
from tamubot.ingestion.pipeline_v5.resources import DoclingConverterResource

defs = Definitions(
    assets=[
        raw_pdf,
        bronze_assets,
        silver_image_recovery,
        silver_false_positive,
        silver_boilerplate,
        silver_relocate_textbook,
        silver_enrich,
        silver_chunk,
    ],
    asset_checks=[
        bronze_markdown_min_tokens,
        bronze_markdown_min_headers,
        bronze_markdown_no_replacement_chars,
        bronze_headers_sidecar_nonempty,
        silver_image_recovery_no_hallucination,
    ],
    resources={
        "docling": DoclingConverterResource(),
    },
)
