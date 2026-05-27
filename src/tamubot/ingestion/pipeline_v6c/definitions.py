"""Dagster Definitions entrypoint for pipeline_v6c.

Loaded by `dagster dev -f src/tamubot/ingestion/pipeline_v6c/definitions.py`.

v6c reads raw PDFs from v5's raw/ tree on disk (see paths.raw_path) — no
v6c raw_pdf asset.
"""

from dagster import Definitions

from tamubot.ingestion.pipeline_v6c.assets.bronze_odl import bronze_odl
from tamubot.ingestion.pipeline_v6c.checks.bronze_odl_checks import (
    v6c_bronze_markdown_letter_drops_vs_baseline,
    v6c_bronze_markdown_min_headers,
    v6c_bronze_markdown_no_letter_drops,
    v6c_bronze_markdown_no_replacement_chars,
    v6c_bronze_markdown_nonempty,
    v6c_bronze_markdown_token_count_vs_baseline,
    v6c_bronze_sidecar_header_count_vs_baseline,
    v6c_bronze_sidecar_hierarchy_valid,
    v6c_bronze_sidecar_nonempty,
)

defs = Definitions(
    assets=[bronze_odl],
    asset_checks=[
        v6c_bronze_markdown_nonempty,
        v6c_bronze_markdown_no_replacement_chars,
        v6c_bronze_markdown_no_letter_drops,
        v6c_bronze_markdown_min_headers,
        v6c_bronze_sidecar_nonempty,
        v6c_bronze_sidecar_hierarchy_valid,
        v6c_bronze_markdown_token_count_vs_baseline,
        v6c_bronze_markdown_letter_drops_vs_baseline,
        v6c_bronze_sidecar_header_count_vs_baseline,
    ],
)
