"""Dagster Definitions entrypoint for pipeline_v6c.

Loaded by `dagster dev -f src/tamubot/ingestion/pipeline_v6c/definitions.py`.

v6c reads raw PDFs from v5's raw/ tree on disk (see paths.raw_path) — no
v6c raw_pdf asset.
"""

from dagster import Definitions

from tamubot.ingestion.pipeline_v6c.assets.bronze_odl import bronze_odl

defs = Definitions(
    assets=[bronze_odl],
)
