"""Dagster resources for pipeline_v6b.

Reuses v5's DoclingConverterResource (same converter, just imported here so
the v6b Definitions doesn't have to reach across pipeline boundaries).

The vision callable is built lazily in silver_modal — we do NOT hold a
genai client on the resource because most asset runs (bronze, chunk, tag,
ingest) never call vision and we don't want to instantiate the client unless
modal is enabled.
"""

from __future__ import annotations

from tamubot.ingestion.pipeline_v5.resources import DoclingConverterResource

__all__ = ["DoclingConverterResource"]
