"""Dagster resources for pipeline_v6b.

Reuses v5's DoclingConverterResource (same converter, just imported here so
the v6b Definitions doesn't have to reach across pipeline boundaries).

NuExtractResource holds the 4-bit NuExtract3 model + processor, loaded once
per process (the load is ~3 min + a few GB of VRAM, so never per-partition).

The vision callable is built lazily in silver_modal — we do NOT hold a
genai client on the resource because most asset runs (bronze, chunk, tag,
ingest) never call vision and we don't want to instantiate the client unless
modal is enabled.
"""

from __future__ import annotations

from dagster import ConfigurableResource

from tamubot.ingestion.pipeline_v5.resources import DoclingConverterResource

__all__ = ["DoclingConverterResource", "NuExtractResource"]


class NuExtractResource(ConfigurableResource):
    """Holds the NuExtract3 extractor (NF4 4-bit), cached per resource instance."""

    quantize: bool = True
    _extractor: object = None

    def get_extractor(self):
        if self._extractor is None:
            from tamubot.ingestion.clients.nuextract_client import get_extractor

            object.__setattr__(self, "_extractor", get_extractor(quantize=self.quantize))
        return self._extractor
