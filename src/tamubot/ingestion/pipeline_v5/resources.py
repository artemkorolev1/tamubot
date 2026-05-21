"""Dagster resources injected into asset compute functions.

DoclingConverterResource holds the expensive-to-create DocumentConverter so it's
loaded once per process, not once per asset materialization."""

from __future__ import annotations

from dagster import ConfigurableResource

# Lazy import so `dagster dev` introspection doesn't pay Docling's startup cost.


class DoclingConverterResource(ConfigurableResource):
    """Wraps Docling's DocumentConverter (cached per resource instance)."""

    _converter: object = None

    def get_converter(self):
        if self._converter is None:
            from tamubot.ingestion.converters.docling_converter import create_converter

            object.__setattr__(self, "_converter", create_converter())
        return self._converter
