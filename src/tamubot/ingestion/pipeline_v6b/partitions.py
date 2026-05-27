"""Partition definitions for v6b. Re-exports the v5 stem partition so the same
syllabus partitions feed both pipelines without duplication."""

from __future__ import annotations

from tamubot.ingestion.pipeline_v5.partitions import stem_partitions

__all__ = ["stem_partitions"]
