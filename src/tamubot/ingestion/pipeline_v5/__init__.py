"""Dagster-native v5 ingestion pipeline.

One asset per stage. Asset checks = QA gates. Dynamic partitions per syllabus
stem, per-dept multi-partitioned. `code_version` auto-derived from source hash.

Entrypoint: `pipeline_v5.definitions:defs`. Run with `dagster asset materialize`.
"""

from tamubot.ingestion.pipeline_v5.definitions import defs

__all__ = ["defs"]
