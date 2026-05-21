"""Partition definitions. Single-dimension dynamic partitioning by syllabus
stem; dept is derived from the stem prefix (e.g. '202541_CSCE_605_600_62077'
→ 'CSCE'). Multi-dim was considered and rejected as redundant — every stem
already encodes its dept."""

from __future__ import annotations

from dagster import DynamicPartitionsDefinition

stem_partitions = DynamicPartitionsDefinition(name="syllabus_stem")
"""One partition per syllabus stem. New stems are registered by the scraper
sensor (Phase A: registered manually via dagster CLI)."""
