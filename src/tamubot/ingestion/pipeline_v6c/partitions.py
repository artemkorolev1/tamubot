"""Partition definitions for v6c — one partition per stem.

Stems are discovered from the v6 bronze directory (same source set as the
bake-off). Mirror v6b's approach.
"""

from dagster import DynamicPartitionsDefinition

stem_partitions = DynamicPartitionsDefinition(name="v6c_stems")
