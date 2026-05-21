"""Dagster assets for pipeline_v5.

One asset per stage in the medallion flow:
  raw_pdf → bronze_markdown + bronze_headers_sidecar → silver/* → gold

Phase A scope: raw + bronze only.
"""
