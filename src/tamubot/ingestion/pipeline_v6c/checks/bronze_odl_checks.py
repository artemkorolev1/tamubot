"""Asset checks for v6c_bronze_markdown + v6c_bronze_headers_sidecar."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6c import paths
from tamubot.ingestion.validation.header_hierarchy import (
    check_header_hierarchy_valid,
    check_min_headers,
)
from tamubot.ingestion.validation.text_quality import (
    check_letter_drops,
    check_no_replacement_chars,
)

MIN_MARKDOWN_BYTES = 500


def _read_md(stem: str) -> str:
    return paths.bronze_md_path(stem).read_text(encoding="utf-8")


def _read_headers(stem: str) -> list:
    raw = json.loads(paths.bronze_sidecar_path(stem).read_text(encoding="utf-8"))
    return raw.get("headers", [])


@asset_check(asset="v6c_bronze_markdown", blocking=True)
def v6c_bronze_markdown_nonempty(context: AssetCheckExecutionContext) -> AssetCheckResult:
    stem = context.partition_key
    md = _read_md(stem)
    size = len(md.encode("utf-8"))
    return AssetCheckResult(
        passed=size >= MIN_MARKDOWN_BYTES,
        severity=AssetCheckSeverity.ERROR,
        metadata={"byte_size": size, "min_required": MIN_MARKDOWN_BYTES},
    )


@asset_check(asset="v6c_bronze_markdown", blocking=True)
def v6c_bronze_markdown_no_replacement_chars(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    outcome = check_no_replacement_chars(_read_md(context.partition_key))
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6c_bronze_markdown", blocking=True)
def v6c_bronze_markdown_no_letter_drops(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    outcome = check_letter_drops(_read_md(context.partition_key), threshold=0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6c_bronze_markdown", blocking=True)
def v6c_bronze_markdown_min_headers(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    headers = _read_headers(context.partition_key)
    outcome = check_min_headers(headers, minimum=2)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )


@asset_check(asset="v6c_bronze_headers_sidecar", blocking=True)
def v6c_bronze_sidecar_nonempty(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    headers = _read_headers(context.partition_key)
    return AssetCheckResult(
        passed=len(headers) > 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"header_count": len(headers)},
    )


@asset_check(asset="v6c_bronze_headers_sidecar", blocking=True)
def v6c_bronze_sidecar_hierarchy_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    outcome = check_header_hierarchy_valid(_read_headers(context.partition_key))
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        metadata=outcome.metadata,
    )
