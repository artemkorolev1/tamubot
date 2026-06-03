"""Asset checks for v6b_silver_structured.

Non-blocking:
  - not_degenerate: the TAMU gateway returns an all-null shape ~20% of the time
    even at temperature 0; after 3 retries silver_structured writes that shape and
    the partition still goes green. This surfaces it as a WARN instead of letting
    a content-free extract pass unnoticed.
"""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths


@asset_check(asset="v6b_silver_structured", blocking=False)
def v6b_silver_structured_not_degenerate(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    data = json.loads(paths.silver_structured_path(stem).read_text(encoding="utf-8"))
    course_code = (data.get("course_code") or "").strip()
    instructor = (data.get("instructor_name") or "").strip()
    has_identity = bool(course_code or instructor)
    return AssetCheckResult(
        passed=has_identity,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "course_code": course_code,
            "instructor_name": instructor,
            "degenerate": not has_identity,
        },
    )
