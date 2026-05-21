"""silver_image_recovery: recover content from `<!-- image -->` placeholders
via Gemini multimodal.

Reuses v4's `_recover_images()` core. Asset wraps that call with v5's
stem-based partitioning and per-dept silver path layout.
"""

import shutil

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from tamubot.ingestion.filters.image_recovery import (
    IMAGE_MARKER,
    MIN_MARKERS,
    _count_markers,
    _recover_images,
)
from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file


def _compute_silver_image_recovery(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)

    bronze_md = paths.bronze_md_path(stem)
    raw_pdf = paths.raw_path(stem)
    out_md = paths.silver_path(stem, "image_recovery")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    text = bronze_md.read_text(encoding="utf-8")
    markers_before = _count_markers(text)

    if markers_before < MIN_MARKERS:
        shutil.copy2(bronze_md, out_md)
        return MaterializeResult(
            metadata={
                "stem": stem,
                "dept": dept,
                "status": "skipped",
                "reason": f"markers ({markers_before}) < MIN_MARKERS ({MIN_MARKERS})",
                "markers_before": markers_before,
                "markers_after": markers_before,
                "output_path": MetadataValue.path(str(out_md)),
                "sha256": hash_file(out_md),
            }
        )

    recovered_text, info = _recover_images(text, raw_pdf)
    out_md.write_text(recovered_text, encoding="utf-8")
    markers_after = _count_markers(recovered_text)

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "status": "recovered" if info.get("recovered") else info.get("reason", "unknown"),
            "markers_before": info["markers_before"],
            "markers_after": markers_after,
            "elapsed_s": info["elapsed_s"],
            "output_path": MetadataValue.path(str(out_md)),
            "sha256": hash_file(out_md),
        }
    )


silver_image_recovery = asset(
    key=AssetKey("silver_image_recovery"),
    deps=[AssetKey("bronze_markdown"), AssetKey("raw_pdf")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_image_recovery),
    description="Replace <!-- image --> placeholders with Gemini-recovered content.",
    group_name="silver",
)(_compute_silver_image_recovery)


@asset_check(asset=AssetKey("silver_image_recovery"), blocking=False)
def silver_image_recovery_no_hallucination(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Warn if recovered output contains MORE image markers than bronze input
    (Gemini hallucinated). The _recover_images function rolls back to bronze in
    this case, so the check should always pass — but worth verifying on disk."""
    stem = context.partition_key
    bronze_md = paths.bronze_md_path(stem)
    silver_md = paths.silver_path(stem, "image_recovery")
    bronze_n = bronze_md.read_text(encoding="utf-8").count(IMAGE_MARKER) if bronze_md.exists() else 0
    silver_n = silver_md.read_text(encoding="utf-8").count(IMAGE_MARKER) if silver_md.exists() else 0
    return AssetCheckResult(
        passed=silver_n <= bronze_n,
        severity=AssetCheckSeverity.WARN,
        metadata={"bronze_markers": bronze_n, "silver_markers": silver_n},
    )
