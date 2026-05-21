"""silver_false_positive: demote known-false-positive headers + apply
inline text cleanups. Rule-based, free.

Reuses v4's FalsePositiveFilter. Per-partition temp-dir wrap; report-writer
calls patched out so v4's report file isn't touched.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.filters.false_positive import FalsePositiveFilter
from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file


def _compute_silver_false_positive(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src = paths.silver_path(stem, "image_recovery")
    dst = paths.silver_path(stem, "false_positive")
    dst.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {}
    with (
        tempfile.TemporaryDirectory() as ti,
        tempfile.TemporaryDirectory() as to,
        patch("tamubot.ingestion.report_writer.get_report", return_value=None),
        patch("tamubot.ingestion.report_writer.update_false_positive"),
    ):
        ti_p, to_p = Path(ti), Path(to)
        shutil.copy2(src, ti_p / f"{stem}.md")
        filt = FalsePositiveFilter()
        result = filt.apply(ti_p, to_p, config={"file_pattern": f"{stem}.md"})
        shutil.copy2(to_p / f"{stem}.md", dst)
        if result.log_entries:
            log_entry = result.log_entries[0]

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "demoted_count": log_entry.get("demoted_count", 0),
            "cleanups_applied": log_entry.get("cleanups_applied", 0),
            "demoted_headers": MetadataValue.json([d["header"] for d in log_entry.get("demoted_headers", [])]),
            "output_path": MetadataValue.path(str(dst)),
            "sha256": hash_file(dst),
        }
    )


silver_false_positive = asset(
    key=AssetKey("silver_false_positive"),
    deps=[AssetKey("silver_image_recovery")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_false_positive),
    description="Demote false-positive headers + text cleanups (URL fixes, etc).",
    group_name="silver",
)(_compute_silver_false_positive)
