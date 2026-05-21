"""silver_boilerplate: strip TAMU policy boilerplate sections. Rule-based, free.

Reuses v4's BoilerplateFilter. Outputs both the cleaned markdown AND a
`<stem>_stripped.txt` sidecar listing what was removed.
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

from tamubot.ingestion.filters.boilerplate import BoilerplateFilter
from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file


def _compute_silver_boilerplate(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src = paths.silver_path(stem, "false_positive")
    dst = paths.silver_path(stem, "boilerplate")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dst = dst.parent / f"{stem}_stripped.txt"

    log_entry = {}
    with (
        tempfile.TemporaryDirectory() as ti,
        tempfile.TemporaryDirectory() as to,
        patch("tamubot.ingestion.report_writer.get_report", return_value=None),
        patch("tamubot.ingestion.report_writer.update_boilerplate"),
    ):
        ti_p, to_p = Path(ti), Path(to)
        shutil.copy2(src, ti_p / f"{stem}.md")
        filt = BoilerplateFilter()
        result = filt.apply(ti_p, to_p, config={"file_pattern": f"{stem}.md"})
        shutil.copy2(to_p / f"{stem}.md", dst)
        sidecar_tmp = to_p / f"{stem}_stripped.txt"
        if sidecar_tmp.exists():
            shutil.copy2(sidecar_tmp, sidecar_dst)
        if result.log_entries:
            log_entry = result.log_entries[0]

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "sections_stripped": log_entry.get("sections_stripped", 0),
            "tokens_removed": log_entry.get("tokens_removed", 0),
            "new_candidates": MetadataValue.json(log_entry.get("new_candidates", [])),
            "stripped_headers": MetadataValue.json([d["header"] for d in log_entry.get("strip_details", [])]),
            "output_path": MetadataValue.path(str(dst)),
            "sha256": hash_file(dst),
        }
    )


silver_boilerplate = asset(
    key=AssetKey("silver_boilerplate"),
    deps=[AssetKey("silver_false_positive")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_boilerplate),
    description="Strip TAMU policy boilerplate sections (ADA, FERPA, Title IX, etc.).",
    group_name="silver",
)(_compute_silver_boilerplate)
