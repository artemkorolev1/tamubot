"""silver_relocate_textbook: move Docling-misplaced textbook blocks under the
correct "Textbook and/or Resource Materials" header. Rule-based, free.

Reuses v4's RelocateTextbookFilter. Docling sometimes anchors the textbook
block to a nearby unrelated header (Grading Policy / Submission of Assignments
/ Late Work Policy) because the textbook-cover image disrupts layout. This
asset detects those blocks (anchored on "This material Is" + ISBN/Authors/
Publisher markers) and relocates them.
"""

import shutil
import tempfile
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.filters.relocate_textbook import RelocateTextbookFilter
from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file


def _compute_silver_relocate_textbook(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src = paths.silver_path(stem, "boilerplate")
    dst = paths.silver_path(stem, "relocate_textbook")
    dst.parent.mkdir(parents=True, exist_ok=True)

    log_entry: dict = {}
    with tempfile.TemporaryDirectory() as ti, tempfile.TemporaryDirectory() as to:
        ti_p, to_p = Path(ti), Path(to)
        shutil.copy2(src, ti_p / f"{stem}.md")
        filt = RelocateTextbookFilter()
        result = filt.apply(ti_p, to_p, config={"file_pattern": f"{stem}.md", "mode": "move"})
        shutil.copy2(to_p / f"{stem}.md", dst)
        if result.log_entries:
            log_entry = result.log_entries[0]

    detections = log_entry.get("detections", [])
    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "blocks_moved": log_entry.get("blocks_moved", 0),
            "blocks_flagged": log_entry.get("blocks_flagged", 0),
            "detections": MetadataValue.json(detections),
            "from_headers": MetadataValue.json([d.get("from_header") for d in detections]),
            "output_path": MetadataValue.path(str(dst)),
            "sha256": hash_file(dst),
        }
    )


silver_relocate_textbook = asset(
    key=AssetKey("silver_relocate_textbook"),
    deps=[AssetKey("silver_boilerplate")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_relocate_textbook),
    description="Relocate misplaced textbook blocks (Docling layout artifact) under the correct header.",
    group_name="silver",
)(_compute_silver_relocate_textbook)
