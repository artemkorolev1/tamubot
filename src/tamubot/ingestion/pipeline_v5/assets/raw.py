"""raw_pdf asset: copy a source PDF (Simple Syllabus or Howdy Portal) into the
per-dept medallion tree at data/syllabi/<DEPT>/raw/<stem>.pdf."""

import shutil

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.source_discovery import discover_source_pdfs
from tamubot.ingestion.pipeline_v5.util import (
    code_version_of,
    dept_from_stem,
    hash_file,
)


def _compute_raw_pdf(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)

    sources = discover_source_pdfs(dept)
    src = sources.get(stem)
    if src is None:
        raise FileNotFoundError(
            f"No source PDF found for stem {stem!r} in dept {dept!r}. Searched simple_syllabus and howdy_portal trees."
        )

    out = paths.raw_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(out))

    sha = hash_file(out)
    size_kb = round(out.stat().st_size / 1024, 1)
    context.log.info(f"copied {src} → {out} ({size_kb} KB, sha {sha})")

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "source_path": str(src),
            "output_path": MetadataValue.path(str(out)),
            "size_kb": size_kb,
            "sha256": sha,
        }
    )


raw_pdf = asset(
    key=AssetKey("raw_pdf"),
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_raw_pdf),
    description="Copy source PDF (Simple Syllabus or Howdy Portal) to the per-dept raw/ tree.",
    group_name="raw",
)(_compute_raw_pdf)
