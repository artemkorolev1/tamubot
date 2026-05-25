"""silver_enrich: LLM-driven metadata + course-summary extraction.

Reuses v4's `enrich_file()`. Outputs JSON to `silver/05_enrich/<stem>.json`.
Costs ~2 TAMU LLM calls per file (metadata + summary).
"""

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.core import config
from tamubot.ingestion.filters.metadata_enrichment import (
    enrich_file,
    load_scraped_metadata,
)
from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file


def _compute_silver_enrich(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src_md = paths.silver_path(stem, "relocate_textbook")
    dst_json = paths.silver_path(stem, "enrich", ext="json")
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir = paths.bronze_sidecar_path(stem).parent

    client = config.get_tamu_client()
    scraped_meta = load_scraped_metadata()

    enrichment = enrich_file(src_md, dst_json.parent, client, scraped_meta, sidecar_dir=sidecar_dir)
    log = enrichment.get("_log", {})
    course_metadata = enrichment.get("course_metadata", {})

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "llm_metadata_status": log.get("llm_metadata_status", "unknown"),
            "llm_summary_status": log.get("llm_summary_status", "unknown"),
            "headers_source": log.get("headers_source", "unknown"),
            "header_count": log.get("header_count", 0),
            "headers_with_page": log.get("headers_with_page", 0),
            "scraped_meta_found": log.get("scraped_meta_found", False),
            "course_id": course_metadata.get("course_id", ""),
            "section": course_metadata.get("section", ""),
            "term": course_metadata.get("term", ""),
            "instructor": MetadataValue.json(course_metadata.get("instructor") or {}),
            "summary_chars": len(enrichment.get("course_summary") or ""),
            "output_path": MetadataValue.path(str(dst_json)),
            "sha256": hash_file(dst_json),
        }
    )


silver_enrich = asset(
    key=AssetKey("silver_enrich"),
    deps=[AssetKey("silver_relocate_textbook")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_enrich),
    description="LLM-driven metadata + COURSE_SUMMARY extraction (~2 TAMU calls/file).",
    group_name="silver",
)(_compute_silver_enrich)
