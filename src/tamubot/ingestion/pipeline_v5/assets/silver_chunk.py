"""silver_chunk: semantic chunking, one chunk per section.

Reuses v4's `chunk_semantic()` and the enrichment JSON for page-number
resolution. Outputs ingestion-ready JSON to `silver/07_chunk/<stem>.json`
with `chunks`, `course_metadata`, and `course_summary` merged. Rule-based,
free.
"""

import json
import re
from datetime import datetime

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.chunker_v4 import chunk_semantic
from tamubot.ingestion.pipeline_v4 import _build_page_lookup, _load_enrichment, _resolve_page
from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file

FLAG_THRESHOLD = 600
MIN_CHUNK_TOKENS = 50

_PREREQ_RE = re.compile(
    r"^#{1,6}\s+(?:\d+(?:\.\d+)*\s+)?Course\s+Prerequisites\s*\n+(.*?)(?=\n#{1,6}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _compute_silver_chunk(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)
    src_md = paths.silver_path(stem, "relocate_textbook")
    enrich_json = paths.silver_path(stem, "enrich", ext="json")
    dst_json = paths.silver_path(stem, "chunk", ext="json")
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    markdown = src_md.read_text(encoding="utf-8")
    chunks, log_info = chunk_semantic(
        markdown,
        flag_threshold=FLAG_THRESHOLD,
        min_chunk_tokens=MIN_CHUNK_TOKENS,
    )

    enrichment = _load_enrichment(enrich_json.parent, stem)
    page_lookup = _build_page_lookup(enrichment)
    for chunk in chunks:
        chunk["page"] = _resolve_page(chunk, page_lookup)
        if chunk["page"] is None and not chunk["header_path"]:
            chunk["page"] = 1

    course_metadata = enrichment.get("course_metadata", {})
    prereq_match = _PREREQ_RE.search(markdown)
    if prereq_match:
        course_metadata["prerequisites"] = prereq_match.group(1).strip()

    out_data = {
        "source_file": stem,
        "pipeline_version": "v5",
        "source": enrichment.get("source", ""),
        "course_type": enrichment.get("course_type", ""),
        "course_metadata": course_metadata,
        "course_summary": enrichment.get("course_summary", ""),
        "chunk_config": {
            "strategy": "semantic",
            "flag_threshold": FLAG_THRESHOLD,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
        },
        "total_chunks": len(chunks),
        "chunks": chunks,
        "_parsed_at": datetime.now().isoformat(),
    }
    dst_json.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = sum(1 for c in chunks if c.get("flags"))
    pages_resolved = sum(1 for c in chunks if c.get("page") is not None)

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "total_chunks": len(chunks),
            "flagged_chunks": flagged,
            "pages_resolved": pages_resolved,
            "has_enrichment": bool(enrichment),
            "chunk_log": MetadataValue.json(log_info),
            "output_path": MetadataValue.path(str(dst_json)),
            "sha256": hash_file(dst_json),
        }
    )


silver_chunk = asset(
    key=AssetKey("silver_chunk"),
    deps=[AssetKey("silver_enrich")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_chunk),
    description="Semantic chunking; one chunk per section, page-annotated via enrichment.",
    group_name="silver",
)(_compute_silver_chunk)
