"""silver_chunk_semantic: run chunker_v4.chunk_semantic on the merged markdown.

Produces ChunkDocV4-shaped dicts with chunk_tag="v6b_semantic". Phase 1 does
not run a separate enrich stage (the v5 plan calls for combined enrich; v6b
reuses the same upstream when ready). Course metadata is derived from the
stem so chunks can still be ingested with the required fields.
"""

import json
from datetime import datetime

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.chunker_v4 import chunk_semantic
from tamubot.ingestion.pipeline_v5.util import (
    code_to_term,
    code_version_of,
    course_from_stem,
    dept_from_stem,
    hash_file,
    section_from_stem,
    term_code_from_stem,
)
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions

FLAG_THRESHOLD = 600
MIN_CHUNK_TOKENS = 50
CHUNK_TAG = "v6b_semantic"


def _crn_from_stem(stem: str) -> str:
    """Stem pattern: '<term>_<dept>_<course>_<section>_<crn>[_HP]'."""
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"cannot extract CRN from stem: {stem!r}")
    return parts[4]


def _compute_silver_chunk_semantic(context: AssetExecutionContext) -> MaterializeResult:
    stem = context.partition_key
    dept = dept_from_stem(stem)

    md_path = paths.silver_modal_markdown_path(stem)
    markdown = md_path.read_text(encoding="utf-8")

    chunks, log_info = chunk_semantic(
        markdown,
        flag_threshold=FLAG_THRESHOLD,
        min_chunk_tokens=MIN_CHUNK_TOKENS,
    )

    crn = _crn_from_stem(stem)
    course_id = course_from_stem(stem)
    section = section_from_stem(stem) or ""
    term = code_to_term(term_code_from_stem(stem))
    source = "howdy_portal" if stem.endswith("_HP") else "simple_syllabus"

    enriched = []
    for c in chunks:
        enriched.append(
            {
                "crn": crn,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "has_table": bool(c.get("has_table", False)),
                "course_id": course_id,
                "section": section,
                "term": term,
                "instructor_name": None,
                "header_path": c.get("header_path"),
                "token_count": c.get("token_count"),
                "flags": c.get("flags", []),
                "split_reason": c.get("split_reason"),
                "page": c.get("page"),
                "source": source,
                "is_boilerplate": False,
                "boilerplate_cluster": None,
                "cluster_confidence": None,
                "chunk_tag": CHUNK_TAG,
                "chunk_size": None,
                "chunk_overlap": None,
                "block_id": None,
                "anchor": c.get("header_path") or "",
                "embedding": None,
                "pipeline_version": "v6b",
                "source_file": stem,
            }
        )

    out_path = paths.silver_chunk_semantic_path(stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "source_file": stem,
        "pipeline_version": "v6b",
        "chunk_tag": CHUNK_TAG,
        "chunk_config": {
            "strategy": "semantic",
            "flag_threshold": FLAG_THRESHOLD,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
        },
        "total_chunks": len(enriched),
        "chunks": enriched,
        "_parsed_at": datetime.now().isoformat(),
    }
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = sum(1 for c in enriched if c.get("flags"))

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "total_chunks": len(enriched),
            "flagged_chunks": flagged,
            "chunk_log": MetadataValue.json(log_info),
            "chunk_tag": CHUNK_TAG,
            "output_path": MetadataValue.path(str(out_path)),
            "sha256": hash_file(out_path),
        }
    )


silver_chunk_semantic = asset(
    name="v6b_silver_chunk_semantic",
    deps=[AssetKey("v6b_silver_modal")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_chunk_semantic),
    group_name="v6b_silver",
    description="Semantic chunks via chunker_v4 over the modal-merged markdown. chunk_tag=v6b_semantic.",
)(_compute_silver_chunk_semantic)
