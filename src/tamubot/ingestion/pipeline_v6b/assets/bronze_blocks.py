"""bronze_blocks: Docling-parse the raw PDF and emit a RAG-Anything block list.

Also persists markdown + headers.json sidecar (the underlying convert() does
both atomically). Downstream silver assets read blocks.json; the markdown is
kept for diffing and as input to the semantic-chunk variant.
"""

import json

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    RetryPolicy,
    asset,
)

from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.pipeline_v6b.resources import DoclingConverterResource
from tamubot.ingestion.validation.header_hierarchy import check_header_levels_normalized
from tamubot.ingestion.validation.pdf_integrity import count_pdf_pages


def _deadletter_path(stem: str):
    return paths.bronze_blocks_path(stem).parent / f"{stem}.error.json"


def _run_docling(*, pdf_path, output_dir, converter, apply_hierarchy):
    from tamubot.ingestion.converters.docling_block_adapter import docling_to_blocks

    return docling_to_blocks(
        pdf_path=pdf_path,
        output_dir=output_dir,
        converter=converter,
        apply_hierarchy=apply_hierarchy,
    )


def _parse_or_deadletter(stem, pdf, bronze_dir, converter=None):
    """Run Docling; on failure write a dead-letter marker then re-raise so the
    partition fails (isolated) but records why."""
    import json as _json

    try:
        return _run_docling(pdf_path=pdf, output_dir=bronze_dir, converter=converter, apply_hierarchy=True)
    except Exception as exc:  # noqa: BLE001 - record-and-reraise
        dl = _deadletter_path(stem)
        dl.parent.mkdir(parents=True, exist_ok=True)
        dl.write_text(_json.dumps({"stem": stem, "error": str(exc)}, indent=2), encoding="utf-8")
        raise


def _compute_bronze_blocks(
    context: AssetExecutionContext,
) -> MaterializeResult:
    docling: DoclingConverterResource = context.resources.docling
    stem = context.partition_key
    dept = dept_from_stem(stem)
    pdf = paths.raw_path(stem)
    if not pdf.exists():
        raise FileNotFoundError(f"raw PDF missing for stem {stem!r}: {pdf}")

    bronze_dir = paths.bronze_blocks_path(stem).parent
    bronze_dir.mkdir(parents=True, exist_ok=True)

    blocks = _parse_or_deadletter(stem, pdf, bronze_dir, converter=docling.get_converter())

    # Clear any stale dead-letter marker from a prior failed run, now that this
    # stem parsed successfully — otherwise it lingers and misleads orphan/corpus
    # reports.
    deadletter = _deadletter_path(stem)
    if deadletter.exists():
        deadletter.unlink()

    blocks_out = paths.bronze_blocks_path(stem)
    blocks_out.write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")

    type_counts: dict[str, int] = {}
    for b in blocks:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1

    page_count = count_pdf_pages(pdf)

    # Raw heading-level skips the normalizer repaired — emitted so the
    # heading_repair_vs_baseline (G6) drift check has a run-over-run history.
    headers = [
        {"raw_level": b.get("raw_level", b.get("level", 1)), "level": b.get("level", 1), "text": b.get("text", "")}
        for b in blocks
        if b.get("type") == "heading"
    ]
    repaired_skip_count = check_header_levels_normalized(headers).metadata["repaired_skip_count"]

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "blocks_path": MetadataValue.path(str(blocks_out)),
            "block_count": len(blocks),
            "block_types": MetadataValue.json(type_counts),
            "page_count": page_count,
            "repaired_skip_count": repaired_skip_count,
            "sha256": hash_file(blocks_out),
            "preview": MetadataValue.json(blocks[:5]),
        }
    )


bronze_blocks = asset(
    name="v6b_bronze_blocks",
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_bronze_blocks),
    group_name="v6b_bronze",
    description="Docling-parsed PDF as RAG-Anything block list (type/text/page_idx/block_id).",
    required_resource_keys={"docling"},
    retry_policy=RetryPolicy(max_retries=2),
)(_compute_bronze_blocks)
