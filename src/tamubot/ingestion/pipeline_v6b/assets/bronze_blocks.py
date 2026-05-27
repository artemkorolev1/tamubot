"""bronze_blocks: Docling-parse the raw PDF and emit a RAG-Anything block list.

Also persists markdown + headers.json sidecar (the underlying convert() does
both atomically). Downstream silver assets read blocks.json; the markdown is
kept for diffing and as input to the semantic-chunk variant.
"""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.pipeline_v6b.resources import DoclingConverterResource


def _compute_bronze_blocks(
    context: AssetExecutionContext,
) -> MaterializeResult:
    from tamubot.ingestion.converters.docling_block_adapter import docling_to_blocks

    docling: DoclingConverterResource = context.resources.docling
    stem = context.partition_key
    dept = dept_from_stem(stem)
    pdf = paths.raw_path(stem)
    if not pdf.exists():
        raise FileNotFoundError(f"raw PDF missing for stem {stem!r}: {pdf}")

    bronze_dir = paths.bronze_blocks_path(stem).parent
    bronze_dir.mkdir(parents=True, exist_ok=True)

    blocks = docling_to_blocks(
        pdf_path=pdf,
        output_dir=bronze_dir,
        converter=docling.get_converter(),
        apply_hierarchy=True,
    )

    blocks_out = paths.bronze_blocks_path(stem)
    blocks_out.write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")

    type_counts: dict[str, int] = {}
    for b in blocks:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "blocks_path": MetadataValue.path(str(blocks_out)),
            "block_count": len(blocks),
            "block_types": MetadataValue.json(type_counts),
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
)(_compute_bronze_blocks)


@asset_check(asset="v6b_bronze_blocks", blocking=True)
def v6b_bronze_blocks_nonempty(context: AssetCheckExecutionContext) -> AssetCheckResult:
    stem = context.partition_key
    p = paths.bronze_blocks_path(stem)
    if not p.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "blocks.json missing"},
        )
    blocks = json.loads(p.read_text(encoding="utf-8"))
    return AssetCheckResult(
        passed=len(blocks) > 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"block_count": len(blocks)},
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False)
def v6b_bronze_blocks_has_text(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """A bronze block list that produced zero text blocks almost certainly
    failed parsing (image-only PDF or empty extraction)."""
    stem = context.partition_key
    p = paths.bronze_blocks_path(stem)
    blocks = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    text_blocks = sum(1 for b in blocks if b.get("type") == "text")
    return AssetCheckResult(
        passed=text_blocks >= 5,
        severity=AssetCheckSeverity.WARN,
        metadata={"text_block_count": text_blocks, "min_recommended": 5},
    )
