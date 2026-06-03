"""silver_modal: enrich image and table blocks with vision descriptions.

Default behaviour is **no-op** (V6B_MODAL_ENABLED=false). The bronze block
list is passed through unmodified and the merged markdown is reconstructed
from text/heading blocks only. Set V6B_MODAL_ENABLED=true and
V6B_MODAL_CALL_BUDGET=N to opt in to vision calls during the pilot.

Vision runs entirely on the local NuExtract3 (Qwen3.5-VL) model via the
``nuextract`` resource — tables are transcribed to GFM and images described,
both with no external multimodal API. NuExtract is template-driven, so each
task uses its own template (TABLE_TEMPLATE / IMAGE_TEMPLATE) rather than a
free-text prompt; the prompt/system_prompt args of the VisionFunc contract are
accepted but unused.
"""

import base64
import io
import json
from typing import Optional

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.core import config
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions


def _b64_to_image(image_b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _make_nuextract_table_func(extractor):
    """VisionFunc for tables: transcribe the rendered table image to GFM via
    NuExtract. Returns JSON in the shape TableModalProcessor expects."""

    def _call(prompt: str, image_b64: Optional[str], system_prompt: str) -> str:
        if not image_b64:
            return json.dumps({"table_markdown": "", "caption": ""})
        data = extractor.transcribe_table(_b64_to_image(image_b64))
        caption = data.get("caption", "") or ""
        return json.dumps(
            {
                "table_markdown": data.get("table_markdown", "") or "",
                "caption": caption,
                "entity_info": {"summary": caption},
            }
        )

    return _call


def _make_nuextract_image_func(extractor):
    """VisionFunc for images: extract visible text via NuExtract (OCR-style). The
    extracted text becomes the block's description. Returns JSON in the shape
    ImageModalProcessor expects."""

    def _call(prompt: str, image_b64: Optional[str], system_prompt: str) -> str:
        if not image_b64:
            return json.dumps({"detailed_description": ""})
        text = extractor.describe_image(_b64_to_image(image_b64)).get("visible_text", "") or ""
        return json.dumps(
            {
                "detailed_description": text,
                "entity_info": {"summary": text[:200]},
            }
        )

    return _call


def _table_body_to_gfm(rows: list[list[str]]) -> str:
    """Render a Docling-extracted table grid (`block["table_body"]`) as GitHub-
    flavored markdown. Row 0 is treated as the header. Returns "" for an empty or
    degenerate grid. This is the fallback that keeps table content alive when the
    modal stage is disabled (no `modal_result.table_markdown`)."""
    cleaned = [
        [(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in row]
        for row in rows
        if row
    ]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    if width == 0:
        return ""
    norm = [r + [""] * (width - len(r)) for r in cleaned]
    out = [
        "| " + " | ".join(norm[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in norm[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _merge_to_markdown(blocks: list[dict]) -> str:
    """Reconstruct a markdown string from a block list.

    Used by the semantic chunker downstream. When silver_modal has run with
    descriptions, image/table blocks contribute their captions; in no-op
    mode they contribute a placeholder so chunker_v4 can still see them.
    """
    lines: list[str] = []
    for b in blocks:
        btype = b.get("type")
        if btype == "heading":
            level = max(1, min(6, int(b.get("level", 1) or 1)))
            lines.append(f"{'#' * level} {b.get('text', '').strip()}")
            lines.append("")
        elif btype == "text":
            lines.append(b.get("text", "").strip())
            lines.append("")
        elif btype == "image":
            modal = b.get("modal_result") or {}
            caption = modal.get("caption") or b.get("image_caption") or ""
            description = modal.get("description") or ""
            if description:
                lines.append(f"![{caption}](#) <!-- {description} -->")
            elif caption:
                lines.append(f"![{caption}](#)")
            lines.append("")
        elif btype == "table":
            modal = b.get("modal_result") or {}
            table_md = (modal.get("table_markdown") or "").strip()
            caption = modal.get("caption") or b.get("table_caption") or ""
            # Fall back to the Docling-extracted cell grid when modal didn't
            # transcribe the table (e.g. V6B_MODAL_ENABLED=false) — otherwise the
            # table's content is silently dropped from the chunker's markdown.
            if not table_md:
                table_md = _table_body_to_gfm(b.get("table_body") or [])
            if table_md:
                if caption:
                    lines.append(f"<!-- table: {caption} -->")
                lines.append(table_md)
            elif caption:
                lines.append(f"<!-- table: {caption} -->")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compute_silver_modal(
    context: AssetExecutionContext,
) -> MaterializeResult:
    from tamubot.vendor.raganything import (
        TableModalProcessor,
        process_blocks,
    )

    stem = context.partition_key
    dept = dept_from_stem(stem)

    bronze_path = paths.bronze_blocks_path(stem)
    blocks = json.loads(bronze_path.read_text(encoding="utf-8"))

    modal_out = paths.silver_modal_path(stem)
    md_out = paths.silver_modal_markdown_path(stem)
    modal_out.parent.mkdir(parents=True, exist_ok=True)

    enabled = config.V6B_MODAL_ENABLED
    budget = config.V6B_MODAL_CALL_BUDGET
    calls_made = 0
    skipped_reason: Optional[str] = None

    if not enabled:
        skipped_reason = "V6B_MODAL_ENABLED=false (no-op pass-through)"
        enriched = blocks
    else:
        from tamubot.vendor.raganything import ImageModalProcessor

        extractor = context.resources.nuextract.get_extractor()
        cache_path = paths.modal_cache_path()
        # Both tables and images run on the local NuExtract3 model: tables → GFM
        # transcription, images → description. Per-block VLM calls share the
        # per-file budget below.
        table_proc = TableModalProcessor(
            vision_func=_make_nuextract_table_func(extractor), cache_path=cache_path
        )
        image_proc = ImageModalProcessor(
            vision_func=_make_nuextract_image_func(extractor), cache_path=cache_path
        )
        try:
            enriched = process_blocks(
                blocks,
                image_processor=image_proc,
                table_processor=table_proc,
                block_id_fn=lambda b, i: b.get("block_id") or f"{stem}_blk_{i}",
                budget=budget,
            )
        except RuntimeError as exc:
            skipped_reason = str(exc)
            enriched = blocks
        for b in enriched:
            mr = b.get("modal_result")
            if mr and not mr.get("from_cache"):
                calls_made += 1

    modal_out.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _merge_to_markdown(enriched)
    md_out.write_text(md, encoding="utf-8")

    low_conf = sum(1 for b in enriched if (b.get("modal_result") or {}).get("confidence", 1.0) < 0.5)

    meta = {
        "stem": stem,
        "dept": dept,
        "modal_enabled": enabled,
        "modal_calls_made": calls_made,
        "modal_budget": budget,
        "low_confidence_blocks": low_conf,
        "blocks_path": MetadataValue.path(str(modal_out)),
        "markdown_path": MetadataValue.path(str(md_out)),
    }
    if skipped_reason:
        meta["skipped_reason"] = skipped_reason

    return MaterializeResult(metadata=meta)


silver_modal = asset(
    name="v6b_silver_modal",
    deps=[AssetKey("v6b_bronze_blocks")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_modal),
    group_name="v6b_silver",
    description=(
        "Local NuExtract3 image + table descriptions merged back into blocks + markdown. "
        "Defaults to no-op pass-through (V6B_MODAL_ENABLED=false)."
    ),
    required_resource_keys={"nuextract"},
)(_compute_silver_modal)
