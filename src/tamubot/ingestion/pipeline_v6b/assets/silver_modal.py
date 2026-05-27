"""silver_modal: enrich image and table blocks with vision-LLM descriptions.

Default behaviour is **no-op** (V6B_MODAL_ENABLED=false). The bronze block
list is passed through unmodified and the merged markdown is reconstructed
from text/heading blocks only. Set V6B_MODAL_ENABLED=true and
V6B_MODAL_CALL_BUDGET=N to opt in to vision calls during the pilot.

Vision calls go directly to Gemini via config.get_genai_client() because TAMU
does not support multimodal input (mirrors the carve-out at
src/tamubot/ingestion/filters/image_recovery.py:96-117). This is the one
asset in the v6b track that bypasses tools/llm.py.
"""

import base64
import json
from typing import Optional

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from tamubot.core import config
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions

VISION_MODEL = "gemini-3.1-flash-lite"


def _make_gemini_vision_func():
    """Build a vision callable matching VisionFunc signature.

    Returns fn(prompt, image_b64, system_prompt) -> raw response text. Lazy-
    imports google.genai so import order doesn't fail when modal is disabled.
    """
    from google.genai import types

    client = config.get_genai_client()

    def _call(prompt: str, image_b64: Optional[str], system_prompt: str) -> str:
        parts = []
        if image_b64:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        data=base64.b64decode(image_b64),
                        mime_type="image/png",
                    )
                )
            )
        parts.append(types.Part(text=prompt))

        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[types.Content(parts=parts, role="user")],
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=system_prompt,
            ),
        )
        return response.text or ""

    return _call


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
        vision_func = _make_gemini_vision_func()
        cache_path = paths.modal_cache_path()
        # Tables only — image blocks are mostly logos / decorative on syllabi; the
        # informational images (grading rubrics rendered as pictures, etc.) are
        # covered by v5's existing recover-images flow. Per-table VLM calls keep
        # the per-file token budget tiny.
        table_proc = TableModalProcessor(vision_func=vision_func, cache_path=cache_path)
        try:
            enriched = process_blocks(
                blocks,
                image_processor=None,
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
        "Vision-LLM image + table descriptions merged back into blocks + markdown. "
        "Defaults to no-op pass-through (V6B_MODAL_ENABLED=false)."
    ),
)(_compute_silver_modal)


@asset_check(asset="v6b_silver_modal", blocking=False)
def v6b_silver_modal_budget_not_exceeded(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """WARN if modal was enabled and the run hit the budget — likely incomplete."""
    stem = context.partition_key
    modal_blocks = json.loads(paths.silver_modal_path(stem).read_text(encoding="utf-8"))

    # Detect "exceeded" by counting image/table blocks without modal_result
    unprocessed = sum(1 for b in modal_blocks if b.get("type") in ("image", "table") and not b.get("modal_result"))
    return AssetCheckResult(
        passed=unprocessed == 0,
        severity=AssetCheckSeverity.WARN,
        metadata={"unprocessed_image_or_table_blocks": unprocessed},
    )
