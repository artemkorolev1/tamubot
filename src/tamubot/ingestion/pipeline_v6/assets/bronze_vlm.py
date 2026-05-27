"""bronze_vlm_markdown + bronze_vlm_headers_sidecar — replaces v5 Docling bronze.

One multimodal call per PDF: Gemini 3.1 Flash-Lite produces clean markdown.
Headers are parsed from the markdown into a HeadersSidecar (no separate sidecar
emitted by the VLM — derived from the regex over `^#{1,6}\\s+`).

Standalone `vlm_convert()` is callable outside Dagster so the bake-off harness
can reuse it without materializing assets.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetKey,
    AssetSpec,
    MaterializeResult,
    MetadataValue,
    asset_check,
    multi_asset,
)

from tamubot.core import config
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.schemas import HeaderEntry, HeadersSidecar
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6 import paths as v6_paths

VLM_MODEL = "gemini-3.1-flash-lite"
VLM_MAX_OUTPUT_TOKENS = 16384
VLM_TEMPERATURE = 0.0

BRONZE_PROMPT = """Extract this PDF as clean markdown. Rules:

- Preserve every section header at the correct level (#, ##, ###, etc.).
- Output markdown only. Do not summarize, paraphrase, comment, or add commentary.
- Render tables as github-flavored markdown.
- Render figures as ![figure description](#) — never use <!-- image --> markers.
- Preserve the original page reading order.
- Do not invent content that is not in the source PDF.
- Preserve full markdown link syntax: `[display text](URL)`. Never drop URLs that appear in the PDF, even if shown as small text or in footnotes.
- Preserve email addresses, phone numbers, and any contact info verbatim.
- If a section header looks like a label or paragraph intro (e.g. "Important Note:", "You may use :"), keep it as bold body text, not as a header."""


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class VlmConvertResult:
    markdown: str
    headers: list[HeaderEntry]
    timing_s: float
    input_image_tokens: int = 0
    input_text_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = VLM_MODEL
    raw_usage: dict = field(default_factory=dict)


def _parse_headers_from_markdown(md: str) -> list[HeaderEntry]:
    """Extract HeaderEntry list from markdown. `page` is None — VLM doesn't
    surface page numbers in the markdown by default. Downstream chunker can
    page-anchor via the enrichment stage as v5 does."""
    return [HeaderEntry(text=m.group(2).strip(), level=len(m.group(1)), page=None) for m in _HEADER_RE.finditer(md)]


def vlm_convert(pdf_path: Path, *, prompt: str = BRONZE_PROMPT) -> VlmConvertResult:
    """Run one Gemini multimodal call on a PDF and return parsed result.

    Standalone so the bake-off and Dagster asset share the same call site.
    Does NOT write any files — caller is responsible for persistence.
    """
    from google.genai import types

    client = config.get_genai_client()
    pdf_bytes = pdf_path.read_bytes()

    t0 = time.time()
    resp = client.models.generate_content(
        model=VLM_MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=VLM_TEMPERATURE,
            max_output_tokens=VLM_MAX_OUTPUT_TOKENS,
        ),
    )
    elapsed = time.time() - t0

    md = resp.text or ""
    headers = _parse_headers_from_markdown(md)

    um = getattr(resp, "usage_metadata", None)
    image_toks = text_toks = 0
    if um is not None:
        for mod in um.prompt_tokens_details or []:
            if str(mod.modality).endswith("IMAGE"):
                image_toks = mod.token_count or 0
            elif str(mod.modality).endswith("TEXT"):
                text_toks = mod.token_count or 0

    return VlmConvertResult(
        markdown=md,
        headers=headers,
        timing_s=elapsed,
        input_image_tokens=image_toks,
        input_text_tokens=text_toks,
        output_tokens=(um.candidates_token_count if um else 0) or 0,
        total_tokens=(um.total_token_count if um else 0) or 0,
        raw_usage=(um.model_dump() if um else {}),
    )


def _bronze_compute(context):
    """Dagster compute: VLM-convert one stem's raw PDF, write markdown + sidecar."""
    stem = context.partition_key
    dept = dept_from_stem(stem)
    # Source PDF lives in v5/raw — v6 reuses v5 raw layer, doesn't re-scrape.
    from tamubot.ingestion.pipeline_v5 import paths as v5_paths

    pdf = v5_paths.raw_path(stem)
    if not pdf.exists():
        raise FileNotFoundError(f"raw PDF missing for stem {stem!r}: {pdf}")

    bronze_dir = v6_paths.bronze_md_path(stem).parent
    bronze_dir.mkdir(parents=True, exist_ok=True)

    result = vlm_convert(pdf)

    md_out = v6_paths.bronze_md_path(stem)
    md_out.write_text(result.markdown, encoding="utf-8")

    sidecar_out = v6_paths.bronze_sidecar_path(stem)
    sidecar = HeadersSidecar(headers=result.headers)
    sidecar_out.write_text(sidecar.model_dump_json(indent=2), encoding="utf-8")

    meta_out = v6_paths.bronze_meta_path(stem)
    meta_out.write_text(
        json.dumps(
            {
                "stem": stem,
                "dept": dept,
                "model": result.model,
                "timing_s": round(result.timing_s, 2),
                "input_image_tokens": result.input_image_tokens,
                "input_text_tokens": result.input_text_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "raw_usage": result.raw_usage,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    md_sha = hash_file(md_out)
    sidecar_sha = hash_file(sidecar_out)
    token_estimate = len(result.markdown) // 4

    md_meta = {
        "stem": stem,
        "dept": dept,
        "output_path": MetadataValue.path(str(md_out)),
        "converter": "gemini-3.1-flash-lite",
        "timing_s": round(result.timing_s, 2),
        "header_count": len(result.headers),
        "token_count_estimate": token_estimate,
        "input_image_tokens": result.input_image_tokens,
        "output_tokens": result.output_tokens,
        "sha256": md_sha,
        "preview": MetadataValue.md(result.markdown[:1000]),
    }
    sidecar_meta = {
        "stem": stem,
        "dept": dept,
        "output_path": MetadataValue.path(str(sidecar_out)),
        "header_entries": len(result.headers),
        "schema_version": sidecar.schema_version,
        "sha256": sidecar_sha,
    }

    yield MaterializeResult(asset_key=AssetKey("bronze_vlm_markdown"), metadata=md_meta)
    yield MaterializeResult(asset_key=AssetKey("bronze_vlm_headers_sidecar"), metadata=sidecar_meta)


bronze_vlm_assets = multi_asset(
    name="bronze_vlm_convert",
    specs=[
        AssetSpec(
            key=AssetKey("bronze_vlm_markdown"),
            deps=[AssetKey("raw_pdf")],
            partitions_def=stem_partitions,
            code_version=code_version_of(_bronze_compute),
            description="VLM-converted markdown of the raw PDF (Gemini 3.1 Flash-Lite).",
            group_name="bronze",
        ),
        AssetSpec(
            key=AssetKey("bronze_vlm_headers_sidecar"),
            deps=[AssetKey("raw_pdf")],
            partitions_def=stem_partitions,
            code_version=code_version_of(_bronze_compute),
            description="Header levels parsed from VLM markdown (no page anchors).",
            group_name="bronze",
        ),
    ],
    can_subset=False,
)(_bronze_compute)


@asset_check(asset=AssetKey("bronze_vlm_markdown"), blocking=True)
def bronze_vlm_min_tokens(context) -> AssetCheckResult:
    """Quarantine: under ~800 tokens means the VLM call failed or returned a stub."""
    stem = context.partition_key
    md = v6_paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    tokens = len(text) // 4
    return AssetCheckResult(
        passed=tokens >= 800,
        severity=AssetCheckSeverity.ERROR,
        metadata={"token_count_estimate": tokens, "min_required": 800},
    )


@asset_check(asset=AssetKey("bronze_vlm_markdown"), blocking=False)
def bronze_vlm_min_headers(context) -> AssetCheckResult:
    """Warn if fewer than 3 headers — likely incomplete extraction."""
    stem = context.partition_key
    md = v6_paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    n_headers = len(_HEADER_RE.findall(text))
    return AssetCheckResult(
        passed=n_headers >= 3,
        severity=AssetCheckSeverity.WARN,
        metadata={"header_count": n_headers, "min_recommended": 3},
    )


@asset_check(asset=AssetKey("bronze_vlm_markdown"), blocking=False)
def bronze_vlm_no_replacement_chars(context) -> AssetCheckResult:
    stem = context.partition_key
    md = v6_paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    n = text.count("�")
    return AssetCheckResult(
        passed=n == 0,
        severity=AssetCheckSeverity.WARN,
        metadata={"replacement_char_count": n},
    )


@asset_check(asset=AssetKey("bronze_vlm_markdown"), blocking=False)
def bronze_vlm_no_image_markers(context) -> AssetCheckResult:
    """Warn if the VLM emitted `<!-- image -->` markers despite the prompt."""
    stem = context.partition_key
    md = v6_paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    n = text.count("<!-- image -->")
    return AssetCheckResult(
        passed=n == 0,
        severity=AssetCheckSeverity.WARN,
        metadata={"image_marker_count": n},
    )


@asset_check(asset=AssetKey("bronze_vlm_headers_sidecar"), blocking=True)
def bronze_vlm_sidecar_nonempty(context) -> AssetCheckResult:
    stem = context.partition_key
    sidecar = v6_paths.bronze_sidecar_path(stem)
    if not sidecar.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "sidecar missing"},
        )
    parsed = HeadersSidecar.model_validate_json(sidecar.read_text(encoding="utf-8"))
    return AssetCheckResult(
        passed=len(parsed.headers) >= 1,
        severity=AssetCheckSeverity.ERROR,
        metadata={"header_entries": len(parsed.headers)},
    )
