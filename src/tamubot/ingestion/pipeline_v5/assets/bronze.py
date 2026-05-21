"""bronze_markdown + bronze_headers_sidecar: Docling-convert the raw PDF.

Both artifacts are produced atomically by Docling, so we use @multi_asset to
emit both from a single compute. Downstream silver assets depend on either
or both.
"""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    AssetSpec,
    MaterializeResult,
    MetadataValue,
    asset_check,
    multi_asset,
)

from tamubot.ingestion.pipeline_v5 import paths
from tamubot.ingestion.pipeline_v5.partitions import stem_partitions
from tamubot.ingestion.pipeline_v5.resources import DoclingConverterResource
from tamubot.ingestion.pipeline_v5.schemas import HeadersSidecar
from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file


def _bronze_compute(
    context: AssetExecutionContext,
    docling: DoclingConverterResource,
):
    from tamubot.ingestion.converters.docling_converter import convert

    stem = context.partition_key
    dept = dept_from_stem(stem)
    pdf = paths.raw_path(stem)
    if not pdf.exists():
        raise FileNotFoundError(f"raw PDF missing for stem {stem!r}: {pdf}")

    bronze_dir = paths.bronze_md_path(stem).parent
    bronze_dir.mkdir(parents=True, exist_ok=True)

    result = convert(pdf, bronze_dir, converter=docling.get_converter(), apply_hierarchy=True)

    md_out = paths.bronze_md_path(stem)
    sidecar_raw = bronze_dir / f"{pdf.stem}.headers.json"
    sidecar_out = paths.bronze_sidecar_path(stem)
    if result.output_path != md_out:
        result.output_path.rename(md_out)
    if sidecar_raw.exists() and sidecar_raw != sidecar_out:
        sidecar_raw.rename(sidecar_out)

    raw_headers = json.loads(sidecar_out.read_text(encoding="utf-8"))
    sidecar_validated = HeadersSidecar(headers=raw_headers)
    sidecar_out.write_text(sidecar_validated.model_dump_json(indent=2), encoding="utf-8")

    md_sha = hash_file(md_out)
    sidecar_sha = hash_file(sidecar_out)
    token_estimate = len(result.markdown) // 4

    md_meta = {
        "stem": stem,
        "dept": dept,
        "output_path": MetadataValue.path(str(md_out)),
        "converter": "docling",
        "timing_s": round(result.timing_s, 2),
        "header_count": result.header_count,
        "hierarchy_depth": MetadataValue.json(result.hierarchy_depth),
        "token_count_estimate": token_estimate,
        "sha256": md_sha,
        "preview": MetadataValue.md(result.markdown[:1000]),
    }
    sidecar_meta = {
        "stem": stem,
        "dept": dept,
        "output_path": MetadataValue.path(str(sidecar_out)),
        "header_entries": len(sidecar_validated.headers),
        "schema_version": sidecar_validated.schema_version,
        "sha256": sidecar_sha,
    }

    yield MaterializeResult(asset_key=AssetKey("bronze_markdown"), metadata=md_meta)
    yield MaterializeResult(asset_key=AssetKey("bronze_headers_sidecar"), metadata=sidecar_meta)


bronze_assets = multi_asset(
    name="bronze_convert",
    specs=[
        AssetSpec(
            key=AssetKey("bronze_markdown"),
            deps=[AssetKey("raw_pdf")],
            partitions_def=stem_partitions,
            code_version=code_version_of(_bronze_compute),
            description="Docling-converted markdown of the raw PDF.",
            group_name="bronze",
        ),
        AssetSpec(
            key=AssetKey("bronze_headers_sidecar"),
            deps=[AssetKey("raw_pdf")],
            partitions_def=stem_partitions,
            code_version=code_version_of(_bronze_compute),
            description="Header→page mapping extracted from Docling's prov metadata.",
            group_name="bronze",
        ),
    ],
    can_subset=False,
)(_bronze_compute)


@asset_check(asset=AssetKey("bronze_markdown"), blocking=True)
def bronze_markdown_min_tokens(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """Quarantine: a converted syllabus under ~800 tokens almost certainly
    failed Docling layout detection (image-only PDF or empty extraction)."""
    stem = context.partition_key
    md = paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    tokens = len(text) // 4
    return AssetCheckResult(
        passed=tokens >= 800,
        severity=AssetCheckSeverity.ERROR,
        metadata={"token_count_estimate": tokens, "min_required": 800},
    )


@asset_check(asset=AssetKey("bronze_markdown"), blocking=False)
def bronze_markdown_min_headers(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """Warn if Docling didn't recognize at least 3 headers — usually means
    the PDF is image-heavy and image_recovery will need to do the heavy lifting."""
    import re

    stem = context.partition_key
    md = paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    n_headers = len(re.findall(r"(?m)^#{1,6}\s+", text))
    return AssetCheckResult(
        passed=n_headers >= 3,
        severity=AssetCheckSeverity.WARN,
        metadata={"header_count": n_headers, "min_recommended": 3},
    )


@asset_check(asset=AssetKey("bronze_markdown"), blocking=False)
def bronze_markdown_no_replacement_chars(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """Warn on U+FFFD characters (PyMuPDF artifact). Docling rarely emits these
    but worth catching."""
    stem = context.partition_key
    md = paths.bronze_md_path(stem)
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    n = text.count("�")
    return AssetCheckResult(
        passed=n == 0,
        severity=AssetCheckSeverity.WARN,
        metadata={"replacement_char_count": n},
    )


@asset_check(asset=AssetKey("bronze_headers_sidecar"), blocking=True)
def bronze_headers_sidecar_nonempty(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """Quarantine: empty sidecar means no headers were detected — the
    downstream chunker can't anchor sections to pages."""
    stem = context.partition_key
    sidecar = paths.bronze_sidecar_path(stem)
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
