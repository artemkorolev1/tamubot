"""Asset checks for v6b_silver_chunk_semantic."""

import json

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    asset_check,
)

from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.ingestion.validation.baseline_diff import (
    compute_baseline_delta,
    describe_delta,
    read_metadata_history,
)
from tamubot.ingestion.validation.image_quality import check_no_bare_image_markers
from tamubot.ingestion.validation.schema_validation import check_chunks_schema_valid
from tamubot.ingestion.validation.table_quality import (
    check_no_grading_weight_drift,
    check_tables_survive_view,
)
from tamubot.ingestion.validation.text_coverage import compute_text_coverage
from tamubot.ingestion.validation.text_quality import (
    check_no_replacement_chars,
    find_garbled_link_anchors,
)
from tamubot.ingestion.validation.token_distribution import (
    check_chunk_count_nonzero,
    check_low_no_header_rate,
    check_no_header_only_chunks,
    check_no_oversized_chunks,
)


def _load_chunks(stem: str) -> list[dict]:
    data = json.loads(paths.silver_chunk_semantic_path(stem).read_text(encoding="utf-8"))
    return data["chunks"]


def _chunk_view_text(chunks: list[dict]) -> str:
    """The text RAG actually retrieves: every chunk's ``content`` concatenated in order.
    This is what the end-to-end fidelity gates below compare against the source — loss
    introduced *after* bronze (the silver_modal merge, the chunker) is only visible here."""
    return "\n".join(c.get("content") or "" for c in chunks)


def _bronze_blocks(stem: str) -> list[dict]:
    """Bronze block list for cross-checking what reached the chunk view. Returns [] if the
    bronze artifact is absent (the per-block gates then trivially pass)."""
    p = paths.bronze_blocks_path(stem)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _pdf_plaintext(pdf_path) -> str:
    """Full PDF text layer via PyMuPDF — ground truth for the chunk-view coverage gate.
    Returns "" if PyMuPDF is unavailable or the open fails (the check then trivially
    passes rather than erroring). Mirrors the helper in ``bronze_blocks_checks``."""
    try:
        import pymupdf
    except ImportError:
        return ""
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return ""
    try:
        return "\n".join(page.get_text() for page in doc)  # type: ignore[arg-type]
    finally:
        doc.close()


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_count_nonzero(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_chunk_count_nonzero(chunks)
    n = outcome.metadata.get("chunk_count", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=f"{n} chunk(s)" if outcome.passed else "0 chunks — chunking produced nothing",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_oversized(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_oversized_chunks(chunks)
    over = outcome.metadata.get("oversized_count", 0)
    total = outcome.metadata.get("total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"no oversized chunks ({total} total)"
            if outcome.passed
            else f"{over} of {total} chunk(s) exceed the token cap"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_low_no_header_rate(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_low_no_header_rate(chunks, max_rate=0.10)
    rate = outcome.metadata.get("no_header_rate", 0.0)
    mx = outcome.metadata.get("max_rate", 0.10)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=(
            f"no-header rate {rate:.0%} within max {mx:.0%}"
            if outcome.passed
            else f"no-header rate {rate:.0%} exceeds max {mx:.0%} — too many headerless chunks"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_header_only(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """CHUNK_ORPHAN_HEADER gate (STAT_620 class): a chunk whose content is ONLY header
    lines (every non-blank line starts with ``#``) — a body-less orphan header. The
    INVERSE of low_no_header_rate (which flags header-LESS chunks). WARN for now;
    promote to blocking once the pass rate is confirmed across the golden set."""
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_header_only_chunks(chunks)
    n = outcome.metadata["header_only_count"]
    total = outcome.metadata["total"]
    paths_sample = outcome.metadata["offending_header_paths"]
    detail = f"; e.g. {', '.join(p for p in paths_sample[:3] if p)}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{n}/{total} chunk(s) are header-only orphans{detail}",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_grading_weights_sum_100(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_TABLE_LOST domain gate (ECEN_688 / CSCE_689 class): a grading-weight table
    whose row percentages don't sum to ~100% (a clean ``95`` = a dropped 5% row).
    Conservative — only evaluates chunks that are clearly grading breakdowns (grading/
    weight context AND >=3 percentage rows). WARN for now; promote to blocking once the
    pass rate is confirmed across the golden set."""
    chunks = _load_chunks(context.partition_key)
    outcome = check_no_grading_weight_drift(chunks)
    n = outcome.metadata["offending_count"]
    evaluated = outcome.metadata["evaluated_grading_tables"]
    offenders = outcome.metadata["offenders"]
    detail = (
        f"; e.g. {offenders[0]['header_path']} sums to {offenders[0]['grading_sum']}%" if n and offenders else ""
    )
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"all {evaluated} grading table(s) sum to ~100%"
            if outcome.passed
            else f"{n}/{evaluated} grading table(s) miss 100±{outcome.metadata['tolerance']:.0f}%{detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=True, partitions_def=stem_partitions)
def v6b_silver_chunk_schema_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    chunks = _load_chunks(context.partition_key)
    outcome = check_chunks_schema_valid(chunks)
    invalid = outcome.metadata.get("invalid_count", 0)
    total = outcome.metadata.get("total", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=(
            f"all {total} chunk(s) schema-valid"
            if outcome.passed
            else f"{invalid} of {total} chunk(s) fail schema validation"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_total_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    chunks = _load_chunks(stem)
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_chunk_semantic",
        partition_key=stem,
        metadata_key="total_chunks",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=len(chunks), history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="chunk_count " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_flagged_rate_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    chunks = _load_chunks(stem)
    current_rate = sum(1 for c in chunks if c.get("flags")) / len(chunks) if chunks else 0.0
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_silver_chunk_semantic",
        partition_key=stem,
        metadata_key="flagged_chunks",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=current_rate, history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="flagged_rate " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )


# ── End-to-end chunk-view fidelity gates ──────────────────────────────────────
# Every fidelity gate upstream sits at bronze (vs PDF) or the no-op silver_modal —
# nothing confirms content survived the merge + chunk steps into the chunk view RAG
# actually retrieves. These four close that gap at the last stage before embedding.


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_bare_image_markers(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_IMAGE_LOST gate at the chunk view (G-A): a bare ``<!-- image -->`` that reached
    a chunk is a figure RAG retrieves with zero recoverable content. Confirmatory
    counterpart to the bronze ``no_content_image_lost`` gate (which is predictive on bbox
    area) — this counts markers that actually survived chunking. WARN; expected non-zero
    while modal is disabled."""
    stem = context.partition_key
    chunks = _load_chunks(stem)
    view = _chunk_view_text(chunks)
    image_blocks = sum(1 for b in _bronze_blocks(stem) if b.get("type") == "image")
    outcome = check_no_bare_image_markers(view, bronze_image_count=image_blocks)
    n = outcome.metadata["bare_image_markers"]
    rate = outcome.metadata.get("bare_marker_rate")
    detail = f" ({rate:.0%} of {image_blocks} image block(s))" if rate is not None and image_blocks else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            "no bare image markers in the chunk view"
            if outcome.passed
            else f"{n} bare <!-- image --> marker(s) reached the chunk view{detail} — content not transcribed"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_tables_survive_view(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_TABLE_LOST gate at the chunk view (G-B): every bronze table block's cell
    content must appear in the concatenated chunk text RAG retrieves. Catches loss between
    bronze and the chunk view (the merge/chunk steps) that the bronze ``table_cell_capture``
    (PDF→grid) and the no-op modal ``no_table_lost`` cannot see. WARN — calibrate before
    promoting to blocking."""
    stem = context.partition_key
    chunks = _load_chunks(stem)
    view = _chunk_view_text(chunks)
    outcome = check_tables_survive_view(_bronze_blocks(stem), view)
    n = outcome.metadata["lost_table_count"]
    evaluated = outcome.metadata["evaluated_tables"]
    offenders = outcome.metadata["offenders"]
    detail = (
        f"; e.g. page {offenders[0]['page_idx']} missing {offenders[0]['sample_missing'][:3]}"
        if n and offenders
        else ""
    )
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"all {evaluated} table(s) survived into the chunk view"
            if outcome.passed
            else f"{n}/{evaluated} table(s) lost between bronze and the chunk view{detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_view_text_coverage(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_CONTENT_DROPPED gate at the chunk view (G-C): fraction of the source PDF's
    content tokens that survived all the way into the chunk view RAG retrieves. The bronze
    ``text_coverage`` gate stops at bronze; this one extends the same token-recall measure
    end-to-end, so loss introduced by the silver_modal merge or the chunker shows up.
    Threshold is deliberately loose (gross loss only); ``sample_missing`` is the triage
    aid. WARN — calibrate before promoting to blocking."""
    stem = context.partition_key
    chunks = _load_chunks(stem)
    view = _chunk_view_text(chunks)
    pdf_text = _pdf_plaintext(paths.raw_path(stem))
    outcome = compute_text_coverage(pdf_text, view, max_missing_rate=0.10)
    cov = outcome.metadata["coverage"]
    miss = outcome.metadata["missing_count"]
    sample = outcome.metadata["sample_missing"]
    detail = f"; e.g. {', '.join(sample[:5])}" if miss else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{cov:.1%} of PDF content tokens survived into the chunk view ({miss} dropped{detail})",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_replacement_chars(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_REPLACEMENT_CHARS gate at the chunk view (G-D, part 1): no U+FFFD replacement
    char should reach a retrievable chunk. The bronze gate scans block ``text`` fields
    only; this scans the rendered chunk content (including table markdown and link labels),
    catching garbage that the merge re-introduced. WARN."""
    stem = context.partition_key
    view = _chunk_view_text(_load_chunks(stem))
    outcome = check_no_replacement_chars(view)
    count = outcome.metadata["replacement_char_count"]
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            "no U+FFFD replacement chars in the chunk view"
            if outcome.passed
            else f"{count} U+FFFD replacement char(s) reached the chunk view"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_silver_chunk_semantic", blocking=False, partitions_def=stem_partitions)
def v6b_silver_chunk_no_garbled_anchors(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Broken-hyperlink gate at the chunk view (G-D, part 2): a markdown link whose anchor
    text is a truncated/garbled prose fragment (``[ms can do so within FERPA Notice to
    webpage.](…)``) — the signature of PDF text-layer damage that sliced body prose into a
    link label. WARN — promote to blocking once confirmed across the golden set."""
    stem = context.partition_key
    view = _chunk_view_text(_load_chunks(stem))
    outcome = find_garbled_link_anchors(view)
    n = outcome.metadata["garbled_anchor_count"]
    total = outcome.metadata["total_links"]
    offenders = outcome.metadata["offenders"]
    detail = f"; e.g. [{offenders[0]['label']}]" if n and offenders else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"all {total} chunk-view link(s) have clean anchor text"
            if outcome.passed
            else f"{n}/{total} link(s) have garbled/truncated anchor text{detail}"
        ),
        metadata=outcome.metadata,
    )
