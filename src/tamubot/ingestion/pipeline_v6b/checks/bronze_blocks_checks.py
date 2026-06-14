"""Asset checks for v6b_bronze_blocks. Moved out of the asset file so all v6b
checks live in one place."""

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
from tamubot.ingestion.validation.image_quality import (
    check_content_bearing_images,
    check_no_ocr_failure_page,
)
from tamubot.ingestion.validation.header_hierarchy import (
    check_header_hierarchy_valid,
    check_header_levels_normalized,
    check_min_headers,
    check_suspicious_heading_rate,
)
from tamubot.ingestion.validation.pdf_integrity import check_pdf_integrity, count_pdf_pages
from tamubot.ingestion.validation.table_quality import compute_table_capture
from tamubot.ingestion.validation.text_coverage import compute_text_coverage
from tamubot.ingestion.validation.text_quality import (
    check_no_replacement_chars,
    count_ligature_damage,
    count_unanswered_labels,
    find_fabricated_links,
)
from tamubot.ingestion.validation.types import CheckOutcome


def _bronze_content_text(blocks: list) -> str:
    """All text that survives into bronze — text/heading bodies PLUS table-grid
    cells, captions and footnotes. Table content lives in ``table_body`` (a grid),
    not a ``text`` field, so a text-only join would falsely flag every schedule
    table's cells (times, dates, grade bands) as dropped."""
    parts: list[str] = []
    for b in blocks:
        t = b.get("text")
        if isinstance(t, str) and t:
            parts.append(t)
        for key in ("table_caption", "image_caption", "table_footnote", "image_footnote"):
            v = b.get(key)
            if isinstance(v, str) and v:
                parts.append(v)
        grid = b.get("table_body")
        if isinstance(grid, list):
            for row in grid:
                if isinstance(row, list):
                    parts.append(" ".join(str(c) for c in row if c))
    return "\n".join(parts)


def _pdf_plaintext(pdf_path) -> str:
    """Full PDF text layer via PyMuPDF — the ground truth the coverage check
    compares bronze against. Returns "" if PyMuPDF is unavailable or the open
    fails (the check then trivially passes rather than erroring)."""
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


@asset_check(asset="v6b_bronze_blocks", blocking=True, partitions_def=stem_partitions)
def v6b_bronze_blocks_nonempty(context: AssetCheckExecutionContext) -> AssetCheckResult:
    stem = context.partition_key
    p = paths.bronze_blocks_path(stem)
    if not p.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="blocks.json missing — bronze parse produced no output",
            metadata={"reason": "blocks.json missing"},
        )
    blocks = json.loads(p.read_text(encoding="utf-8"))
    n = len(blocks)
    return AssetCheckResult(
        passed=n > 0,
        severity=AssetCheckSeverity.ERROR,
        description=f"{n} block(s) parsed" if n else "blocks.json is empty (0 blocks)",
        metadata={"block_count": n},
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_has_text(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """A bronze block list with zero text blocks almost certainly failed parsing."""
    stem = context.partition_key
    p = paths.bronze_blocks_path(stem)
    blocks = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    text_blocks = sum(1 for b in blocks if b.get("type") == "text")
    return AssetCheckResult(
        passed=text_blocks >= 5,
        severity=AssetCheckSeverity.WARN,
        description=f"{text_blocks} text block(s) (min recommended 5) — few/none suggests a failed parse",
        metadata={"text_block_count": text_blocks, "min_recommended": 5},
    )


@asset_check(asset="v6b_bronze_blocks", blocking=True, partitions_def=stem_partitions)
def v6b_bronze_blocks_no_replacement_chars(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    text = "\n".join(b.get("text", "") for b in blocks if isinstance(b.get("text"), str))
    outcome = check_no_replacement_chars(text)
    count = outcome.metadata.get("replacement_char_count", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.ERROR,
        description=(
            "no U+FFFD replacement chars"
            if outcome.passed
            else f"{count} U+FFFD replacement char(s) — undecodable bytes survived parsing"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_header_hierarchy_valid(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Post-condition assertion on the *normalized* levels. The bronze adapter
    re-levels every heading to a skip-free hierarchy, so this must always pass —
    a failure means the normalizer was bypassed/regressed, not a content loss.
    Downgraded from blocking-ERROR to WARN: a single skip used to abort the whole
    document; re-leveling repairs it deterministically without dropping content."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    headers = [{"level": b.get("level", 1), "text": b.get("text", "")} for b in blocks if b.get("type") == "heading"]
    outcome = check_header_hierarchy_valid(headers)
    skips = outcome.metadata.get("skip_count", 0)
    total_headers = outcome.metadata.get("total_headers", 0)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"normalized hierarchy skip-free across {total_headers} headers"
            if outcome.passed
            else f"{skips} residual skip(s) after normalization — normalizer bypassed/regressed"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_header_levels_normalized(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Observability: how many raw heading-level skips the normalizer repaired
    (reads each heading's pre-normalization ``raw_level``). WARN when > 0 so the
    signal the old ERROR gate gave survives — now non-fatal."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    headers = [
        {"raw_level": b.get("raw_level", b.get("level", 1)), "level": b.get("level", 1), "text": b.get("text", "")}
        for b in blocks
        if b.get("type") == "heading"
    ]
    outcome = check_header_levels_normalized(headers)
    repaired = outcome.metadata.get("repaired_skip_count", 0)
    total_headers = outcome.metadata.get("total_headers", 0)
    return AssetCheckResult(
        passed=repaired == 0,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"no level-skips to repair across {total_headers} headers"
            if repaired == 0
            else f"repaired {repaired} raw level-skip(s) across {total_headers} headers (re-leveled, no content lost)"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_min_headers(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Catches the failure no-skip can't see: a multi-page doc whose real
    headings were all demoted to body text comes out nearly flat (zero skips but
    zero structure). WARN if <2 headers, or <2 distinct levels on a multi-page doc."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    headers = [{"level": b.get("level", 1), "text": b.get("text", "")} for b in blocks if b.get("type") == "heading"]
    page_count = count_pdf_pages(paths.raw_path(stem))
    outcome = check_min_headers(headers, page_count=page_count)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"{outcome.metadata['header_count']} headers / "
            f"{outcome.metadata['distinct_levels']} distinct level(s) across {page_count} page(s)"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_text_coverage(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Generic extraction-loss gate (FID_CONTENT_DROPPED): fraction of the source
    PDF's content tokens that survived into bronze. PyMuPDF reliably has the full
    text layer; a low coverage means Docling dropped content (URLs, label values
    like ``ISBN: 978-…``, table cells). ``sample_missing`` names what was lost.
    WARN — calibrate the threshold on the corpus before any promotion to blocking."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    bronze_text = _bronze_content_text(blocks)
    pdf_text = _pdf_plaintext(paths.raw_path(stem))
    outcome = compute_text_coverage(pdf_text, bronze_text)
    cov = outcome.metadata["coverage"]
    miss = outcome.metadata["missing_count"]
    sample = outcome.metadata["sample_missing"]
    detail = f"; e.g. {', '.join(sample[:5])}" if miss else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{cov:.1%} of PDF content tokens survived into bronze ({miss} dropped{detail})",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_table_cell_capture(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Table under-capture gate (FID_TABLE_LOST): content tokens PyMuPDF's
    ``find_tables`` finds in the table region but the Docling grid dropped. Catches
    what ``text_coverage`` misses (a dropped table token often also appears in body
    text). Runs on post-recovery bronze, so it also guards the adapter's table-cell
    recovery against regressions. WARN — calibrate on the corpus."""
    from tamubot.ingestion.converters.docling_block_adapter import _pymupdf_table_grids_by_page

    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    docling_grids = [b["table_body"] for b in blocks if b.get("type") == "table" and b.get("table_body")]
    pages = {b.get("page_idx") or 0 for b in blocks if b.get("type") == "table" and b.get("table_body")}
    pdf_grids = [g for grids in _pymupdf_table_grids_by_page(paths.raw_path(stem), pages).values() for g in grids]
    outcome = compute_table_capture(docling_grids, pdf_grids)
    miss = outcome.metadata["missing_count"]
    sample = outcome.metadata["sample_missing"]
    detail = f"; e.g. {', '.join(sample[:5])}" if miss else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{miss} table cell token(s) PyMuPDF found but the Docling grid dropped{detail}",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_no_orphaned_labels(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """Two-column reading-order gate (FID_HEADER_BROKEN): short ``Label:`` blocks
    left with no value beside them — the signature of a two-column course-info block
    Docling read column-first and orphaned. Catches what ``text_coverage`` misses
    (the values survive, just disconnected); also guards the adapter's two-column
    recovery. WARN — calibrate on the corpus."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    outcome = count_unanswered_labels(blocks)
    n = outcome.metadata["unanswered_labels"]
    total = outcome.metadata["total_labels"]
    sample = outcome.metadata["sample_unanswered"]
    detail = f"; e.g. {', '.join(sample[:5])}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{n}/{total} field labels have no value beside them{detail}",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_no_fabricated_links(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_HALLUCINATION gate (ISEN_633 class): a synthesized ``[label](mailto:X)`` /
    ``[label](http…)`` link whose target already appears as plain text elsewhere — the
    signature of a fabricated link for a value that was already present. WARN for now;
    promote to blocking once the pass rate is confirmed across the golden set."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    outcome = find_fabricated_links(blocks)
    n = outcome.metadata["fabricated_link_count"]
    total = outcome.metadata["total_links"]
    sample = outcome.metadata["sample_fabricated"]
    detail = f"; e.g. {', '.join(sample[:3])}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=f"{n}/{total} link(s) fabricated for already-present plain text{detail}",
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_suspicious_heading_rate(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """G3: catches body text wrongly promoted to a heading (inline-label-shaped,
    over-long, or sentence-terminated). WARN above 15% — a high rate means the
    bronze hierarchy is inventing structure from body lines."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    headers = [{"text": b.get("text", "")} for b in blocks if b.get("type") == "heading"]
    outcome = check_suspicious_heading_rate(headers)
    count = outcome.metadata["suspicious_heading_count"]
    total = outcome.metadata["total_headers"]
    rate = outcome.metadata["suspicious_rate"]
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"{count}/{total} headings look body-like ({rate:.0%}, threshold {outcome.metadata['max_rate']:.0%})"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_heading_repair_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """G6 drift watchdog: a code change that suddenly repairs 10x more level-skips
    is a regression signal. Compares ``repaired_skip_count`` against the run-over-run
    baseline. WARN on drift."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    headers = [
        {"raw_level": b.get("raw_level", b.get("level", 1)), "level": b.get("level", 1), "text": b.get("text", "")}
        for b in blocks
        if b.get("type") == "heading"
    ]
    repaired = check_header_levels_normalized(headers).metadata["repaired_skip_count"]
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_bronze_blocks",
        partition_key=stem,
        metadata_key="repaired_skip_count",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=repaired, history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="repaired_skip_count " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_block_count_vs_baseline(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    history = read_metadata_history(
        context.instance,
        asset_key="v6b_bronze_blocks",
        partition_key=stem,
        metadata_key="block_count",
        last_n=5,
    )
    outcome = compute_baseline_delta(current=len(blocks), history=history, max_drift_pct=0.20)
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description="block_count " + describe_delta(outcome.metadata),
        metadata=outcome.metadata,
    )


def _evaluate_source_integrity(stem: str) -> CheckOutcome:
    md_path = paths.bronze_md_path(stem)
    markdown_chars = len(md_path.read_text(encoding="utf-8")) if md_path.exists() else 0
    page_count = count_pdf_pages(paths.raw_path(stem))
    return check_pdf_integrity(page_count=page_count, markdown_chars=markdown_chars)


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_no_content_image_lost(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """FID_IMAGE_LOST / FID_TABLE_LOST gate (ISEN_665 class): a large, uncaptioned image
    block — a content figure/table that the default modal-disabled path emits as a bare
    ``<!-- image -->``. A non-zero count is EXPECTED while modal is off; it names exactly
    which stems/pages need a GPU modal/VLM pass to recover the trapped content. WARN."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    outcome = check_content_bearing_images(blocks)
    n = outcome.metadata["content_image_count"]
    pages = outcome.metadata["offending_pages"]
    detail = f"; pages {pages[:5]}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            "no large uncaptioned content images"
            if outcome.passed
            else f"{n} large uncaptioned image(s) not transcribed (modal disabled){detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_no_ocr_failure_page(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """OCR-failure page gate (CSCE_704 class): a page carrying a large image but almost
    no extracted text — a whole content region trapped in an un-OCR'd page image that no
    host-side text fix can reach. Names the pages needing a GPU modal/VLM re-pass. WARN."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    outcome = check_no_ocr_failure_page(blocks)
    n = outcome.metadata["ocr_failure_page_count"]
    pages = outcome.metadata["offending_pages"]
    detail = f"; page(s) {pages}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            "no image-only / OCR-failure pages"
            if outcome.passed
            else f"{n} image-only page(s) with trapped content{detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_low_ligature_damage(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """OCR ligature-damage gate (STAT_651 class): counts common university-policy words
    arriving in their f-ligature-dropped form ("ofce", "confdentiality", "signifcant").
    The signature fold makes matching robust, but a high count means this stem's text
    layer is badly damaged and any text-similarity (dedup/boilerplate/retrieval) on it is
    degraded — worth a human glance. WARN above the threshold."""
    stem = context.partition_key
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))
    text = _bronze_content_text(blocks)
    outcome = count_ligature_damage(text)
    n = outcome.metadata["ligature_damage_count"]
    sample = outcome.metadata["matches"]
    detail = f"; e.g. {', '.join(sample[:5])}" if n else ""
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"{n} ligature-damaged policy word(s) (threshold {outcome.metadata['threshold']}){detail}"
        ),
        metadata=outcome.metadata,
    )


@asset_check(asset="v6b_bronze_blocks", blocking=False, partitions_def=stem_partitions)
def v6b_bronze_blocks_source_integrity(
    context: AssetCheckExecutionContext,
) -> AssetCheckResult:
    """WARN if the source PDF has no pages or near-zero extractable text
    (scanned/corrupt syllabus — would index as empty)."""
    outcome = _evaluate_source_integrity(context.partition_key)
    cpp = outcome.metadata.get("chars_per_page")
    pages = outcome.metadata.get("page_count")
    min_cpp = outcome.metadata.get("min_chars_per_page")
    return AssetCheckResult(
        passed=outcome.passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"source OK — {cpp} chars/page across {pages} page(s)"
            if outcome.passed
            else f"weak source — {cpp} chars/page (min {min_cpp}); scanned/corrupt PDF would index empty"
        ),
        metadata=outcome.metadata,
    )
