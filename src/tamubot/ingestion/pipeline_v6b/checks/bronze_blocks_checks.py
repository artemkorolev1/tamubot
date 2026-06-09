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
from tamubot.ingestion.validation.header_hierarchy import (
    check_header_hierarchy_valid,
    check_header_levels_normalized,
    check_min_headers,
)
from tamubot.ingestion.validation.pdf_integrity import check_pdf_integrity, count_pdf_pages
from tamubot.ingestion.validation.text_quality import check_no_replacement_chars
from tamubot.ingestion.validation.types import CheckOutcome


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
