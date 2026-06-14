"""Pure validators + renderers for VLM table-transcription quality (silver_modal).

No GPU / Dagster / Docling imports — host-testable, mirrors ``text_quality.py``.
These functions observe whether a table's content actually survives into the
merged markdown RAG sees, and detect the repetition-loop degeneration mode that
silently dropped whole tables (taxonomy ``FID_TABLE_LOST``).
"""

from __future__ import annotations

import re
from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome

# A percentage value in a table cell, e.g. ``40%`` / ``15 %`` / ``12.5%``.
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# Signals that a chunk is a grading/weight breakdown rather than some other %-bearing
# table (e.g. an attendance-policy table). Matched against header_path + content.
_GRADING_KEYWORDS = ("grad", "weight", "assessment", "evaluation")

# Per-table outcome labels (best -> worst), recorded for observability.
OUTCOME_TRANSCRIBED = "transcribed"  # VLM markdown accepted
OUTCOME_GRID_FALLBACK = "grid_fallback"  # Docling cell grid rescued it
OUTCOME_PARTIAL_KEPT = "partial_kept"  # degraded-but-present VLM partial text
OUTCOME_LOST = "lost"  # nothing usable — content dropped


def _is_separator_row(row: str) -> bool:
    """A GFM header/body separator like ``| --- | --- |``."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return (
        bool(cells)
        and all(set(c) <= {"-", ":"} and "-" in c for c in cells if c != "")
        and any("-" in c for c in cells)
    )


def _table_rows(gfm: str) -> list[str]:
    """Pipe-delimited rows from a GFM table string, separators excluded."""
    rows = [r for r in gfm.splitlines() if r.strip().startswith("|")]
    return [r for r in rows if not _is_separator_row(r)]


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def table_row_count(gfm: str) -> int:
    """Number of data rows (excluding the header and separator). 0 if none."""
    rows = _table_rows(gfm)
    return max(0, len(rows) - 1)


def empty_cell_ratio(gfm: str) -> float:
    """Fraction of all cells (across data + header rows) that are empty."""
    cells = [c for row in _table_rows(gfm) for c in _cells(row)]
    if not cells:
        return 0.0
    empty = sum(1 for c in cells if c == "")
    return empty / len(cells)


def max_empty_cell_run(gfm: str) -> int:
    """Longest run of consecutive empty cells read row-major — the signature of
    a `| | | | …` repetition loop."""
    run = best = 0
    for row in _table_rows(gfm):
        for c in _cells(row):
            if c == "":
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best


def is_degenerate_table(
    gfm: str,
    *,
    max_empty_run: int = 8,
    max_empty_ratio: float = 0.6,
    min_cells: int = 6,
) -> bool:
    """True when a table string shows repetition-loop degeneration: a long run
    of empty cells, or a majority-empty grid. Guarded by ``min_cells`` so a small
    legitimately-sparse table isn't flagged."""
    if not gfm.strip():
        return False
    total_cells = sum(len(_cells(row)) for row in _table_rows(gfm))
    if total_cells < min_cells:
        return False
    if max_empty_cell_run(gfm) >= max_empty_run:
        return True
    return empty_cell_ratio(gfm) > max_empty_ratio


def is_failure_marker(caption: str) -> bool:
    """True for a VLM error string the fail path stuffs into ``caption``
    (``[table processing failed: …]`` / ``[image processing failed: …]``).
    These must never leak into the merged markdown as a real caption — when a
    table falls back to the grid/partial tier the error string is meaningless
    noise that would otherwise land in a RAG chunk."""
    c = (caption or "").lstrip()
    return c.startswith("[table processing failed") or c.startswith("[image processing failed")


def looks_like_table(text: str) -> bool:
    """Heuristic: >=2 pipe-delimited rows with >=2 cells each. Used to decide
    whether a 0.5-confidence VLM partial is worth keeping (P4 rescue)."""
    rows = [r for r in text.splitlines() if r.strip().startswith("|") and not _is_separator_row(r)]
    return sum(1 for r in rows if len(_cells(r)) >= 2) >= 2


def table_body_to_gfm(rows: list[list[str]]) -> str:
    """Render a Docling-extracted table grid (``block["table_body"]``) as GFM.
    Row 0 is the header. Returns "" for an empty/degenerate grid. The fallback
    that keeps table content alive when the VLM produced no usable markdown."""
    cleaned = [[(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in row] for row in rows if row]
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


def _grid_rows_as_data(rows: list[list[str]], width: int) -> list[str]:
    """Render every row of a grid as GFM *data* rows (no header/separator line).

    Used to splice a continuation table's rows onto the preceding table. Each row
    is padded/truncated to ``width`` so it lines up with the table it merges into.
    Returns ``[]`` for an empty grid."""
    out: list[str] = []
    for row in rows:
        if not row:
            continue
        cleaned = [(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in row]
        cleaned = (cleaned + [""] * width)[:width]
        out.append("| " + " | ".join(cleaned) + " |")
    return out


def is_continuation_table(prev_grid: list[list[str]], grid: list[list[str]]) -> bool:
    """True when ``grid`` is a page-break continuation of ``prev_grid``.

    Docling splits a table that spans a page boundary into two consecutive table
    blocks; the second is often a single "header-only" row the degradation ladder
    scores as 0 data rows and loses (taxonomy ``FID_TABLE_LOST``). A continuation
    is recognised structurally: both grids are non-empty and share the same column
    width. Kept deliberately narrow (same-width only) so an unrelated adjacent
    table is never merged in."""
    if not prev_grid or not grid:
        return False
    pw = len(prev_grid[0]) if prev_grid[0] else 0
    gw = len(grid[0]) if grid[0] else 0
    return pw > 0 and pw == gw


# Keep the VLM transcription only if it preserves at least this fraction of the
# rows the Docling grid found. Below it the VLM almost certainly *truncated* the
# table (its rows are a prefix of the grid) — a clean-but-short transcription
# that is NOT degenerate-by-repetition and so would otherwise shadow the fuller
# grid. Row-count conservation: never let the shorter representation win.
_VLM_ROW_COMPLETENESS = 0.8


def _grid_tokens_covered(table_md: str, table_body: list[list[str]]) -> bool:
    """True when every content token in the Docling grid also appears in the VLM
    ``table_markdown`` — i.e. the VLM dropped no real cell/row content. Uses the
    same content-token normalization as ``text_coverage`` (lowercased words /
    numbers, punctuation and layout ignored) so reformatting never reads as a drop.
    The guard that stops a single dropped spanned row from sneaking under the
    row-count completeness threshold."""
    from tamubot.ingestion.validation.text_coverage import content_tokens

    grid_tokens = content_tokens(" ".join(c for row in table_body for c in row if c))
    if not grid_tokens:
        return True
    return grid_tokens <= content_tokens(table_md)


def render_table_markdown(block: dict[str, Any]) -> tuple[str, str]:
    """Pick the best surviving representation of a table block.

    Graceful-degradation ladder (content must NEVER be silently dropped, and the
    representation with MORE rows wins — truncation is as bad as repetition):
      1. VLM ``table_markdown``, non-degenerate AND not truncated vs the grid
                                                           -> transcribed
      2. Docling ``table_body`` grid with >=1 data row     -> grid_fallback
         (also rescues a VLM table that was truncated or degenerated)
      3. VLM rows present but grid empty                   -> transcribed
      4. VLM 0.5-confidence partial text that looks tabular -> partial_kept
      5. nothing usable                                    -> lost

    Returns ``(markdown_body, outcome)``. The caller wraps caption/markers.
    """
    modal = block.get("modal_result") or {}
    table_md = (modal.get("table_markdown") or "").strip()
    table_body = block.get("table_body") or []
    grid_md = table_body_to_gfm(table_body)

    vlm_ok = bool(table_md) and not is_degenerate_table(table_md)
    vlm_rows = table_row_count(table_md) if vlm_ok else 0
    grid_rows = table_row_count(grid_md) if grid_md else 0

    # 1. Accept the VLM transcription only when it is at least as complete as the
    #    grid (better cell formatting) — otherwise it truncated and the grid wins.
    #    Row-count conservation alone misses a single dropped spanned row that
    #    still clears the 0.8 guard (ECEN_688 mid-term row), so also require the
    #    VLM markdown to be a content-token SUPERSET of the grid: if any grid
    #    content token is absent from the VLM output the VLM dropped real content
    #    and the grid wins. Compares content tokens (words/numbers), so normal
    #    reformatting (whitespace, casing, pipes) never counts as a drop.
    if vlm_rows and (grid_rows == 0 or vlm_rows >= grid_rows * _VLM_ROW_COMPLETENESS):
        if grid_rows == 0 or _grid_tokens_covered(table_md, table_body):
            return table_md, OUTCOME_TRANSCRIBED

    # 2. Docling grid fallback — fuller than a truncated/degenerate VLM table.
    if grid_rows >= 1:
        return grid_md, OUTCOME_GRID_FALLBACK

    # 3. VLM had rows but the grid was empty — keep them rather than lose content.
    if vlm_rows:
        return table_md, OUTCOME_TRANSCRIBED

    # 4. P4: keep a degraded-but-present partial instead of total loss. The
    # degenerate `| |…` loop is itself tabular-looking, so exclude it explicitly.
    partial = (modal.get("description") or "").strip()
    if looks_like_table(partial) and not is_degenerate_table(partial):
        return partial, OUTCOME_PARTIAL_KEPT

    return "", OUTCOME_LOST


def summarize_table_outcomes(blocks: list[dict[str, Any]]) -> dict[str, int]:
    """Per-stem rollup of table-block outcomes for metadata + checks."""
    summary = {
        "table_blocks_total": 0,
        OUTCOME_TRANSCRIBED: 0,
        OUTCOME_GRID_FALLBACK: 0,
        OUTCOME_PARTIAL_KEPT: 0,
        OUTCOME_LOST: 0,
        "degenerate_tables": 0,
        # VLM produced a clean-but-short table that the grid had to rescue from
        # truncation (row-count conservation kicked in). Distinct from the
        # repetition-loop "degenerate" signal; a high count means the VLM decode
        # is dropping rows on long tables (candidate for tiling / max_tokens).
        "vlm_truncated_tables": 0,
    }
    for b in blocks:
        if b.get("type") != "table":
            continue
        summary["table_blocks_total"] += 1
        _, outcome = render_table_markdown(b)
        summary[outcome] += 1
        modal = b.get("modal_result") or {}
        table_md = (modal.get("table_markdown") or "").strip()
        if is_degenerate_table(table_md):
            summary["degenerate_tables"] += 1
        elif table_md:
            grid_rows = table_row_count(table_body_to_gfm(b.get("table_body") or []))
            vlm_rows = table_row_count(table_md)
            if grid_rows and vlm_rows < grid_rows * _VLM_ROW_COMPLETENESS:
                summary["vlm_truncated_tables"] += 1
    return summary


def _grid_token_set(grids: list[list[list[str]]]) -> set[str]:
    """Content-token set across every cell of every grid in ``grids``."""
    from tamubot.ingestion.validation.text_coverage import content_tokens

    out: set[str] = set()
    for grid in grids:
        for row in grid:
            out |= content_tokens(" ".join(c for c in row if c))
    return out


def compute_table_capture(
    docling_grids: list[list[list[str]]],
    pdf_grids: list[list[list[str]]],
    *,
    max_missing_rate: float = 0.10,
) -> CheckOutcome:
    """Table-scoped extraction-loss gate (``FID_TABLE_LOST`` / table
    ``FID_CONTENT_DROPPED``): the fraction of content tokens present in the PyMuPDF
    ``find_tables`` reconstruction but absent from EVERY Docling table grid — i.e.
    cells/rows TableFormer dropped from the structured grid. Pure — the caller
    supplies both grid lists.

    This is the deterministic detector for table under-capture, which
    ``text_coverage`` *misses*: a dropped table token (e.g. ``midterm`` from a whole
    dropped schedule row) often also appears in body text, so document-scoped token
    recall stays high while the table itself is broken. Scoping the comparison to
    table grids surfaces the structural loss. ``sample_missing`` names the dropped
    cells. Runs on post-recovery bronze, so it doubles as a regression guard on the
    adapter's table-cell recovery. WARN — calibrate ``max_missing_rate`` on the
    corpus (``find_tables`` headers/captions Docling stores separately add a small
    baseline residual)."""
    pdf = _grid_token_set(pdf_grids)
    if not pdf:
        return CheckOutcome(
            passed=True,
            metadata={
                "pdf_table_tokens": 0,
                "docling_table_tokens": len(_grid_token_set(docling_grids)),
                "missing_count": 0,
                "missing_rate": 0.0,
                "max_missing_rate": max_missing_rate,
                "sample_missing": [],
            },
        )
    docling = _grid_token_set(docling_grids)
    missing = pdf - docling
    rate = len(missing) / len(pdf)
    return CheckOutcome(
        passed=rate <= max_missing_rate,
        metadata={
            "pdf_table_tokens": len(pdf),
            "docling_table_tokens": len(docling),
            "missing_count": len(missing),
            "missing_rate": round(rate, 4),
            "max_missing_rate": max_missing_rate,
            "sample_missing": sorted(missing)[:20],
        },
    )


def check_tables_survive_view(
    blocks: list[dict[str, Any]],
    chunk_view: str,
    *,
    max_missing_rate: float = 0.5,
    min_table_tokens: int = 3,
) -> CheckOutcome:
    """End-to-end FID_TABLE_LOST gate at the **chunk view**: every bronze ``table``
    block's cell content must reach the concatenated chunk text RAG retrieves.

    Distinct from the two existing table gates, which both stop short of the chunk view:
      * bronze ``compute_table_capture`` compares *PDF -> Docling grid* (did TableFormer
        drop a cell?);
      * silver_modal ``check_no_table_lost`` runs on the *modal blocks* (a no-op by
        default), asking whether the degradation ladder produced rows.
    Neither confirms the rows actually survived the merge + chunk steps into a retrievable
    chunk. This one does: for each table block it builds the content-token set of its grid
    cells + caption and measures how many are absent from the chunk view. A table whose
    tokens are mostly gone was dropped between bronze and the chunk RAG sees.

    Conservative: tables with fewer than ``min_table_tokens`` content tokens (tiny/empty
    grids) are skipped, and a table flags only when ``> max_missing_rate`` of its tokens
    are missing (default 50% — well clear of the token-normalization noise that leaves
    coverage at ~0.98 on healthy tables). ``offenders`` names the dropped cells. Shipped
    WARN — promote to blocking once the pass rate is confirmed across the golden set."""
    from tamubot.ingestion.validation.text_coverage import content_tokens

    view_tokens = content_tokens(chunk_view)
    offenders: list[dict[str, Any]] = []
    evaluated = 0
    for b in blocks:
        if b.get("type") != "table":
            continue
        cells = " ".join(c for row in (b.get("table_body") or []) for c in row if c)
        caption = b.get("table_caption") or ""
        toks = content_tokens(f"{cells} {caption}")
        if len(toks) < min_table_tokens:
            continue
        evaluated += 1
        missing = toks - view_tokens
        rate = len(missing) / len(toks)
        if rate > max_missing_rate:
            offenders.append(
                {
                    "page_idx": b.get("page_idx") or 0,
                    "missing_rate": round(rate, 4),
                    "missing_count": len(missing),
                    "table_tokens": len(toks),
                    "sample_missing": sorted(missing)[:8],
                }
            )
    offenders.sort(key=lambda o: o["missing_rate"], reverse=True)
    return CheckOutcome(
        passed=len(offenders) == 0,
        metadata={
            "lost_table_count": len(offenders),
            "evaluated_tables": evaluated,
            "max_missing_rate": max_missing_rate,
            "offenders": offenders[:20],
        },
    )


def check_no_table_lost(blocks: list[dict[str, Any]]) -> CheckOutcome:
    """The FID_TABLE_LOST gate: every table block must contribute >=1 row to the
    merged markdown via *some* tier of the ladder. Shipped as WARN; promote to
    BLOCK after corpus calibration."""
    summary = summarize_table_outcomes(blocks)
    lost = summary[OUTCOME_LOST]
    return CheckOutcome(passed=lost == 0, metadata=summary)


def check_no_degenerate_tables(blocks: list[dict[str, Any]]) -> CheckOutcome:
    """WARN when any table's VLM markdown shows repetition-loop degeneration —
    signals the presence_penalty lever didn't fully tame this table."""
    summary = summarize_table_outcomes(blocks)
    degenerate = summary["degenerate_tables"]
    return CheckOutcome(
        passed=degenerate == 0,
        metadata={"degenerate_tables": degenerate, "table_blocks_total": summary["table_blocks_total"]},
    )


def check_no_truncated_tables(blocks: list[dict[str, Any]]) -> CheckOutcome:
    """G2 truncation gate: the VLM produced a clean-but-short table that the
    Docling grid had to rescue from truncation (row-count conservation kicked in).
    A non-zero count means the VLM decode is dropping rows on long tables — a
    candidate for tiling / a higher ``max_tokens``. WARN for now."""
    summary = summarize_table_outcomes(blocks)
    truncated = summary["vlm_truncated_tables"]
    return CheckOutcome(
        passed=truncated == 0,
        metadata={"vlm_truncated_tables": truncated, "table_blocks_total": summary["table_blocks_total"]},
    )


# Substrings the VLM fail path stuffs into a caption when transcription throws.
# These must never survive into the merged markdown — the degradation ladder
# strips them when it falls back to the grid/partial tier, so any that leak
# signal the ladder didn't engage.
_FAILURE_MARKER_SUBSTRINGS = ("[table processing failed", "[image processing failed")


def count_failure_markers(markdown: str) -> int:
    """Count ``[table/image processing failed`` markers leaking into merged markdown."""
    md = markdown or ""
    return sum(md.count(s) for s in _FAILURE_MARKER_SUBSTRINGS)


def check_no_failure_markers(markdown: str, blocks: list[dict[str, Any]] | None = None) -> CheckOutcome:
    """G5 failure-marker gate: no ``[… processing failed]`` string should reach the
    merged markdown RAG sees. When ``blocks`` is supplied, also report whether any
    table was *actually lost* (no grid/partial fallback) — the caller escalates
    severity for that unrecoverable case. WARN otherwise."""
    markers = count_failure_markers(markdown)
    lost = 0
    if blocks is not None:
        lost = summarize_table_outcomes(blocks)[OUTCOME_LOST]
    return CheckOutcome(
        passed=markers == 0,
        metadata={"failure_markers": markers, "tables_lost": lost},
    )


def check_confidence_floor(
    blocks: list[dict[str, Any]],
    *,
    max_low_confidence_rate: float = 0.25,
    confidence_threshold: float = 0.5,
) -> CheckOutcome:
    """G4 confidence-floor gate: fraction of modal results below
    ``confidence_threshold``. A high rate means vision transcription is broadly
    weak on this doc (not just one bad table). WARN above ``max_low_confidence_rate``.
    Empty (no modal blocks) trivially passes."""
    modal_blocks = [b for b in blocks if b.get("modal_result")]
    total = len(modal_blocks)
    low = sum(1 for b in modal_blocks if (b.get("modal_result") or {}).get("confidence", 1.0) < confidence_threshold)
    rate = (low / total) if total else 0.0
    return CheckOutcome(
        passed=rate <= max_low_confidence_rate,
        metadata={
            "low_confidence_blocks": low,
            "total_modal_blocks": total,
            "low_confidence_rate": round(rate, 4),
            "max_low_confidence_rate": max_low_confidence_rate,
        },
    )


def pct_tables_transcribed(blocks: list[dict[str, Any]]) -> float:
    """Fraction of table blocks whose VLM transcription was accepted (best tier).
    The headline modal-quality scalar for run-over-run drift. 0.0 when no tables."""
    summary = summarize_table_outcomes(blocks)
    total = summary["table_blocks_total"]
    return (summary[OUTCOME_TRANSCRIBED] / total) if total else 0.0


def mean_table_confidence(blocks: list[dict[str, Any]]) -> float | None:
    """Mean VLM confidence across table blocks that carry a modal result, or
    ``None`` when there are none (so it never poisons a baseline with a zero)."""
    confs = [
        float((b.get("modal_result") or {}).get("confidence", 0.0))
        for b in blocks
        if b.get("type") == "table" and b.get("modal_result")
    ]
    if not confs:
        return None
    return sum(confs) / len(confs)


def _gfm_table_blocks(content: str) -> list[list[str]]:
    """Split chunk content into contiguous runs of GFM table rows (lines that, after
    stripping, start with ``|``). Each run is one logical table."""
    tables: list[list[str]] = []
    current: list[str] = []
    for line in content.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _row_weight_percent(row: str) -> float | None:
    """The single weight percentage a grading-table data row contributes.

    A grading row like ``| Homework | 4 | report | 40% (10% each) |`` carries the row
    weight as the FIRST percentage in the row (``40%``); the parenthetical ``10% each``
    is a per-item breakdown, not an additional row. So we take only the first ``%`` per
    row. Returns ``None`` for a row with no percentage (header / separator / label row)."""
    m = _PCT_RE.search(row)
    return float(m.group(1)) if m else None


def _looks_like_grading_context(header_path: str, content: str) -> bool:
    hp = (header_path or "").lower()
    body = (content or "").lower()
    return any(k in hp for k in _GRADING_KEYWORDS) or any(k in body for k in _GRADING_KEYWORDS)


def check_grading_weights_sum_to_100(
    chunk: dict[str, Any],
    *,
    tolerance: float = 5.0,
    min_rows: int = 3,
) -> CheckOutcome:
    """The FID_TABLE_LOST domain gate (ECEN_688 / CSCE_689 class): a grading-weight
    table whose row percentages do NOT sum to ~100% — the signature of a dropped
    weight row (a clean ``95`` with a tolerance-of-5 miss is exactly a vanished ``5%``
    row).

    Conservative by design — only evaluates a chunk that is *clearly* a grading
    breakdown: its ``header_path``/``content`` mentions grading/weight AND it has a GFM
    table with ``>= min_rows`` rows each carrying a percentage. Non-grading %-tables
    (e.g. a single attendance-threshold row) never reach the sum check. When evaluated,
    FLAG if the summed weights miss 100 by ``>= tolerance`` (strict, so a clean ``95`` =
    a dropped 5% row lands on the boundary and flags). Reports the computed
    sum + the per-row weights. Shipped WARN — promote to blocking once confirmed on the
    golden set.

    ``passed`` is True (vacuously) for any chunk that is not a gradeable table, so it is
    safe to map over every chunk.
    """
    content = chunk.get("content") or ""
    header_path = chunk.get("header_path") or ""
    base_meta = {
        "evaluated": False,
        "grading_sum": None,
        "row_weights": [],
        "tolerance": tolerance,
    }
    if not _looks_like_grading_context(header_path, content):
        return CheckOutcome(passed=True, metadata=base_meta)

    # Pick the table with the most percentage rows (the grading breakdown), if any.
    best_weights: list[float] = []
    for table in _gfm_table_blocks(content):
        weights = [w for w in (_row_weight_percent(r) for r in table if not _is_separator_row(r)) if w is not None]
        if len(weights) > len(best_weights):
            best_weights = weights
    if len(best_weights) < min_rows:
        return CheckOutcome(passed=True, metadata=base_meta)

    total = round(sum(best_weights), 2)
    # Strict ``<`` (not ``<=``): a sum landing EXACTLY on a clean multiple short of 100
    # (e.g. 95 = a dropped 5% row, the ECEN_688/CSCE_689 signature) sits right on the
    # tolerance boundary and must FLAG; only rounding noise strictly inside the band passes.
    passed = abs(total - 100.0) < tolerance
    return CheckOutcome(
        passed=passed,
        metadata={
            "evaluated": True,
            "grading_sum": total,
            "row_weights": best_weights,
            "tolerance": tolerance,
            "header_path": header_path,
        },
    )


def check_no_grading_weight_drift(chunks: list[dict[str, Any]], *, tolerance: float = 5.0) -> CheckOutcome:
    """Document-level rollup of ``check_grading_weights_sum_to_100`` over all chunks:
    FLAG if ANY clearly-grading chunk's weights miss 100 +/- tolerance. Reports the
    count of offending chunks with their sums + header paths."""
    offenders: list[dict[str, Any]] = []
    evaluated = 0
    for c in chunks:
        out = check_grading_weights_sum_to_100(c, tolerance=tolerance)
        if not out.metadata.get("evaluated"):
            continue
        evaluated += 1
        if not out.passed:
            offenders.append(
                {
                    "header_path": out.metadata.get("header_path"),
                    "grading_sum": out.metadata.get("grading_sum"),
                    "row_weights": out.metadata.get("row_weights"),
                }
            )
    return CheckOutcome(
        passed=len(offenders) == 0,
        metadata={
            "offending_count": len(offenders),
            "evaluated_grading_tables": evaluated,
            "offenders": offenders[:20],
            "tolerance": tolerance,
        },
    )
