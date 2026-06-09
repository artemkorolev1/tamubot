"""Pure validators + renderers for VLM table-transcription quality (silver_modal).

No GPU / Dagster / Docling imports — host-testable, mirrors ``text_quality.py``.
These functions observe whether a table's content actually survives into the
merged markdown RAG sees, and detect the repetition-loop degeneration mode that
silently dropped whole tables (taxonomy ``FID_TABLE_LOST``).
"""

from __future__ import annotations

from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome

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


# Keep the VLM transcription only if it preserves at least this fraction of the
# rows the Docling grid found. Below it the VLM almost certainly *truncated* the
# table (its rows are a prefix of the grid) — a clean-but-short transcription
# that is NOT degenerate-by-repetition and so would otherwise shadow the fuller
# grid. Row-count conservation: never let the shorter representation win.
_VLM_ROW_COMPLETENESS = 0.8


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
    grid_md = table_body_to_gfm(block.get("table_body") or [])

    vlm_ok = bool(table_md) and not is_degenerate_table(table_md)
    vlm_rows = table_row_count(table_md) if vlm_ok else 0
    grid_rows = table_row_count(grid_md) if grid_md else 0

    # 1. Accept the VLM transcription only when it is at least as complete as the
    #    grid (better cell formatting) — otherwise it truncated and the grid wins.
    if vlm_rows and (grid_rows == 0 or vlm_rows >= grid_rows * _VLM_ROW_COMPLETENESS):
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
