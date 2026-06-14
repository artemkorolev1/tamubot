"""Tests for the new silver_modal QC gate validators (G2/G4/G5/G6 metrics).

Pure — no GPU/Dagster. Covers truncation, failure-marker, confidence-floor,
and the modal-quality scalars used by the run-over-run drift check.
"""

from tamubot.ingestion.validation.table_quality import (
    check_confidence_floor,
    check_grading_weights_sum_to_100,
    check_no_failure_markers,
    check_no_grading_weight_drift,
    check_no_truncated_tables,
    compute_table_capture,
    count_failure_markers,
    mean_table_confidence,
    pct_tables_transcribed,
)

# --- table-capture (FID_TABLE_LOST deterministic detector) ----------------


def test_table_capture_clean_when_grid_matches_pdf():
    grids = [[["Week", "Topic"], ["1", "Intro to modeling"]]]
    out = compute_table_capture(grids, grids)
    assert out.passed
    assert out.metadata["missing_count"] == 0


def test_table_capture_flags_dropped_row():
    """PyMuPDF found a row (`1st MIDTERM | Mar. 3`) the Docling grid never had."""
    docling = [[["Schedule", ""], ["V. Transmitter Analysis", "Week 9-14"]]]
    pdf = [[["Schedule", ""], ["1st", "MIDTERM"], ["V. Transmitter Analysis", "Week 9-14"]]]
    out = compute_table_capture(docling, pdf, max_missing_rate=0.0)
    assert not out.passed
    assert "midterm" in out.metadata["sample_missing"]


def test_table_capture_empty_pdf_passes():
    out = compute_table_capture([[["A", "B"]]], [])
    assert out.passed
    assert out.metadata["missing_count"] == 0


def _gfm(n_data_rows: int) -> str:
    """A clean (non-degenerate) GFM table with ``n_data_rows`` data rows."""
    lines = ["| Week | Topic |", "| --- | --- |"]
    lines += [f"| {i} | Topic {i} |" for i in range(1, n_data_rows + 1)]
    return "\n".join(lines)


def _grid(n_data_rows: int) -> list[list[str]]:
    """A Docling ``table_body`` grid: row 0 header + ``n_data_rows`` data rows."""
    return [["Week", "Topic"]] + [[str(i), f"Topic {i}"] for i in range(1, n_data_rows + 1)]


# --- G2: truncation -------------------------------------------------------


def test_no_truncated_tables_clean():
    """VLM rows match the grid -> not truncated."""
    block = {
        "type": "table",
        "modal_result": {"table_markdown": _gfm(10), "confidence": 1.0},
        "table_body": _grid(10),
    }
    out = check_no_truncated_tables([block])
    assert out.passed
    assert out.metadata["vlm_truncated_tables"] == 0


def test_no_truncated_tables_flags_short_vlm():
    """VLM transcribed only 2 of the grid's 10 rows -> truncation flagged."""
    block = {
        "type": "table",
        "modal_result": {"table_markdown": _gfm(2), "confidence": 1.0},
        "table_body": _grid(10),
    }
    out = check_no_truncated_tables([block])
    assert out.passed is False
    assert out.metadata["vlm_truncated_tables"] == 1


# --- G5: failure markers --------------------------------------------------


def test_count_failure_markers():
    md = "Some text\n<!-- table: [table processing failed: oom] -->\n| a | b |\n"
    assert count_failure_markers(md) == 1
    assert count_failure_markers("clean markdown") == 0


def test_no_failure_markers_clean():
    out = check_no_failure_markers("| a | b |\n| 1 | 2 |\n", blocks=[])
    assert out.passed
    assert out.metadata["failure_markers"] == 0


def test_no_failure_markers_with_loss():
    """A leaked marker AND a genuinely lost table (no fallback rows)."""
    lost_block = {
        "type": "table",
        "modal_result": {"table_markdown": "", "confidence": 0.0},
        "table_body": [],
    }
    md = "<!-- table: [table processing failed: timeout] -->\n"
    out = check_no_failure_markers(md, blocks=[lost_block])
    assert out.passed is False
    assert out.metadata["failure_markers"] == 1
    assert out.metadata["tables_lost"] == 1


# --- G4: confidence floor -------------------------------------------------


def test_confidence_floor_pass():
    blocks = [
        {"type": "image", "modal_result": {"confidence": 1.0}},
        {"type": "table", "modal_result": {"confidence": 0.9}},
    ]
    out = check_confidence_floor(blocks)
    assert out.passed
    assert out.metadata["low_confidence_rate"] == 0.0


def test_confidence_floor_fail():
    blocks = [
        {"type": "table", "modal_result": {"confidence": 0.3}},
        {"type": "table", "modal_result": {"confidence": 0.3}},
        {"type": "image", "modal_result": {"confidence": 1.0}},
        {"type": "image", "modal_result": {"confidence": 1.0}},
    ]
    out = check_confidence_floor(blocks)  # 2/4 = 50% > 25%
    assert out.passed is False
    assert out.metadata["low_confidence_blocks"] == 2
    assert out.metadata["total_modal_blocks"] == 4


def test_confidence_floor_no_modal_blocks_passes():
    out = check_confidence_floor([{"type": "text", "text": "hi"}])
    assert out.passed
    assert out.metadata["total_modal_blocks"] == 0


# --- G6: modal-quality scalars --------------------------------------------


def test_pct_tables_transcribed():
    blocks = [
        {"type": "table", "modal_result": {"table_markdown": _gfm(5), "confidence": 1.0}, "table_body": _grid(5)},
        {"type": "table", "modal_result": {"table_markdown": "", "confidence": 0.0}, "table_body": []},
    ]
    # One transcribed, one lost -> 0.5.
    assert pct_tables_transcribed(blocks) == 0.5


def test_pct_tables_transcribed_no_tables():
    assert pct_tables_transcribed([{"type": "text", "text": "hi"}]) == 0.0


def test_mean_table_confidence():
    blocks = [
        {"type": "table", "modal_result": {"confidence": 1.0}},
        {"type": "table", "modal_result": {"confidence": 0.5}},
        {"type": "image", "modal_result": {"confidence": 0.0}},  # ignored (not a table)
    ]
    assert mean_table_confidence(blocks) == 0.75


def test_mean_table_confidence_none_when_no_tables():
    assert mean_table_confidence([{"type": "text", "text": "hi"}]) is None


# --- Check 3: grading-weight table sums to ~100% (FID_TABLE_LOST / ECEN_688) ---

_GRADING_OK = (
    "## Grading\n\n"
    "| Item | Percent |\n| --- | --- |\n"
    "| Exam 1 | 15% |\n| Exam 2 | 15% |\n| Labs | 40% |\n| Activities | 30% |\n"
)
# A dropped 5% row: the table sums to 95, not 100.
_GRADING_DROPPED = (
    "## Grading\n\n"
    "| Item | Percent |\n| --- | --- |\n"
    "| Homework | 40% |\n| Midterms | 45% |\n| Quizzes | 10% |\n"
)


def test_grading_weights_sum_100_passes():
    chunk = {"header_path": "Grading Policy", "content": _GRADING_OK}
    out = check_grading_weights_sum_to_100(chunk)
    assert out.metadata["evaluated"]
    assert out.passed
    assert out.metadata["grading_sum"] == 100.0


def test_grading_weights_dropped_row_flagged():
    """ECEN_688 class: a grading table summing to 95% (a dropped 5% row)."""
    chunk = {"header_path": "Grading", "content": _GRADING_DROPPED}
    out = check_grading_weights_sum_to_100(chunk)
    assert out.metadata["evaluated"]
    assert out.passed is False
    assert out.metadata["grading_sum"] == 95.0


def test_grading_weights_uses_first_percent_per_row():
    """`40% (10% each)` contributes 40, not 40+10 — the parenthetical is a breakdown."""
    content = (
        "## Grading\n\n"
        "| Item | Percent |\n| --- | --- |\n"
        "| Homework | 40% (10% each) |\n| Midterms | 45% (15% each) |\n| Quizzes | 15% (5% each) |\n"
    )
    out = check_grading_weights_sum_to_100({"header_path": "Grading", "content": content})
    assert out.metadata["grading_sum"] == 100.0
    assert out.passed


def test_non_grading_percent_table_not_evaluated():
    """An attendance table with a single % threshold is not a grading breakdown."""
    content = "## Attendance\n\n| Rule | Threshold |\n| --- | --- |\n| Min attendance | 90% |\n"
    out = check_grading_weights_sum_to_100({"header_path": "Attendance Policy", "content": content})
    assert out.metadata["evaluated"] is False
    assert out.passed  # vacuously


def test_grading_context_but_too_few_rows_not_evaluated():
    """Grading context with <3 percent rows is too sparse to judge — not evaluated."""
    content = "## Grading\n\n| Item | Percent |\n| --- | --- |\n| Final | 100% |\n"
    out = check_grading_weights_sum_to_100({"header_path": "Grading", "content": content})
    assert out.metadata["evaluated"] is False
    assert out.passed


def test_grading_weight_drift_rollup():
    chunks = [
        {"header_path": "Grading", "content": _GRADING_OK},
        {"header_path": "Grading", "content": _GRADING_DROPPED},
        {"header_path": "Attendance", "content": "Just prose, no table."},
    ]
    out = check_no_grading_weight_drift(chunks)
    assert out.passed is False
    assert out.metadata["offending_count"] == 1
    assert out.metadata["evaluated_grading_tables"] == 2
