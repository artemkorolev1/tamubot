"""silver_modal markdown reconstruction — especially that table content survives
when the modal stage is disabled (H1: render the Docling `table_body` grid)."""

from __future__ import annotations

from tamubot.ingestion.pipeline_v6b.assets import silver_modal as mod


def test_table_body_to_gfm_renders_grid():
    rows = [["Component", "Weight"], ["Homework", "40%"], ["Final", "60%"]]
    md = mod._table_body_to_gfm(rows)
    assert "| Component | Weight |" in md
    assert "| --- | --- |" in md
    assert "| Homework | 40% |" in md
    assert "| Final | 60% |" in md


def test_table_body_to_gfm_handles_empty_and_degenerate():
    assert mod._table_body_to_gfm([]) == ""
    assert mod._table_body_to_gfm([[]]) == ""


def test_table_body_to_gfm_escapes_pipes_and_pads_ragged_rows():
    rows = [["a|b", "c"], ["d"]]  # ragged + a literal pipe
    md = mod._table_body_to_gfm(rows)
    assert "a\\|b" in md
    # ragged row padded to width 2
    assert "| d |  |" in md


def test_merge_to_markdown_renders_table_body_when_modal_absent():
    blocks = [
        {"type": "heading", "text": "Grading", "level": 2},
        {
            "type": "table",
            "table_body": [["Item", "Pct"], ["Exam", "50"]],
            "table_caption": "Grade breakdown",
        },
    ]
    md = mod._merge_to_markdown(blocks)
    assert "## Grading" in md
    assert "| Item | Pct |" in md
    assert "| Exam | 50 |" in md
    assert "Grade breakdown" in md  # caption comment preserved


def test_merge_to_markdown_prefers_modal_transcription_over_raw_grid():
    blocks = [
        {
            "type": "table",
            "table_body": [["a", "b"]],
            "modal_result": {"table_markdown": "| x |\n| --- |\n| 1 |"},
        }
    ]
    md = mod._merge_to_markdown(blocks)
    assert "| x |" in md
    assert "| a | b |" not in md  # modal transcription wins when present


def test_merge_to_markdown_recovers_page_break_continuation_row():
    """FID_TABLE_LOST: a single-row continuation table (Docling page-break split,
    scored 0 data rows by the ladder) is spliced onto the preceding table instead
    of being dropped. Mirrors CSCE_689 'Async Progress Check-in Posts … 5%'."""
    blocks = [
        {
            "type": "table",
            "table_body": [
                ["Component", "Weight", "Description"],
                ["Final Project", "30%", "Tiered deliverable"],
            ],
        },
        # Continuation: its lone row would be consumed as a header → 0 data rows.
        {
            "type": "table",
            "table_body": [
                ["Async Check-in Posts", "5%", "Forum posts in Weeks 12 and 13"],
            ],
        },
    ]
    md = mod._merge_to_markdown(blocks)
    # The continuation row survives, merged into the preceding table as a data row.
    assert "| Async Check-in Posts | 5% | Forum posts in Weeks 12 and 13 |" in md
    # …and it appears after the main table's last row, before the next blank line,
    # i.e. as part of the same table (a single header row, exactly one separator).
    assert md.count("| --- | --- | --- |") == 1
    assert "| Final Project | 30% | Tiered deliverable |" in md
    # Recovered exactly once — never duplicated.
    assert md.count("Async Check-in Posts") == 1


def test_merge_to_markdown_does_not_duplicate_continuation_row_already_present():
    """Negative: if the orphan single-row table's content is ALREADY in the
    preceding table (Docling repeated it), the gate (content tokens missing)
    suppresses the splice — no duplicate row."""
    blocks = [
        {
            "type": "table",
            "table_body": [
                ["Component", "Weight", "Description"],
                ["Async Check-in Posts", "5%", "Forum posts in Weeks 12 and 13"],
            ],
        },
        # Same row repeated as a standalone continuation block.
        {
            "type": "table",
            "table_body": [
                ["Async Check-in Posts", "5%", "Forum posts in Weeks 12 and 13"],
            ],
        },
    ]
    md = mod._merge_to_markdown(blocks)
    # Present exactly once — the duplicate continuation was suppressed, not added.
    assert md.count("| Async Check-in Posts | 5% | Forum posts in Weeks 12 and 13 |") == 1


def test_merge_to_markdown_does_not_merge_continuation_across_heading():
    """A single-row table that follows a heading (not a table) is NOT a page-break
    continuation — it must not be spliced backward into an earlier table."""
    blocks = [
        {
            "type": "table",
            "table_body": [["Component", "Weight"], ["Exam", "50%"]],
        },
        {"type": "heading", "text": "Phase 2 Schedule", "level": 3},
        {
            "type": "table",
            "table_body": [["Week", "Activity"]],  # orphan header, different table
        },
    ]
    md = mod._merge_to_markdown(blocks)
    # The orphan row is NOT glued onto the first (grading) table.
    grading_block = md.split("Phase 2 Schedule")[0]
    assert "Week" not in grading_block


def test_merge_to_markdown_does_not_merge_continuation_of_different_width():
    """Width guard: a single-row table with a different column count is not a
    continuation of the preceding table and is not spliced into it."""
    blocks = [
        {
            "type": "table",
            "table_body": [["Component", "Weight", "Description"], ["Exam", "50%", "Final"]],
        },
        {
            "type": "table",
            "table_body": [["Two", "Cols"]],  # width 2 ≠ width 3
        },
    ]
    md = mod._merge_to_markdown(blocks)
    # Not glued onto the 3-col grading table's body.
    assert "| Two | Cols |  |" not in md
