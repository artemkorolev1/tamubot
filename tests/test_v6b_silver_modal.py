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
