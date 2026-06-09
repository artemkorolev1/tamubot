"""Host-testable unit tests for the three v6b parsing-QC fixes:

  Issue 1 — metadata sections survive into v6b chunks (chunker_v4)
  Issue 2 — table degeneracy detection + graceful-degradation ladder (table_quality)
  Issue 3 — stack-based heading re-leveling guarantees a skip-free hierarchy

All pure Python — no Docling / inspect_ai / GPU. The Issue-3 fixtures are the real
heading-level sequences of the three documents that were dropped in a prior run
(two ISEN, one CSCE), embedded here because data/ is gitignored.
"""

from __future__ import annotations

from tamubot.ingestion.chunker_v4 import chunk_semantic
from tamubot.ingestion.validation.header_hierarchy import (
    count_level_skips,
    normalize_heading_levels,
)
from tamubot.ingestion.validation.table_quality import (
    OUTCOME_GRID_FALLBACK,
    OUTCOME_LOST,
    OUTCOME_PARTIAL_KEPT,
    OUTCOME_TRANSCRIBED,
    is_degenerate_table,
    is_failure_marker,
    looks_like_table,
    render_table_markdown,
    summarize_table_outcomes,
    table_row_count,
)

# ── Issue 3: stack-based heading re-leveling ────────────────────────────────

# Real level sequences of the three docs dropped by the old blocking-ERROR gate.
DROPPED_DOC_LEVELS = {
    "202611_ISEN_667_600_54399": [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4],
    "202611_ISEN_665_600_59566": [1, 1, 2, 2, 2, 2, 2, 4, 4],
    "202631_CSCE_704_600_59750": [
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        3,
        2,
        4,
        2,
        2,
        2,
        3,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
    ],
}


def test_normalize_worked_example():
    # The canonical H1->H3 failing case from the research doc.
    assert normalize_heading_levels([1, 3, 3, 2, 4]) == [1, 2, 2, 2, 3]


def test_normalize_is_skip_free_and_length_preserving():
    for name, levels in DROPPED_DOC_LEVELS.items():
        assert count_level_skips(levels) > 0, f"{name} fixture should have a raw skip"
        out = normalize_heading_levels(levels)
        assert len(out) == len(levels), name
        assert count_level_skips(out) == 0, f"{name} still has a skip after normalization"


def test_normalize_preserves_relative_nesting():
    # A raw-deeper heading must never become shallower than its predecessor's depth.
    levels = [1, 2, 4, 2, 3]
    out = normalize_heading_levels(levels)
    assert out == [1, 2, 3, 2, 3]


def test_normalize_caps_at_h6():
    assert normalize_heading_levels([1, 2, 3, 4, 5, 6, 7, 8]) == [1, 2, 3, 4, 5, 6, 6, 6]


def test_normalize_handles_empty_and_single():
    assert normalize_heading_levels([]) == []
    assert normalize_heading_levels([3]) == [1]


# ── Issue 1: metadata sections kept in v6b chunks ───────────────────────────

_MD = """# CSCE 633 Course Syllabus

## Course Information

Instructor: Dr. Smith
Email: smith@tamu.edu
Meeting: MWF 10:00-10:50, ZACH 350
Credit Hours: 3

## Course Description

This course covers machine learning fundamentals and practical applications
across supervised and unsupervised settings, with weekly programming work.

## Course Prerequisites

CSCE 221 and STAT 211, or equivalent background in data structures and stats.
"""


def test_metadata_dropped_by_default_v4_behavior():
    chunks, log = chunk_semantic(_MD)  # default drop_metadata_sections=True
    text = "\n".join(c["content"] for c in chunks)
    assert "smith@tamu.edu" not in text
    assert log["dropped_metadata"] >= 1


def test_metadata_kept_when_disabled_v6b_behavior():
    chunks, log = chunk_semantic(_MD, drop_metadata_sections=False)
    text = "\n".join(c["content"] for c in chunks)
    # The contact/meeting/prereq facts students ask about must survive.
    assert "smith@tamu.edu" in text
    assert "MWF 10:00-10:50, ZACH 350" in text
    assert "CSCE 221 and STAT 211" in text
    assert log["dropped_metadata"] == 0


# Sparse top-level leading sections, each individually below MIN_CHUNK_TOKENS=50,
# with no prior chunk to merge backward into — the CSCE_685 / STAT_620 drop
# pattern (after heading re-leveling these sit at the top with no parent body).
_MD_TINY_LEADING = """## Course Information

Credit Hours: 1 TO 12

## Instructor Details

Duncan Walker
Email: d-walker@tamu.edu

## Course Description

This directed-studies course lets a student pursue an approved research topic
under faculty supervision, producing a written report and a final presentation
that together demonstrate graduate-level mastery of the chosen subject area, and
includes weekly advisor meetings plus a literature review submitted at midterm.
"""


def test_tiny_leading_sections_merge_forward_not_dropped():
    # At the v6b floor (50), each leading section is sub-min and sits at the top
    # with no previous chunk; they must merge FORWARD, never drop.
    chunks, log = chunk_semantic(_MD_TINY_LEADING, min_chunk_tokens=50, drop_metadata_sections=False)
    text = "\n".join(c["content"] for c in chunks)
    assert "Credit Hours: 1 TO 12" in text
    assert "d-walker@tamu.edu" in text
    assert "Duncan Walker" in text
    assert log["dropped_tiny"] == 0
    assert log["merged_forward"] >= 1


def test_all_sub_min_doc_keeps_content():
    # Pathological: the whole doc is below the floor — must still emit, never drop.
    md = "## Course Information\n\nCredit Hours: 3\n\n## Instructor\n\nDr. Smith\n"
    chunks, log = chunk_semantic(md, min_chunk_tokens=50, drop_metadata_sections=False)
    text = "\n".join(c["content"] for c in chunks)
    assert chunks, "must emit at least one chunk rather than drop everything"
    assert "Credit Hours: 3" in text
    assert "Dr. Smith" in text
    assert log["dropped_tiny"] == 0


# ── Issue 2: table degeneracy + graceful-degradation ladder ─────────────────


def test_degenerate_table_detected():
    loop = "| Week | Topic |\n| --- | --- |\n" + "\n".join("| | | | | | |" for _ in range(10))
    assert is_degenerate_table(loop)


def test_dense_table_not_flagged_degenerate():
    dense = (
        "| Day | Time | Room |\n| --- | --- | --- |\n"
        "| Mon | 10:00 | 350 |\n| Wed | 10:00 | 350 |\n| Fri | 10:00 | 350 |"
    )
    assert not is_degenerate_table(dense)


def test_sparse_small_table_not_flagged():
    # A legitimately sparse 2-col table must not trip the detector.
    small = "| A | B |\n| --- | --- |\n| 1 | |\n| | 2 |"
    assert not is_degenerate_table(small)


def test_table_row_count():
    gfm = "| H1 | H2 |\n| --- | --- |\n| a | b |\n| c | d |"
    assert table_row_count(gfm) == 2


def test_ladder_transcribed():
    block = {"type": "table", "modal_result": {"table_markdown": "| x |\n| --- |\n| 1 |"}}
    md, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_TRANSCRIBED
    assert "| x |" in md


def test_ladder_grid_fallback_when_vlm_empty():
    block = {"type": "table", "table_body": [["Item", "Pct"], ["Exam", "50"]], "modal_result": {"table_markdown": ""}}
    md, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_GRID_FALLBACK
    assert "| Exam | 50 |" in md


def _grid(n_rows: int) -> list[list[str]]:
    # Header + n_rows data rows of a Week/Topic schedule.
    return [["Week", "Topic"]] + [[str(i), f"Module {i}"] for i in range(1, n_rows + 1)]


def test_ladder_prefers_grid_when_vlm_truncated():
    # The CSCE_611/682 bug: VLM emits a CLEAN but truncated 2-row table while
    # Docling's grid captured 12 rows. Row-count conservation must pick the grid.
    vlm_short = "| Week | Topic |\n| --- | --- |\n| 1 | Module 1 |\n| 2 | Module 2 |"
    block = {
        "type": "table",
        "table_body": _grid(12),
        "modal_result": {"table_markdown": vlm_short},
    }
    md, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_GRID_FALLBACK
    assert table_row_count(md) == 12
    assert "| 12 | Module 12 |" in md


def test_ladder_keeps_vlm_when_complete():
    # When the VLM is as complete as the grid, keep it (better cell formatting).
    vlm_full = "| Week | Topic |\n| --- | --- |\n" + "\n".join(f"| {i} | Module {i} |" for i in range(1, 13))
    block = {
        "type": "table",
        "table_body": _grid(12),
        "modal_result": {"table_markdown": vlm_full},
    }
    md, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_TRANSCRIBED
    assert table_row_count(md) == 12


def test_summary_counts_vlm_truncation():
    blocks = [
        {
            "type": "table",
            "table_body": _grid(12),
            "modal_result": {"table_markdown": "| Week | Topic |\n| --- | --- |\n| 1 | Module 1 |"},
        }
    ]
    summary = summarize_table_outcomes(blocks)
    assert summary["vlm_truncated_tables"] == 1
    assert summary[OUTCOME_GRID_FALLBACK] == 1
    assert summary[OUTCOME_LOST] == 0


def test_ladder_partial_kept_rescues_degraded_vlm_text():
    # The motivating bug: JSON parse failed, leaving a usable partial in description,
    # and Docling's grid was near-empty. The partial must be kept, not dropped.
    partial = "| Week | Topic |\n| 1 | Intro |\n| 2 | Search |"
    block = {
        "type": "table",
        "table_body": [["Week"]],  # degenerate single-cell grid
        "modal_result": {"table_markdown": "", "description": partial, "confidence": 0.5},
    }
    md, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_PARTIAL_KEPT
    assert "| 2 | Search |" in md


def test_ladder_lost_only_when_nothing_usable():
    block = {"type": "table", "table_body": [], "modal_result": {"table_markdown": "", "description": "garbage"}}
    md, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_LOST
    assert md == ""


def test_ladder_degenerate_partial_not_kept():
    # A degenerate `| |…` loop in description must NOT be rescued as partial.
    loop = "\n".join("| | | | |" for _ in range(10))
    block = {"type": "table", "table_body": [], "modal_result": {"table_markdown": "", "description": loop}}
    _, outcome = render_table_markdown(block)
    assert outcome == OUTCOME_LOST


def test_looks_like_table():
    assert looks_like_table("| a | b |\n| c | d |")
    assert not looks_like_table("just some prose\nwith no pipes")


def test_is_failure_marker():
    assert is_failure_marker("[table processing failed: no JSON object in model output: ...]")
    assert is_failure_marker("[image processing failed: timeout]")
    assert not is_failure_marker("Course schedule")
    assert not is_failure_marker("")


def test_merge_suppresses_failure_marker_caption_but_keeps_grid_rows():
    # A table whose VLM failed (error string in caption) but whose Docling grid
    # rescues the rows: the error string must NOT leak; the rows must survive.
    from tamubot.ingestion.pipeline_v6b.assets import silver_modal as mod

    blocks = [
        {
            "type": "table",
            "table_body": [["Assignment", "Week"], ["Q-learning", "4"]],
            "modal_result": {
                "table_markdown": "",
                "caption": "[table processing failed: no JSON object in model output: ...]",
                "confidence": 0.0,
            },
        }
    ]
    md = mod._merge_to_markdown(blocks)
    assert "processing failed" not in md
    assert "| Q-learning | 4 |" in md


def test_summarize_outcomes():
    blocks = [
        {"type": "table", "modal_result": {"table_markdown": "| x |\n| --- |\n| 1 |"}},
        {"type": "table", "table_body": [["A", "B"], ["1", "2"]], "modal_result": {"table_markdown": ""}},
        {"type": "table", "table_body": [], "modal_result": {"table_markdown": "", "description": ""}},
        {"type": "text", "text": "ignored"},
    ]
    s = summarize_table_outcomes(blocks)
    assert s["table_blocks_total"] == 3
    assert s[OUTCOME_TRANSCRIBED] == 1
    assert s[OUTCOME_GRID_FALLBACK] == 1
    assert s[OUTCOME_LOST] == 1
