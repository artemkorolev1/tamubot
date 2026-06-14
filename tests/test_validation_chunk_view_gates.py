"""Tests for the end-to-end chunk-view fidelity validators (G-A/G-B/G-C/G-D).

Pure — no GPU/Dagster/PyMuPDF. These guard the chunk view RAG actually retrieves:
bare image markers (FID_IMAGE_LOST), table survival across the merge+chunk steps
(FID_TABLE_LOST), and garbled link anchors (FID_REPLACEMENT_CHARS). The chunk-view
content-coverage gate (G-C) reuses ``compute_text_coverage``, covered in
``test_validation_text_coverage.py``.
"""

from tamubot.ingestion.validation.image_quality import (
    check_no_bare_image_markers,
    count_bare_image_markers,
)
from tamubot.ingestion.validation.table_quality import check_tables_survive_view
from tamubot.ingestion.validation.text_quality import find_garbled_link_anchors

# --- G-A: bare image markers in the chunk view ----------------------------


def test_count_bare_image_markers_counts_only_bare():
    view = (
        "## Schedule\n<!-- image -->\n"
        "![grading rubric](#) <!-- 40% homework 60% exams -->\n"  # has description -> not bare
        "![logo](#)\n"  # has caption -> not bare
        "<!-- image -->\n"
    )
    assert count_bare_image_markers(view) == 2


def test_bare_image_markers_clean_passes():
    out = check_no_bare_image_markers("![figure](#) <!-- a real description -->")
    assert out.passed
    assert out.metadata["bare_image_markers"] == 0


def test_bare_image_markers_flagged_with_rate():
    view = "## Topical outline\n<!-- image -->\n<!-- image -->\n"
    out = check_no_bare_image_markers(view, bronze_image_count=4)
    assert out.passed is False
    assert out.metadata["bare_image_markers"] == 2
    assert out.metadata["bronze_image_count"] == 4
    assert out.metadata["bare_marker_rate"] == 0.5


def test_bare_image_markers_zero_bronze_images_no_div_by_zero():
    out = check_no_bare_image_markers("<!-- image -->", bronze_image_count=0)
    assert out.passed is False
    assert out.metadata["bare_marker_rate"] == 0.0


# --- G-B: tables survive into the chunk view ------------------------------


def _table_block(rows, *, page=3, caption=""):
    return {"type": "table", "page_idx": page, "table_body": rows, "table_caption": caption}


def test_table_present_in_view_passes():
    block = _table_block([["Week", "Topic"], ["1", "Introduction to modeling"]])
    view = "## Schedule\n| Week | Topic |\n| --- | --- |\n| 1 | Introduction to modeling |\n"
    out = check_tables_survive_view([block], view)
    assert out.passed
    assert out.metadata["evaluated_tables"] == 1
    assert out.metadata["lost_table_count"] == 0


def test_table_dropped_from_view_flagged():
    """A grading table whose rows never reached any chunk (empty merge → lost)."""
    block = _table_block(
        [["Component", "Weight"], ["Midterm", "30 percent"], ["Final", "40 percent"]],
        caption="Grading breakdown",
    )
    view = "## Course Description\nThis course covers system modeling.\n"  # table content absent
    out = check_tables_survive_view([block], view)
    assert out.passed is False
    assert out.metadata["lost_table_count"] == 1
    assert "midterm" in out.metadata["offenders"][0]["sample_missing"]


def test_tiny_table_skipped():
    """A near-empty grid (< min_table_tokens) is not evaluated — avoids noise."""
    block = _table_block([["x"], [""]])
    out = check_tables_survive_view([block], "nothing here")
    assert out.passed
    assert out.metadata["evaluated_tables"] == 0


def test_non_table_blocks_ignored():
    blocks = [{"type": "text", "text": "hello world this is body prose"}]
    out = check_tables_survive_view(blocks, "unrelated")
    assert out.passed
    assert out.metadata["evaluated_tables"] == 0


def test_partial_table_loss_under_threshold_passes():
    """Most cells survive (token normalization noise) — should not flag at 50%."""
    block = _table_block([["Week", "Topic"], ["1", "Intro modeling basics"], ["2", "Advanced topics"]])
    # All content tokens present, reformatted/reordered.
    view = "Topic Intro modeling basics Advanced Week topics 1 2"
    out = check_tables_survive_view([block], view)
    assert out.passed


# --- G-D: garbled link anchors --------------------------------------------


def test_clean_anchors_pass():
    view = "See the [course website](https://cs.tamu.edu) and [Student Rules](https://student-rules.tamu.edu)."
    out = find_garbled_link_anchors(view)
    assert out.passed
    assert out.metadata["total_links"] == 2


def test_lowercase_prose_anchor_flagged():
    view = "[keup work for an (Student Rule 7)](https://student-rules.tamu.edu/rule07/)"
    out = find_garbled_link_anchors(view)
    assert out.passed is False
    assert out.metadata["garbled_anchor_count"] == 1
    reasons = out.metadata["offenders"][0]["reasons"]
    assert "lowercase_prose" in reasons or "unbalanced_parens" in reasons


def test_midsentence_break_anchor_flagged():
    """A sentence boundary continuing in lowercase = spilled prose."""
    view = "[see the policy. students must comply with all rules](https://rules.tamu.edu)"
    out = find_garbled_link_anchors(view)
    assert out.passed is False
    assert "midsentence_break" in out.metadata["offenders"][0]["reasons"]


def test_abbreviation_and_org_name_anchors_not_flagged():
    """``U.S.`` abbreviation and a Title-Case org name with a trailing period are not
    garbled prose — the lowercase-continuation requirement skips them."""
    view = (
        "[U.S. Department of Education Office for Civil Rights](https://ed.gov) and "
        "[University Health Services.](https://uhs.tamu.edu)"
    )
    out = find_garbled_link_anchors(view)
    assert out.passed


def test_bare_domain_label_not_flagged():
    """``988lifeline.org.`` is a bare-hostname autolink, not a garbled anchor."""
    out = find_garbled_link_anchors("[988lifeline.org.](https://988lifeline.org)")
    assert out.passed


def test_unbalanced_parens_anchor_flagged():
    view = "[Office Hours (by appointment](mailto:prof@tamu.edu)"
    out = find_garbled_link_anchors(view)
    assert out.passed is False
    assert "unbalanced_parens" in out.metadata["offenders"][0]["reasons"]


def test_short_lowercase_anchor_not_flagged():
    """``course website`` is lowercase but under the prose-word floor — must pass."""
    out = find_garbled_link_anchors("[course website](https://cs.tamu.edu)")
    assert out.passed


def test_autolink_url_label_not_flagged():
    """A bare-URL label ending in ``?`` (zoom query string) is an autolink, not garbled."""
    view = "[https://tamu.zoom.us/j/99872586429?](https://tamu.zoom.us/j/99872586429?pwd=abc.1)"
    out = find_garbled_link_anchors(view)
    assert out.passed
    assert out.metadata["total_links"] == 1


def test_autolink_email_label_with_trailing_period_not_flagged():
    out = find_garbled_link_anchors("[civilrights@tamu.edu.](mailto:civilrights@tamu.edu)")
    assert out.passed


def test_no_links_passes():
    out = find_garbled_link_anchors("Plain text with no markdown links at all.")
    assert out.passed
    assert out.metadata["total_links"] == 0
