"""Tests for text quality validators."""

from tamubot.ingestion.validation.text_quality import (
    check_letter_drops,
    check_no_replacement_chars,
    count_unanswered_labels,
    find_fabricated_links,
)


# --- Check 1: fabricated mailto/URL links (FID_HALLUCINATION / ISEN_633) -------


def test_fabricated_link_flags_mailto_already_in_plain_text():
    """The ISEN_633 signature: a [label](mailto:X) link whose target already appears
    as plain text in another block -> fabricated."""
    blocks = [
        {"type": "text", "text": "Contact the instructor at jiang@tamu.edu for questions."},
        {"type": "text", "text": "Email: [jiang@tamu.edu](mailto:jiang@tamu.edu)"},
    ]
    out = find_fabricated_links(blocks)
    assert not out.passed
    assert out.metadata["fabricated_link_count"] == 1
    assert "jiang@tamu.edu" in out.metadata["sample_fabricated"][0]


def test_fabricated_link_flags_url_already_in_plain_text():
    blocks = [
        {"type": "text", "text": "See the rules at student-rules.tamu.edu/rule07 before submitting."},
        {"type": "text", "text": "[Student Rule 7](https://student-rules.tamu.edu/rule07)"},
    ]
    out = find_fabricated_links(blocks)
    assert not out.passed
    assert out.metadata["fabricated_link_count"] == 1


def test_genuine_link_not_in_plain_text_passes():
    """A real link whose target appears nowhere else as plain text is legitimate."""
    blocks = [
        {"type": "text", "text": "Course homepage: [Canvas](https://canvas.tamu.edu/courses/12345)"},
        {"type": "text", "text": "Office hours are Tuesdays."},
    ]
    out = find_fabricated_links(blocks)
    assert out.passed
    assert out.metadata["fabricated_link_count"] == 0
    assert out.metadata["total_links"] == 1


def test_no_links_at_all_passes():
    blocks = [{"type": "text", "text": "Plain prose with no markdown links at all."}]
    out = find_fabricated_links(blocks)
    assert out.passed
    assert out.metadata["total_links"] == 0


def test_unanswered_labels_flags_orphaned_run():
    """The broken column-first run: consecutive bare labels with values orphaned."""
    blocks = [
        {"type": "text", "text": "Course Number:", "page_idx": 1},
        {"type": "text", "text": "Course Title:", "page_idx": 1},
        {"type": "text", "text": "Section:", "page_idx": 1},
    ]
    out = count_unanswered_labels(blocks)
    assert not out.passed
    assert out.metadata["unanswered_labels"] == 3


def test_unanswered_labels_clean_after_recovery():
    """Repaired labels carry their value inline -> not counted; adjacent values pass."""
    blocks = [
        {"type": "text", "text": "Course Number: ECEN 671", "page_idx": 1},
        {"type": "text", "text": "Instructor:", "page_idx": 1},
        {"type": "text", "text": "Prof. H.R. Harris", "page_idx": 1},
    ]
    out = count_unanswered_labels(blocks)
    assert out.passed
    assert out.metadata["unanswered_labels"] == 0


def test_unanswered_labels_ignores_long_lead_in_sentence():
    """A prose lead-in ending in ':' (not a short field label) is not counted."""
    blocks = [
        {"type": "text", "text": "The expected learning outcomes are as follows:", "page_idx": 1},
        {"type": "text", "text": "Students will be able to analyze devices.", "page_idx": 1},
    ]
    out = count_unanswered_labels(blocks)
    assert out.metadata["total_labels"] == 0
    assert out.passed


def test_unanswered_label_followed_by_heading_flagged():
    blocks = [
        {"type": "text", "text": "Location:", "page_idx": 1},
        {"type": "heading", "text": "Course Description", "page_idx": 1},
    ]
    out = count_unanswered_labels(blocks)
    assert out.metadata["unanswered_labels"] == 1


def test_no_replacement_chars_pass():
    out = check_no_replacement_chars("normal text with no issues")
    assert out.passed is True
    assert out.metadata["replacement_char_count"] == 0


def test_no_replacement_chars_fail():
    out = check_no_replacement_chars("broken text � here")
    assert out.passed is False
    assert out.metadata["replacement_char_count"] == 1


def test_no_replacement_chars_multiple():
    out = check_no_replacement_chars("���")
    assert out.passed is False
    assert out.metadata["replacement_char_count"] == 3


def test_letter_drops_pass():
    out = check_letter_drops("College Meeting Office")
    assert out.passed is True
    assert out.metadata["letter_drop_count"] == 0


def test_letter_drops_fail():
    out = check_letter_drops("Colege Meting Off ce")
    assert out.passed is False
    assert out.metadata["letter_drop_count"] >= 2
    assert "Colege" in out.metadata["matches"]


def test_letter_drops_threshold_zero_strict():
    """Default threshold is 0 — even a single drop fails."""
    out = check_letter_drops("only one Colege")
    assert out.passed is False
