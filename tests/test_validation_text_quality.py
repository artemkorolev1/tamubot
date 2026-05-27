"""Tests for text quality validators."""

from tamubot.ingestion.validation.text_quality import (
    check_letter_drops,
    check_no_replacement_chars,
)


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
