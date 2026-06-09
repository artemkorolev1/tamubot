"""Tests for header hierarchy validation."""

from tamubot.ingestion.validation.header_hierarchy import (
    check_header_hierarchy_valid,
    check_min_headers,
    check_suspicious_heading_rate,
)


def test_valid_hierarchy():
    headers = [
        {"level": 1, "text": "Title"},
        {"level": 2, "text": "Sub"},
        {"level": 3, "text": "Sub-sub"},
    ]
    assert check_header_hierarchy_valid(headers).passed


def test_skip_level_fails():
    headers = [
        {"level": 1, "text": "Title"},
        {"level": 3, "text": "Skipped H2"},
    ]
    out = check_header_hierarchy_valid(headers)
    assert out.passed is False
    assert "skip" in out.metadata


def test_empty_headers_passes():
    """No headers = trivially valid hierarchy."""
    assert check_header_hierarchy_valid([]).passed


def test_min_headers_pass():
    headers = [{"level": 1, "text": "a"}, {"level": 2, "text": "b"}]
    out = check_min_headers(headers, minimum=2)
    assert out.passed


def test_min_headers_fail():
    out = check_min_headers([{"level": 1, "text": "a"}], minimum=2)
    assert out.passed is False
    assert out.metadata["header_count"] == 1
    assert out.metadata["minimum"] == 2


def test_suspicious_heading_clean_passes():
    headers = [
        {"text": "Course Information"},
        {"text": "Grading Policy"},
        {"text": "Weekly Schedule"},
    ]
    out = check_suspicious_heading_rate(headers)
    assert out.passed
    assert out.metadata["suspicious_heading_count"] == 0
    assert out.metadata["suspicious_rate"] == 0.0


def test_suspicious_heading_inline_label_flagged():
    """An inline-label-shaped heading ("Instructor: …") is body text mis-promoted."""
    headers = [{"text": "Instructor: Dr. Jane Doe"}]
    out = check_suspicious_heading_rate(headers)
    assert out.metadata["suspicious_heading_count"] == 1


def test_suspicious_heading_long_and_sentence_flagged():
    headers = [
        {"text": "This is a very long heading that clearly runs well past twelve words and is body"},
        {"text": "This ends like a sentence."},
        {"text": "Syllabus"},  # clean
    ]
    out = check_suspicious_heading_rate(headers)
    assert out.metadata["suspicious_heading_count"] == 2
    assert out.metadata["total_headers"] == 3


def test_suspicious_heading_rate_threshold():
    # 1 of 10 suspicious = 10% <= 15% -> pass; 2 of 10 = 20% -> fail.
    nine_clean = [{"text": f"Section {i}"} for i in range(9)]
    out_ok = check_suspicious_heading_rate([*nine_clean, {"text": "Email: x@y.edu"}])
    assert out_ok.passed
    out_bad = check_suspicious_heading_rate(
        [*nine_clean[:-1], {"text": "Email: x@y.edu"}, {"text": "Phone: 555-1234"}]
    )
    assert out_bad.passed is False


def test_suspicious_heading_empty_passes():
    assert check_suspicious_heading_rate([]).passed
