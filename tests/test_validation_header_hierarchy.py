"""Tests for header hierarchy validation."""

from tamubot.ingestion.validation.header_hierarchy import (
    check_header_hierarchy_valid,
    check_min_headers,
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
