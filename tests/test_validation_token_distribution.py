"""Tests for chunk-distribution validators using chunker_v4 flags."""

from tamubot.ingestion.validation.token_distribution import (
    check_chunk_count_nonzero,
    check_low_no_header_rate,
    check_no_oversized_chunks,
)


def test_chunk_count_nonzero_pass():
    out = check_chunk_count_nonzero([{}, {}, {}])
    assert out.passed
    assert out.metadata["chunk_count"] == 3


def test_chunk_count_nonzero_fail():
    out = check_chunk_count_nonzero([])
    assert out.passed is False


def test_no_oversized_chunks_pass():
    chunks = [{"flags": []}, {"flags": ["HAS_TABLE"]}]
    assert check_no_oversized_chunks(chunks).passed


def test_no_oversized_chunks_fail():
    chunks = [{"flags": ["OVERSIZED"]}, {"flags": []}]
    out = check_no_oversized_chunks(chunks)
    assert out.passed is False
    assert out.metadata["oversized_count"] == 1


def test_low_no_header_rate_pass():
    chunks = [{"flags": []}] * 9 + [{"flags": ["NO_HEADER"]}]  # 10% exactly
    out = check_low_no_header_rate(chunks, max_rate=0.10)
    assert out.passed


def test_low_no_header_rate_fail():
    chunks = [{"flags": []}] * 8 + [{"flags": ["NO_HEADER"]}] * 2  # 20%
    out = check_low_no_header_rate(chunks, max_rate=0.10)
    assert out.passed is False
    assert out.metadata["no_header_rate"] == 0.20
