"""Chunk-distribution validators. Use chunker_v4 flags (OVERSIZED, NO_HEADER,
HAS_TABLE) rather than recomputing thresholds — single source of truth."""

from __future__ import annotations

from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome


def check_chunk_count_nonzero(chunks: list[Any]) -> CheckOutcome:
    return CheckOutcome(
        passed=len(chunks) > 0,
        metadata={"chunk_count": len(chunks)},
    )


def check_no_oversized_chunks(chunks: list[dict]) -> CheckOutcome:
    oversized = sum(1 for c in chunks if "OVERSIZED" in (c.get("flags") or []))
    return CheckOutcome(
        passed=oversized == 0,
        metadata={"oversized_count": oversized, "total": len(chunks)},
    )


def check_low_no_header_rate(chunks: list[dict], max_rate: float = 0.10) -> CheckOutcome:
    if not chunks:
        return CheckOutcome(passed=True, metadata={"no_header_rate": 0.0, "max_rate": max_rate})
    no_header = sum(1 for c in chunks if "NO_HEADER" in (c.get("flags") or []))
    rate = no_header / len(chunks)
    return CheckOutcome(
        passed=rate <= max_rate,
        metadata={
            "no_header_rate": round(rate, 4),
            "no_header_count": no_header,
            "total": len(chunks),
            "max_rate": max_rate,
        },
    )
