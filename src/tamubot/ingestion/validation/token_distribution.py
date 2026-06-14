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


def check_no_header_only_chunks(chunks: list[dict]) -> CheckOutcome:
    """The CHUNK_ORPHAN_HEADER gate (STAT_620 class): a chunk whose ``content``,
    after dropping blank lines, consists of NOTHING but markdown header lines (every
    non-blank line starts with ``#``) — a body-less orphan header carrying no answer.

    This is the INVERSE of ``check_low_no_header_rate`` (which flags *header-LESS*
    chunks): here the chunk is all header and no body. Reports the count + the
    offending ``header_path``s. Shipped WARN — promote to blocking once the pass rate
    is confirmed across the golden set.
    """
    offenders: list[str] = []
    for c in chunks:
        lines = [ln.strip() for ln in (c.get("content") or "").splitlines() if ln.strip()]
        if not lines:
            continue  # empty content is a different failure class, not an orphan header
        if all(ln.startswith("#") for ln in lines):
            offenders.append(c.get("header_path") or "")
    return CheckOutcome(
        passed=len(offenders) == 0,
        metadata={
            "header_only_count": len(offenders),
            "total": len(chunks),
            "offending_header_paths": offenders[:20],
        },
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
