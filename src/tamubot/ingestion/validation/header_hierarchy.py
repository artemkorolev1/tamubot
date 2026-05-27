"""Header hierarchy validators. Operate on a list of {level, text} dicts."""

from __future__ import annotations

from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome


def check_header_hierarchy_valid(headers: list[dict[str, Any]]) -> CheckOutcome:
    """Pass iff no header skips more than one level from the previous."""
    skips: list[dict] = []
    prev_level = 0
    for h in headers:
        level = int(h.get("level", 0))
        if prev_level and level > prev_level + 1:
            skips.append({"from": prev_level, "to": level, "text": h.get("text", "")[:60]})
        prev_level = level
    return CheckOutcome(
        passed=len(skips) == 0,
        metadata={"skip_count": len(skips), "skip": skips[:10], "total_headers": len(headers)},
    )


def check_min_headers(headers: list[dict[str, Any]], minimum: int = 2) -> CheckOutcome:
    return CheckOutcome(
        passed=len(headers) >= minimum,
        metadata={"header_count": len(headers), "minimum": minimum},
    )
