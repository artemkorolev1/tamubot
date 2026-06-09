"""Header hierarchy validators. Operate on a list of {level, text} dicts."""

from __future__ import annotations

import re
from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome

# Mirror of ``_INLINE_LABEL_RE`` in ``converters/docling_converter.py`` — kept in
# sync by hand. We cannot import the converter here: it pulls in Docling at module
# import, which is not host-loadable, and this module must stay pure/host-testable.
# A heading whose text is shaped like an inline label ("Instructor: Jane Doe") is
# almost certainly body text that was wrongly promoted to a heading.
_INLINE_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9 /&'\"-]{2,40}:\s.+$")

# A real heading is short and label-like; a long phrase or a sentence-terminated
# line is body text masquerading as a heading.
_MAX_HEADING_WORDS = 12
_SENTENCE_END = (".", "!", "?")


def _is_suspicious_heading(text: str) -> bool:
    """True when a heading's text looks like body text mis-promoted to a heading:
    inline-label-shaped, over-long, or ending in sentence punctuation."""
    t = (text or "").strip()
    if not t:
        return False
    if _INLINE_LABEL_RE.match(t):
        return True
    if len(t.split()) > _MAX_HEADING_WORDS:
        return True
    return t.endswith(_SENTENCE_END)


def count_level_skips(levels: list[int]) -> int:
    """Count forward heading-level skips (level > prev_level + 1)."""
    skips = 0
    prev_level = 0
    for lvl in levels:
        level = int(lvl)
        if prev_level and level > prev_level + 1:
            skips += 1
        prev_level = level
    return skips


def normalize_heading_levels(levels: list[int]) -> list[int]:
    """Stack-based tree-depth re-leveling (skip-free by construction).

    Treat the input levels as a *relative* nesting order and re-emit each
    heading's tree depth. Maintain a monotonic stack of "open" raw levels; for
    each heading pop every open level >= its raw level, then emit
    ``depth = len(stack) + 1`` (capped at H6). The result satisfies
    ``out[i] - out[i-1] <= 1`` for all i — no forward skip — while preserving
    relative nesting (a raw H3 under a raw H1 stays deeper than the H1). Pure,
    O(n), never drops or reorders a heading. Mirrors the HTML5 document-outline
    depth assignment.

    Worked example: ``[1, 3, 3, 2, 4] -> [1, 2, 2, 2, 3]``.
    """
    out: list[int] = []
    stack: list[int] = []  # raw levels currently "open"
    for lvl in levels:
        raw = max(1, int(lvl))
        while stack and stack[-1] >= raw:
            stack.pop()
        depth = len(stack) + 1
        stack.append(raw)
        out.append(min(6, depth))
    return out


def check_header_hierarchy_valid(headers: list[dict[str, Any]]) -> CheckOutcome:
    """Pass iff no header skips more than one level from the previous.

    Post-normalization this is a *post-condition assertion*: the upstream
    re-leveling pass guarantees it holds, so a failure here signals the
    normalizer was bypassed or regressed, not a content problem.
    """
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


def check_header_levels_normalized(headers: list[dict[str, Any]]) -> CheckOutcome:
    """Observability gate: how many raw level-skips the normalizer repaired.

    Reads each heading's pre-normalization ``raw_level`` (set by the bronze
    adapter) and counts the forward skips that were present before re-leveling.
    WARN when > 0 so a human can see the doc's hierarchy was off — this is the
    signal the old blocking-ERROR gate used to give, now non-fatal. Falls back
    to the post-normalization ``level`` when ``raw_level`` is absent (older
    blocks), in which case it reports 0.
    """
    raw_levels = [int(h.get("raw_level", h.get("level", 0)) or 0) for h in headers]
    repaired = count_level_skips(raw_levels)
    sample = []
    prev = 0
    for h in headers:
        lvl = int(h.get("raw_level", h.get("level", 0)) or 0)
        if prev and lvl > prev + 1:
            sample.append({"from": prev, "to": lvl, "text": h.get("text", "")[:60]})
        prev = lvl
    return CheckOutcome(
        passed=True,  # observability only — never fails
        metadata={
            "repaired_skip_count": repaired,
            "repaired_skips": sample[:10],
            "total_headers": len(headers),
        },
    )


def check_min_headers(
    headers: list[dict[str, Any]], minimum: int = 2, *, page_count: int | None = None
) -> CheckOutcome:
    """A multi-page doc with <2 headers (or <2 distinct levels) almost certainly
    had its real headings demoted to body text — a failure no-skip can't see
    (a fully-flat doc has zero skips). Only enforced when page_count > 1."""
    count = len(headers)
    distinct = len({int(h.get("level", 0)) for h in headers})
    enough = count >= minimum
    if page_count is not None and page_count > 1:
        enough = enough and distinct >= 2
    return CheckOutcome(
        passed=enough,
        metadata={
            "header_count": count,
            "distinct_levels": distinct,
            "minimum": minimum,
            "page_count": page_count,
        },
    )


def check_suspicious_heading_rate(headers: list[dict[str, Any]], *, max_rate: float = 0.15) -> CheckOutcome:
    """Catch body text wrongly promoted to a heading (heading-doc G3).

    Counts headings whose text is inline-label-shaped, over-long, or
    sentence-terminated — the signatures of a font-cluster mis-promotion. A high
    rate means the bronze hierarchy is inventing structure from body lines. WARN
    above ``max_rate``; pure observability otherwise (calibrate on the corpus
    before any promotion to blocking).
    """
    texts = [str(h.get("text", "")) for h in headers]
    suspicious = [t for t in texts if _is_suspicious_heading(t)]
    total = len(texts)
    rate = (len(suspicious) / total) if total else 0.0
    return CheckOutcome(
        passed=rate <= max_rate,
        metadata={
            "suspicious_heading_count": len(suspicious),
            "total_headers": total,
            "suspicious_rate": round(rate, 4),
            "max_rate": max_rate,
            "sample": [t[:60] for t in suspicious[:10]],
        },
    )
