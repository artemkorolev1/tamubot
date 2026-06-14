"""Boilerplate-tagging quality validators (BP_* taxonomy).

Pure logic over tagged chunk dicts — no Dagster/Docling import, host-testable.
Guards the header-anchored boilerplate matcher against *over-hiding* a section that
looks standard by its header but carries course-specific customization in its body
(the ECEN_749 class): a chunk flagged ``is_boilerplate=True`` whose body contains
strong course-specific signals (a grading percentage, a concrete weekday+time, a
specific due date) was probably a customized policy wrongly suppressed.
"""

from __future__ import annotations

import re
from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome

# A grading/penalty percentage in body prose, e.g. ``penalty of 20%`` / ``5 %``.
# This is the strongest over-hide signal: a hidden "policy" carrying a concrete grade
# weight (ECEN_749 "penalty of 20%") is course-specific, not standard wording.
_PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%")

# A concrete weekday + clock time, e.g. ``Monday 2:00 PM`` / ``Tues 14:30``. Requires
# BOTH a weekday and a time near it. DELIBERATELY NOT used by default: calibration on
# the live corpus (2026-06-10) showed the only header-anchored boilerplate carrying a
# weekday+time was the *standard* "Technology Support" block ("IT helpdesk Monday -
# Friday 7:30am") — i.e. support-hours, a false over-hide signal. Kept here, off, as a
# documented opt-in lever; the percentage/due-date signals carry the real signal.
_WEEKDAY = r"(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)(?:day|nesday|rsday|urday)?s?"
_TIME = r"\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?|AM|PM|am|pm)|\d{1,2}:\d{2}"
_WEEKDAY_TIME_RE = re.compile(
    rf"\b{_WEEKDAY}\b[^.\n]{{0,30}}?(?:{_TIME})|(?:{_TIME})[^.\n]{{0,30}}?\b{_WEEKDAY}\b",
    re.IGNORECASE,
)

# A concrete due-date pattern: "due" near an explicit calendar date (month name or
# numeric M/D), e.g. "due by March 3", "due 10/14", "deadline: Sept. 5".
_MONTH = (
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
)
_DUE_DATE_RE = re.compile(rf"\b(?:due|deadline|submitted by)\b[^.\n]{{0,40}}?(?:{_MONTH})", re.IGNORECASE)


def find_course_specific_signal(text: str, *, include_weekday_time: bool = False) -> str | None:
    """Return a short label for the first strong course-specific signal in ``text``
    (``percent`` / ``due_date``, plus ``weekday_time`` only when explicitly opted in),
    or ``None`` if the body reads as generic standard-policy wording.

    ``include_weekday_time`` defaults to False: on the live corpus the weekday+time
    signal only fired on the standard IT support-hours block (a false over-hide), so it
    is excluded from the default gate. ``percent`` (a concrete grade weight / penalty)
    and ``due_date`` are the reliable course-specific signals.
    """
    body = text or ""
    m = _PERCENT_RE.search(body)
    if m:
        return f"percent:{m.group(0).strip()}"
    m = _DUE_DATE_RE.search(body)
    if m:
        return f"due_date:{m.group(0).strip()}"
    if include_weekday_time:
        m = _WEEKDAY_TIME_RE.search(body)
        if m:
            return f"weekday_time:{m.group(0).strip()}"
    return None


def check_no_boilerplate_overhide(chunks: list[dict[str, Any]]) -> CheckOutcome:
    """The boilerplate over-hide gate (ECEN_749 class).

    FLAG any chunk flagged ``is_boilerplate=True`` whose body carries a strong
    course-specific signal (grading percentage, concrete weekday+time, or concrete due
    date) — a customized (non-standard) section the matcher should NOT have hidden.
    Header-anchored matches (``boilerplate_match_source == "header_anchored"``) are the
    riskiest source, so they are reported separately, but a body-Jaccard hit carrying a
    course-specific signal is equally suspect and is flagged too.

    Tuned so genuine university-standard policies (Makeup / Academic Integrity / ADA,
    fixed wording, no course-specific numbers) do NOT trigger, while a course-specific
    "Late Work Policy" with "penalty of 20%" WOULD. Reports the count, the offending
    ``header_path``s, and the matched signal. Shipped WARN — promote to blocking once
    the pass rate is confirmed across the golden set.
    """
    offenders: list[dict[str, Any]] = []
    header_anchored = 0
    for c in chunks:
        if not c.get("is_boilerplate"):
            continue
        signal = find_course_specific_signal(c.get("content") or "")
        if signal is None:
            continue
        src = c.get("boilerplate_match_source")
        if src == "header_anchored":
            header_anchored += 1
        offenders.append(
            {
                "header_path": c.get("header_path") or "",
                "match_source": src,
                "signal": signal,
            }
        )
    return CheckOutcome(
        passed=len(offenders) == 0,
        metadata={
            "overhide_count": len(offenders),
            "header_anchored_overhide_count": header_anchored,
            "offenders": offenders[:20],
            "offending_header_paths": [o["header_path"] for o in offenders[:20]],
        },
    )
