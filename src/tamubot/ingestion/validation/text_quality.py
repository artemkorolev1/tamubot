"""Text quality validators: replacement chars (U+FFFD) and letter-drops.

Letter-drop dictionary mirrors pipeline_v6c.run_bakeoff._LETTER_DROP_TOKENS.
"""

from __future__ import annotations

import re

from tamubot.ingestion.validation.types import CheckOutcome

REPLACEMENT_CHAR = "�"

_LETTER_DROP_TOKENS = (
    r"\bColege\b",
    r"\bSylabus\b",
    r"\bMeting\b",
    r"\bAditional\b",
    r"\bTextbok\b",
    r"\bwil\b",
    r"\bfal\b",
    r"\binstaling\b",
)
_LETTER_DROP_RE = re.compile("|".join(_LETTER_DROP_TOKENS))


def check_no_replacement_chars(text: str) -> CheckOutcome:
    """Pass iff text contains zero U+FFFD characters."""
    count = text.count(REPLACEMENT_CHAR)
    return CheckOutcome(
        passed=count == 0,
        metadata={"replacement_char_count": count},
    )


def check_letter_drops(text: str, threshold: int = 0) -> CheckOutcome:
    """Pass iff letter-drop token count <= threshold (default 0)."""
    matches = _LETTER_DROP_RE.findall(text)
    return CheckOutcome(
        passed=len(matches) <= threshold,
        metadata={
            "letter_drop_count": len(matches),
            "matches": matches[:20],
            "threshold": threshold,
        },
    )


def count_unanswered_labels(
    blocks: list,
    *,
    max_label_len: int = 35,
    max_label_words: int = 4,
) -> CheckOutcome:
    """Deterministic detector for the two-column reading-order split
    (``FID_HEADER_BROKEN``): a short ``Label:`` block whose value Docling orphaned
    far away, leaving the label with no value beside it. Pure — no PyMuPDF.

    A field label is a short text block (``<= max_label_len`` chars,
    ``<= max_label_words`` words, single line) ending in ``:``. It is *answered*
    when the next text block on the same page is a non-label value. A label
    followed by another bare label (the broken column-first run
    ``Course Number:`` / ``Course Title:`` / …) or by a heading is *unanswered*.
    After the adapter's two-column recovery a repaired label reads
    ``Course Number: ECEN 671`` — no trailing ``:`` — so it is not counted; a
    non-zero count means orphaning the recovery did not repair. Catches what
    ``text_coverage`` cannot (the values survive, just disconnected). Shipped WARN
    — calibrate on the corpus before promotion.
    """
    labels = 0
    unanswered = 0
    samples: list[str] = []
    for i, b in enumerate(blocks):
        if b.get("type") != "text":
            continue
        t = (b.get("text") or "").strip()
        if not t.endswith(":") or len(t) < 3 or len(t) > max_label_len:
            continue
        if "\n" in t or len(t.split()) > max_label_words:
            continue
        labels += 1
        page = b.get("page_idx") or 0
        nxt = None
        for ob in blocks[i + 1 :]:
            if (ob.get("page_idx") or 0) != page:
                break
            if ob.get("type") == "heading":
                break  # heading right after a label -> no value beside it
            if ob.get("type") == "text":
                nxt = (ob.get("text") or "").strip()
                break
        answered = bool(nxt) and not nxt.endswith(":")
        if not answered:
            unanswered += 1
            if len(samples) < 20:
                samples.append(t)
    return CheckOutcome(
        passed=unanswered == 0,
        metadata={
            "unanswered_labels": unanswered,
            "total_labels": labels,
            "sample_unanswered": samples,
        },
    )
