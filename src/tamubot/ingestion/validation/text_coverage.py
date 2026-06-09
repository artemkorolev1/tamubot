"""Text-coverage validator: how much of the source PDF's text survived into the
bronze extraction.

Pure logic (no PyMuPDF / Docling / Dagster import) — host-testable. The caller
supplies the two strings: the raw PDF plaintext (PyMuPDF ``page.get_text()``,
which reliably has the full text layer) and the bronze block text (what Docling
produced and RAG ultimately sees). A high *missing* rate means Docling dropped
content during extraction — the generic signature of the ``FID_CONTENT_DROPPED``
failures (dropped URLs, label values like ``ISBN: 978-…``, table cells) that the
sub-agent judge otherwise has to find by eye.

This compares token *sets*, not lines, so it is robust to Docling reflowing or
re-ordering text — it only asks "did this content token survive somewhere?".
"""

from __future__ import annotations

import re

from tamubot.ingestion.validation.types import CheckOutcome

# A content token: starts alnum, length >= 4, may contain url/isbn/email punctuation
# so "https://cesg.tamu.edu/..", "978-0-387-69957-8" and "jiang@tamu.edu" stay whole.
# The >=4 floor drops page numbers and one/two-letter noise that Docling legitimately
# reflows or omits (running headers, bullets) without flagging it as content loss.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./:@_+#-]{3,}")


def content_tokens(text: str) -> set[str]:
    """Lowercased set of content tokens (len >= 4) in ``text``. Trailing url/sentence
    punctuation is stripped so ``daily.`` and ``daily`` match."""
    out: set[str] = set()
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0).lower().strip("./:-")
        if len(tok) >= 4:
            out.add(tok)
    return out


def compute_text_coverage(
    pdf_text: str,
    bronze_text: str,
    *,
    max_missing_rate: float = 0.05,
) -> CheckOutcome:
    """Fraction of source-PDF content tokens missing from the bronze extraction.

    ``passed`` iff ``missing_rate <= max_missing_rate``. An empty PDF (no tokens)
    trivially passes. ``sample_missing`` lists up to 20 dropped tokens so a human
    sees *what* was lost (e.g. an ISBN, a URL, a schedule cell) without re-reading
    the PDF.

    Calibration (30-stem v6b eval set, post URL-recovery): coverage runs
    0.978–0.999 (median 0.998). The residual is dominated by *tokenization noise*
    — e.g. the PDF's colon-joined ``A:100-90`` grade band vs bronze's spaced
    ``A: 100-90`` (``100-90`` survives, the joined form looks missing). So the
    rate gate is deliberately loose (``0.05`` = catch only gross loss like a
    dropped page/section); the per-token signal lives in ``sample_missing``, which
    is a triage aid, not ground truth. Shipped as WARN — re-calibrate before any
    promotion to blocking.
    """
    pdf = content_tokens(pdf_text)
    if not pdf:
        return CheckOutcome(
            passed=True,
            metadata={
                "pdf_tokens": 0,
                "bronze_tokens": len(content_tokens(bronze_text)),
                "missing_count": 0,
                "missing_rate": 0.0,
                "coverage": 1.0,
                "max_missing_rate": max_missing_rate,
                "sample_missing": [],
            },
        )
    bronze = content_tokens(bronze_text)
    missing = pdf - bronze
    missing_rate = len(missing) / len(pdf)
    return CheckOutcome(
        passed=missing_rate <= max_missing_rate,
        metadata={
            "pdf_tokens": len(pdf),
            "bronze_tokens": len(bronze),
            "missing_count": len(missing),
            "missing_rate": round(missing_rate, 4),
            "coverage": round(1 - missing_rate, 4),
            "max_missing_rate": max_missing_rate,
            "sample_missing": sorted(missing)[:20],
        },
    )
