"""Source-integrity validators for v6b bronze. Pure decision function +
a pypdfium2 page-count helper. Catches scanned/corrupt PDFs that produce
empty markdown under do_ocr=False."""

from __future__ import annotations

from pathlib import Path

from tamubot.ingestion.validation.types import CheckOutcome


def check_pdf_integrity(
    page_count: int,
    markdown_chars: int,
    min_chars_per_page: int = 50,
) -> CheckOutcome:
    """Pass iff the PDF has >0 pages and extracted text density is plausible."""
    if page_count <= 0:
        return CheckOutcome(
            passed=False,
            metadata={"reason": "zero_pages", "page_count": page_count},
        )
    chars_per_page = markdown_chars / page_count
    if chars_per_page < min_chars_per_page:
        return CheckOutcome(
            passed=False,
            metadata={
                "reason": "low_text_extractability",
                "page_count": page_count,
                "markdown_chars": markdown_chars,
                "chars_per_page": round(chars_per_page, 2),
                "min_chars_per_page": min_chars_per_page,
            },
        )
    return CheckOutcome(
        passed=True,
        metadata={
            "page_count": page_count,
            "markdown_chars": markdown_chars,
            "chars_per_page": round(chars_per_page, 2),
        },
    )


def count_pdf_pages(pdf_path: Path) -> int:
    """Page count via pypdfium2; 0 if the file is unreadable/corrupt."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return 0
