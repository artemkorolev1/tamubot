"""Tests for the PDF→bronze text-coverage validator."""

from tamubot.ingestion.validation.text_coverage import compute_text_coverage, content_tokens


def test_content_tokens_keeps_urls_isbns_emails():
    toks = content_tokens("ISBN: 978-0-387-69957-8 see https://cesg.tamu.edu/x or a@b.edu")
    assert "978-0-387-69957-8" in toks
    assert "https://cesg.tamu.edu/x" in toks
    assert "a@b.edu" in toks
    assert "isbn" in toks


def test_content_tokens_drops_short_noise():
    toks = content_tokens("a 5 of in to 42")
    assert toks == set()  # all under length 4


def test_full_coverage_passes():
    pdf = "Instructor Jane Doe teaches SystemC modeling"
    bronze = "Instructor Jane Doe teaches SystemC modeling"
    out = compute_text_coverage(pdf, bronze)
    assert out.passed
    assert out.metadata["missing_count"] == 0
    assert out.metadata["coverage"] == 1.0


def test_dropped_isbn_value_flagged():
    """The ISEN/ECEN failure shape: label survives, value dropped."""
    pdf = "Textbook SystemC From the Ground Up ISBN: 978-0-387-69957-8 Optional"
    bronze = "Textbook SystemC From the Ground Up ISBN: Optional"  # value gone
    out = compute_text_coverage(pdf, bronze, max_missing_rate=0.0)
    assert out.passed is False
    assert "978-0-387-69957-8" in out.metadata["sample_missing"]


def test_dropped_url_flagged():
    pdf = "Webpage https://cesg.tamu.edu/faculty/jiang-hu/ Catalog Description"
    bronze = "Catalog Description"
    out = compute_text_coverage(pdf, bronze)
    assert out.passed is False
    # trailing slash is normalized off (consistently on both sides) for matching
    assert "https://cesg.tamu.edu/faculty/jiang-hu" in out.metadata["sample_missing"]


def test_reflow_does_not_false_flag():
    """Docling re-orders/merges text — token-set comparison must not flag that."""
    pdf = "alpha beta gamma delta epsilon"
    bronze = "epsilon delta\n\ngamma\nalpha beta"  # same tokens, reordered + reflowed
    out = compute_text_coverage(pdf, bronze)
    assert out.passed
    assert out.metadata["missing_count"] == 0


def test_empty_pdf_passes():
    out = compute_text_coverage("", "anything")
    assert out.passed
    assert out.metadata["pdf_tokens"] == 0


def test_threshold_band():
    # 1 of 5 tokens missing = 20% missing.
    pdf = "alpha beta gamma delta epsilon"
    bronze = "alpha beta gamma delta"  # epsilon dropped
    assert compute_text_coverage(pdf, bronze, max_missing_rate=0.25).passed
    assert compute_text_coverage(pdf, bronze, max_missing_rate=0.10).passed is False
