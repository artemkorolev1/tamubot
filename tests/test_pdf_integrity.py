from tamubot.ingestion.validation.pdf_integrity import check_pdf_integrity


def test_zero_pages_fails():
    out = check_pdf_integrity(page_count=0, markdown_chars=0)
    assert out.passed is False
    assert out.metadata["reason"] == "zero_pages"


def test_scanned_pdf_low_text_fails():
    # 10 pages but almost no extracted text -> likely scanned image PDF
    out = check_pdf_integrity(page_count=10, markdown_chars=120, min_chars_per_page=50)
    assert out.passed is False
    assert out.metadata["reason"] == "low_text_extractability"


def test_healthy_pdf_passes():
    out = check_pdf_integrity(page_count=5, markdown_chars=4000, min_chars_per_page=50)
    assert out.passed is True
    assert out.metadata["chars_per_page"] == 800.0
