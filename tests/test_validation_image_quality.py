"""Unit tests for image/layout fidelity validators (image_quality.py)."""

from __future__ import annotations

from tamubot.ingestion.validation.image_quality import (
    check_content_bearing_images,
    check_no_ocr_failure_page,
)


def _img(page, bbox, caption="", footnote=""):
    return {
        "type": "image",
        "page_idx": page,
        "bbox": bbox,
        "image_caption": caption,
        "image_footnote": footnote,
    }


def _text(page, text):
    return {"type": "text", "page_idx": page, "text": text}


# ---------- check_content_bearing_images ----------

def test_large_uncaptioned_image_flagged():
    # A half-page schedule-table image (ISEN_665 class): big bbox, no caption.
    blocks = [_img(6, [50, 700, 550, 300])]  # area = 500 * 400 = 200_000
    out = check_content_bearing_images(blocks)
    assert not out.passed
    assert out.metadata["content_image_count"] == 1
    assert out.metadata["offending_pages"] == [6]


def test_small_logo_not_flagged():
    # A course seal/logo (~85x65) is below the content-area floor.
    blocks = [_img(0, [429, 705, 514, 641])]  # area ~ 5_485
    out = check_content_bearing_images(blocks)
    assert out.passed
    assert out.metadata["content_image_count"] == 0


def test_captioned_large_image_not_flagged():
    # A captioned image at least leaves text behind, so it is not "lost".
    blocks = [_img(3, [50, 700, 550, 300], caption="Figure 1: course roadmap")]
    out = check_content_bearing_images(blocks)
    assert out.passed


def test_bbox_area_robust_to_coordinate_order():
    # PDF y often runs bottom-up; area must be the same regardless of order.
    a = check_content_bearing_images([_img(1, [50, 300, 550, 700])])
    b = check_content_bearing_images([_img(1, [50, 700, 550, 300])])
    assert a.metadata["largest_area"] == b.metadata["largest_area"] == 200_000.0


# ---------- check_no_ocr_failure_page ----------

def test_image_only_page_flagged():
    # CSCE_704 class: a page with a big image and no text = trapped content.
    blocks = [
        _text(0, "Normal page with plenty of extracted body text " * 5),
        _img(3, [40, 760, 580, 60]),  # area = 540 * 700 = 378_000, page 3 has no text
    ]
    out = check_no_ocr_failure_page(blocks)
    assert not out.passed
    assert out.metadata["offending_pages"] == [3]


def test_text_bearing_page_with_image_not_flagged():
    # A schedule page that has both a big image AND real text is not an OCR failure.
    blocks = [
        _text(6, "Week 1 Introduction; Week 2 Methods; Week 3 Results " * 3),
        _img(6, [40, 760, 580, 60]),
    ]
    out = check_no_ocr_failure_page(blocks)
    assert out.passed


def test_textless_page_with_only_small_logo_not_flagged():
    # A near-empty page whose only image is a tiny logo is not a content-loss page.
    blocks = [_img(2, [429, 705, 514, 641])]  # tiny
    out = check_no_ocr_failure_page(blocks)
    assert out.passed


def test_textless_page_with_unmeasurable_image_flagged():
    # CSCE_704 exact shape: a full-page scanned image arrives bbox-less (area
    # unmeasurable) on an otherwise textless page — the un-OCR'd-page signature.
    blocks = [
        _text(0, "Front matter with normal body text " * 5),
        {"type": "image", "page_idx": 3, "bbox": None, "image_caption": "", "image_footnote": ""},
    ]
    out = check_no_ocr_failure_page(blocks)
    assert not out.passed
    assert out.metadata["offending_pages"] == [3]
    assert out.metadata["offenders"][0]["unmeasurable_image"] is True
