"""Image/layout fidelity validators (FID_IMAGE_LOST / FID_TABLE_LOST / FID_CONTENT_DROPPED).

Pure logic over bronze block dicts (each block has ``type``, ``page_idx``, ``bbox``,
and — for images — ``image_caption`` / ``image_footnote``). No PyMuPDF/Docling import,
host-testable. Guards the default modal-disabled path, where a content-bearing figure or
an un-OCR'd page is silently reduced to a bare ``<!-- image -->`` marker.

Two complementary signals:
  * ``check_content_bearing_images`` — a large image with NO caption/footnote anywhere
    (the ISEN_665 schedule-table-as-image class: modal disabled => content never
    transcribed).
  * ``check_no_ocr_failure_page`` — a page carrying a large image but almost no text
    (the CSCE_704 class: a whole grading region trapped in an un-OCR'd page image).
"""

from __future__ import annotations

from typing import Any

from tamubot.ingestion.validation.types import CheckOutcome

# The neutral, non-citable placeholder ``silver_modal._merge_to_markdown`` emits for an
# image block that left NO recoverable text behind (no modal description, no caption).
# A bare marker that reaches the chunk view = a figure RAG retrieves with zero content.
BARE_IMAGE_MARKER = "<!-- image -->"


def count_bare_image_markers(text: str) -> int:
    """Number of bare ``<!-- image -->`` placeholders in ``text`` (e.g. the concatenated
    chunk view). Markers carrying a description (``![cap](#) <!-- … -->``) or a caption
    (``![cap](#)``) are *not* bare and are not counted."""
    return (text or "").count(BARE_IMAGE_MARKER)


def check_no_bare_image_markers(
    chunk_view: str, *, bronze_image_count: int | None = None
) -> CheckOutcome:
    """End-to-end FID_IMAGE_LOST gate at the chunk view: a bare ``<!-- image -->`` that
    survived into the text RAG retrieves is a figure delivered with zero recoverable
    content (no caption, description, or OCR).

    Complements the *predictive* bronze ``check_content_bearing_images`` (which flags a
    large uncaptioned image by bbox area, before chunking): this is the *confirmatory*
    end-of-line signal — it counts the markers that actually reached a chunk, after the
    merge + chunk steps, regardless of bbox. When ``bronze_image_count`` is supplied it
    also reports the share of image blocks that degraded to a bare marker. Shipped WARN —
    a non-zero count is expected while modal is disabled; it names how much figure content
    a GPU modal/VLM pass would recover. ``passed`` iff zero bare markers."""
    markers = count_bare_image_markers(chunk_view)
    meta: dict[str, Any] = {"bare_image_markers": markers}
    if bronze_image_count is not None:
        meta["bronze_image_count"] = bronze_image_count
        meta["bare_marker_rate"] = round(markers / bronze_image_count, 4) if bronze_image_count else 0.0
    return CheckOutcome(passed=markers == 0, metadata=meta)

# Area (pt^2) above which an uncaptioned image is treated as content-bearing rather than
# a logo/decoration. US-Letter is 612x792 = ~485k pt^2; 50k ~ 10% of the page. A course
# logo/seal is ~5k pt^2; a half-page schedule table image is ~200k+. Tunable per call.
DEFAULT_MIN_CONTENT_AREA = 50_000.0
# Below this many text+heading chars a page is "textless" — an image-only / OCR-failure
# page when it also carries a large image.
DEFAULT_MIN_PAGE_TEXT_CHARS = 50


def _bbox_area(bbox: Any) -> float:
    """Area of a ``[x0, y0, x1, y1]`` bbox, robust to coordinate order (PDF y can run
    bottom-up). Returns 0.0 for a missing/short bbox."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    return abs(x1 - x0) * abs(y1 - y0)


def _has_caption(block: dict[str, Any]) -> bool:
    cap = (block.get("image_caption") or "").strip()
    foot = (block.get("image_footnote") or "").strip()
    return bool(cap or foot)


def check_content_bearing_images(
    blocks: list[dict[str, Any]], *, min_area: float = DEFAULT_MIN_CONTENT_AREA
) -> CheckOutcome:
    """FLAG large, uncaptioned image blocks — a content figure/table that the default
    modal-disabled path will emit as a bare ``<!-- image -->`` (ISEN_665 class).

    An image is content-bearing when its bbox area ``>= min_area`` AND it has no
    caption/footnote (a captioned image at least leaves text behind). Reports the count,
    the offending pages, and the largest offender's area. Shipped WARN — a non-zero
    count is expected while modal is disabled; it names exactly which stems/pages need a
    GPU modal/VLM pass to recover the trapped content.
    """
    offenders: list[dict[str, Any]] = []
    for b in blocks:
        if b.get("type") != "image":
            continue
        area = _bbox_area(b.get("bbox"))
        if area >= min_area and not _has_caption(b):
            offenders.append({"page_idx": b.get("page_idx") or 0, "area": round(area, 1)})
    offenders.sort(key=lambda o: o["area"], reverse=True)
    return CheckOutcome(
        passed=len(offenders) == 0,
        metadata={
            "content_image_count": len(offenders),
            "offending_pages": sorted({o["page_idx"] for o in offenders}),
            "largest_area": offenders[0]["area"] if offenders else 0.0,
            "min_area": min_area,
            "offenders": offenders[:20],
        },
    )


def check_no_ocr_failure_page(
    blocks: list[dict[str, Any]],
    *,
    min_text_chars: int = DEFAULT_MIN_PAGE_TEXT_CHARS,
    min_image_area: float = DEFAULT_MIN_CONTENT_AREA,
) -> CheckOutcome:
    """FLAG a page that carries a large image but almost no extracted text — an
    image-only / OCR-failure page whose content is trapped in pixels (CSCE_704 class).

    Per page: sum text+heading chars and the largest image area. A page is an OCR-failure
    page when its text is below ``min_text_chars`` AND it holds an image that is either at
    least ``min_image_area`` in size OR has no measurable bbox (a full-page scanned image
    often arrives bbox-less — the CSCE_704 case — and "textless page + unmeasurable image"
    is the un-OCR'd-page signature). A textless page whose only image is a small,
    *measurable* logo is NOT flagged. Reports the offending page indices. WARN — these
    pages need a GPU modal/VLM re-pass; no host-side text fix can reach them.
    """
    pages: dict[int, dict[str, float]] = {}
    for b in blocks:
        pg = int(b.get("page_idx") or 0)
        rec = pages.setdefault(pg, {"text_chars": 0.0, "max_image_area": 0.0, "unmeasurable_image": 0.0})
        btype = b.get("type")
        if btype in ("text", "heading"):
            rec["text_chars"] += len((b.get("text") or ""))
        elif btype == "image":
            area = _bbox_area(b.get("bbox"))
            rec["max_image_area"] = max(rec["max_image_area"], area)
            if area == 0.0:
                rec["unmeasurable_image"] = 1.0
    offenders = [
        {
            "page_idx": pg,
            "text_chars": int(r["text_chars"]),
            "image_area": round(r["max_image_area"], 1),
            "unmeasurable_image": bool(r["unmeasurable_image"]),
        }
        for pg, r in sorted(pages.items())
        if r["text_chars"] < min_text_chars
        and (r["max_image_area"] >= min_image_area or r["unmeasurable_image"])
    ]
    return CheckOutcome(
        passed=len(offenders) == 0,
        metadata={
            "ocr_failure_page_count": len(offenders),
            "offending_pages": [o["page_idx"] for o in offenders],
            "min_text_chars": min_text_chars,
            "min_image_area": min_image_area,
            "offenders": offenders[:20],
        },
    )
