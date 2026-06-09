"""Docling → RAG-Anything block list adapter.

Wraps the existing :mod:`tamubot.ingestion.converters.docling_converter` and
re-emits its parsed document as a `List[Dict[str, Any]]` in the block format
defined by :class:`tamubot.vendor.raganything.parser.Parser`.

This is the seam that lets v6b feed Docling output into RAG-Anything-shaped
modal processors and chunkers without modifying v5 code.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from tamubot.ingestion.converters.docling_converter import convert, create_converter
from tamubot.vendor.raganything.parser import Parser, register_parser

log = logging.getLogger(__name__)


def _bbox_tuple(item) -> Optional[tuple]:
    """Extract (l, t, r, b) from the first provenance, or None."""
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0]
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return None
    return (
        getattr(bbox, "l", None),
        getattr(bbox, "t", None),
        getattr(bbox, "r", None),
        getattr(bbox, "b", None),
    )


def _page_idx(item) -> int:
    prov = getattr(item, "prov", None)
    if not prov:
        return 0
    pn = getattr(prov[0], "page_no", None)
    return int(pn) if pn is not None else 0


def _bbox_inside(inner: tuple, outer: tuple) -> bool:
    """Treat inner as 'inside' outer when its center sits within outer.

    bboxes here are docling-bottomleft (l, t, r, b) tuples where t > b.
    The check is forgiving — a cell that straddles a table-region edge by
    a few points still counts as inside.
    """
    il, it, ir, ib = inner
    ol, ot, or_, ob = outer
    if any(v is None for v in (il, it, ir, ib, ol, ot, or_, ob)):
        return False
    cx = (il + ir) / 2
    cy = (it + ib) / 2
    return ol - 2 <= cx <= or_ + 2 and ob - 2 <= cy <= ot + 2


def _inside_any_table(bbox: Optional[tuple], page: int, table_bboxes_by_page: Dict[int, List[tuple]]) -> bool:
    if bbox is None or any(v is None for v in bbox):
        return False
    for tb in table_bboxes_by_page.get(page, ()):
        if _bbox_inside(bbox, tb):
            return True
    return False


def _caption_text(item, doc) -> str:
    """Docling's caption_text is a method that needs the DoclingDocument.
    Returns "" on any failure so the parser never blocks on a stub caption."""
    fn = getattr(item, "caption_text", None)
    if fn is None:
        return ""
    if callable(fn):
        try:
            return (fn(doc) or "").strip()
        except Exception:
            return ""
    return (fn or "").strip()


def _recover_urls_by_page(pdf_path: Path) -> tuple[Dict[int, List[tuple]], Dict[int, float]]:
    """Recover PDF link annotations Docling drops during text extraction.

    Returns ``(urls_by_page, page_heights)`` where ``urls_by_page`` maps a
    1-indexed page to ``[(link_text, uri, link_cy_top), ...]`` — ``link_cy_top``
    is the annotation's vertical centre measured from the page top (PyMuPDF
    frame), used to place a recovered URL line where it actually sits on the
    page. ``page_heights`` maps page -> height (points), needed to convert
    Docling's bottom-origin block geometry into the same top-origin frame.

    Docling drops URL annotations during text extraction; v6's Gemini-VLM
    bronze keeps them because URLs are visually rendered. PyMuPDF's
    ``page.get_links()`` gives annotation-level access — no OCR or LLM needed.
    """
    try:
        import pymupdf
    except ImportError:
        log.warning("pymupdf not installed; URL recovery disabled")
        return ({}, {})

    out: Dict[int, List[tuple]] = {}
    heights: Dict[int, float] = {}
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        log.warning("URL recovery: pymupdf failed to open %s: %s", pdf_path, exc)
        return ({}, {})
    try:
        for pno, page in enumerate(doc, 1):  # type: ignore[var-annotated,arg-type]
            entries: List[tuple] = []
            for lk in page.get_links():
                uri = lk.get("uri")
                rect = lk.get("from")
                if not uri or rect is None:
                    continue
                try:
                    text = (page.get_textbox(rect) or "").strip()
                except Exception:
                    text = ""
                if not text:
                    continue
                # Collapse newlines/extra whitespace so "Click\nHere" matches
                # "Click Here" in the Docling markdown.
                text = " ".join(text.split())
                link_cy = (float(rect.y0) + float(rect.y1)) / 2  # centre, from top
                entries.append((text, uri, link_cy))
            if entries:
                out[pno] = entries
                heights[pno] = float(page.rect.height)
    finally:
        doc.close()
    return (out, heights)


def _orphan_url_insert_index(
    blocks: List[Dict[str, Any]],
    page: int,
    link_cy: Optional[float],
    tops_by_blockid: Dict[str, float],
    page_height: Optional[float],
) -> Optional[int]:
    """Index to insert a recovered URL line at so it lands where it actually sits
    on the page. Inserts right *after* the page block immediately above the link
    (by vertical position); if the link is above every block on the page, returns
    the index of the page's first block (insert before it). Falls back to the last
    block on the page when geometry is unavailable. ``None`` -> page absent,
    caller appends at the document end.

    Docling block geometry is bottom-origin (``t`` grows upward); the link centre
    is top-origin — convert via ``top_from_top = page_height - t``.
    """
    page_indices = [i for i, b in enumerate(blocks) if (b.get("page_idx") or 0) == page]
    if not page_indices:
        return None
    if link_cy is None or page_height is None:
        return page_indices[-1] + 1  # geometry unavailable -> end of page

    best_idx: Optional[int] = None
    best_top = float("-inf")
    for i in page_indices:
        t = tops_by_blockid.get(blocks[i].get("block_id", ""))
        if t is None:
            continue
        top_from_top = page_height - t
        if top_from_top <= link_cy and top_from_top > best_top:
            best_top = top_from_top
            best_idx = i
    if best_idx is not None:
        return best_idx + 1  # after the block just above the link
    return page_indices[0]  # link is above all page blocks -> before the first


def _append_orphan_urls(
    blocks: List[Dict[str, Any]],
    pending: Dict[int, List[tuple]],
    stem: str,
    tops_by_blockid: Optional[Dict[str, float]] = None,
    page_heights: Optional[Dict[int, float]] = None,
) -> int:
    """Tier 2 of URL recovery: insert URLs that matched no block as new text
    blocks. Docling sometimes drops a URL's whole line (e.g. ``Webpage: <url>`` /
    ``Biography: <url>``), leaving no host text to wrap — without this the URL,
    confirmed present in the source PDF, is lost from everything RAG sees
    (taxonomy ``FID_CONTENT_DROPPED``). The recovered line is placed by vertical
    position so it doesn't masquerade as a neighbouring field's value. Skips a URL
    whose uri or link-text is already present in some block (don't duplicate one
    Docling did keep)."""
    tops_by_blockid = tops_by_blockid or {}
    page_heights = page_heights or {}
    appended = 0
    for page, queue in pending.items():
        for entry in queue:
            link_text, uri = entry[0], entry[1]
            link_cy = entry[2] if len(entry) > 2 else None
            if not uri:
                continue
            if any(
                b.get("type") == "text" and (uri in b.get("text", "") or (link_text and link_text in b.get("text", "")))
                for b in blocks
            ):
                continue
            # Prefer the human-visible link text when it is itself a URL — a PDF
            # annotation's uri target is sometimes malformed in the source
            # (typo'd scheme, embedded prose), while the displayed text is clean.
            # Reserve the [label](uri) form for a non-URL label ("Course Webpage").
            if link_text and link_text.lower().startswith(("http://", "https://")):
                line = link_text
            elif link_text and link_text != uri:
                line = f"[{link_text}]({uri})"
            else:
                line = uri
            new_block = {
                "type": "text",
                "text": line,
                "page_idx": page,
                "block_id": f"{stem}_recovered_url_p{page}_{appended}",
                "recovered_url": True,
            }
            insert_at = _orphan_url_insert_index(blocks, page, link_cy, tops_by_blockid, page_heights.get(page))
            if insert_at is None:
                blocks.append(new_block)
            else:
                blocks.insert(insert_at, new_block)
            appended += 1
    return appended


def _inject_urls_into_blocks(
    blocks: List[Dict[str, Any]],
    urls_by_page: Dict[int, List[tuple]],
    tops_by_blockid: Optional[Dict[str, float]] = None,
    page_heights: Optional[Dict[int, float]] = None,
    stem: str = "",
) -> tuple[int, int]:
    """Recover PDF link-annotation URLs that Docling drops during text extraction.

    Two tiers (content must never be silently lost):
      1. **wrap** — when the link-text already appears in a text block on the same
         page, wrap that occurrence as ``[text](uri)`` in place.
      2. **append** — when Docling dropped the URL's whole line so no host block
         exists, insert the recovered URL as a NEW text block placed by vertical
         position (``_append_orphan_urls``) rather than discard it.

    Entries are ``(link_text, uri)`` or ``(link_text, uri, link_cy)``. Each URL is
    consumed on its first wrap so it isn't applied twice. Returns
    ``(wrapped, appended)`` counts.
    """
    if not urls_by_page:
        return (0, 0)

    # Per-page queue of (link_text, uri[, link_cy]) entries still to consume.
    pending = {pno: list(items) for pno, items in urls_by_page.items()}
    wrapped = 0

    for block in blocks:
        if block.get("type") != "text":
            continue
        page = block.get("page_idx") or 0
        queue = pending.get(page)
        if not queue:
            continue
        text = block.get("text", "")
        if not text:
            continue

        # Try each pending URL on this page; consume on first match.
        remaining: List[tuple] = []
        for entry in queue:
            link_text, uri = entry[0], entry[1]
            if not link_text:
                remaining.append(entry)
                continue
            # Avoid double-wrapping if the URL is already a markdown link.
            already = f"]({uri})" in text or f"[{link_text}]" in text
            if not already and link_text in text:
                text = text.replace(link_text, f"[{link_text}]({uri})", 1)
                wrapped += 1
            else:
                remaining.append(entry)
        pending[page] = remaining
        block["text"] = text

    appended = _append_orphan_urls(blocks, pending, stem, tops_by_blockid, page_heights)
    return (wrapped, appended)


def _all_block_tokens(blocks: List[Dict[str, Any]]) -> set:
    """Content tokens across every bronze block — text bodies, table-grid cells,
    captions. Used to tell a genuinely-dropped value from one Docling kept
    elsewhere (e.g. in a table)."""
    from tamubot.ingestion.validation.text_coverage import content_tokens

    parts: List[str] = []
    for b in blocks:
        t = b.get("text")
        if isinstance(t, str) and t:
            parts.append(t)
        for key in ("table_caption", "image_caption"):
            v = b.get(key)
            if isinstance(v, str) and v:
                parts.append(v)
        grid = b.get("table_body")
        if isinstance(grid, list):
            for row in grid:
                if isinstance(row, list):
                    parts.append(" ".join(str(c) for c in row if c))
    return content_tokens("\n".join(parts))


def _extend_labels_with_values(
    blocks: List[Dict[str, Any]],
    lines_by_page: Dict[int, List[str]],
    bronze_tokens: set,
) -> int:
    """Recover ``Label: value`` lines where Docling kept the label block but
    dropped the trailing value (e.g. ``ISBN:`` survives, ``978-0-387-69957-8``
    lost — the FID_CONTENT_DROPPED right-cell class). Extends a text block that is
    a bare ``…:`` label with the full PyMuPDF line, but only when the tail carries
    a content token missing from bronze (so a value Docling already kept is never
    duplicated). Pure — host-testable. Returns the number of blocks extended."""
    from tamubot.ingestion.validation.text_coverage import content_tokens

    recovered = 0
    for b in blocks:
        if b.get("type") != "text":
            continue
        bt = b.get("text")
        if not isinstance(bt, str):
            continue
        label = bt.rstrip()
        if not label.endswith(":") or len(label) < 3:
            continue
        page = b.get("page_idx") or 0
        for line in lines_by_page.get(page, ()):
            if line == label or not line.startswith(label) or len(line) <= len(label) + 1:
                continue
            tail = content_tokens(line[len(label) :])
            if any(tok not in bronze_tokens and len(tok) >= 4 for tok in tail):
                b["text"] = line
                b["recovered_value"] = True
                bronze_tokens |= tail
                recovered += 1
                break
    return recovered


def _recover_dropped_label_values(pdf_path: Path, blocks: List[Dict[str, Any]]) -> int:
    """PyMuPDF wrapper for ``_extend_labels_with_values``: read each page's visual
    lines and re-attach values Docling dropped from ``Label:`` blocks."""
    try:
        import pymupdf
    except ImportError:
        return 0
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        log.warning("label-value recovery: pymupdf failed to open %s: %s", pdf_path, exc)
        return 0
    try:
        lines_by_page: Dict[int, List[str]] = {}
        for pno, page in enumerate(doc, 1):  # type: ignore[var-annotated,arg-type]
            lines: List[str] = []
            for blk in page.get_text("dict").get("blocks", []):
                for ln in blk.get("lines", []):
                    t = " ".join(sp.get("text", "") for sp in ln.get("spans", []))
                    t = " ".join(t.split())
                    if t:
                        lines.append(t)
            if lines:
                lines_by_page[pno] = lines
    finally:
        doc.close()
    return _extend_labels_with_values(blocks, lines_by_page, _all_block_tokens(blocks))


def _render_table_png(
    pdf_path: Path,
    page_idx: int,
    bbox: Optional[tuple],
    out_path: Path,
    pad: float = 8.0,
) -> bool:
    """Render the PDF region containing a table as a PNG via PyMuPDF.

    Docling reports bbox in BOTTOMLEFT coords (y grows up, origin at page
    bottom-left). PyMuPDF's Rect uses TOPLEFT coords (y grows down). We
    convert: rect_top = page.height - bbox.t, rect_bot = page.height - bbox.b.

    Returns True on success, False on any failure (caller falls back to no
    img_path).
    """
    if bbox is None or any(v is None for v in bbox):
        return False
    try:
        import pymupdf

        x0, t, r, b = bbox
        doc = pymupdf.open(str(pdf_path))
        try:
            # PyMuPDF pages are 0-indexed; Docling's page_no is 1-indexed
            pno = max(0, int(page_idx) - 1)
            if pno >= len(doc):
                return False
            page = doc[pno]
            height = page.rect.height
            rect = pymupdf.Rect(
                max(0.0, x0 - pad),
                max(0.0, height - t - pad),
                min(page.rect.width, r + pad),
                min(height, height - b + pad),
            )
            if rect.is_empty or rect.width < 2 or rect.height < 2:
                return False
            pix = page.get_pixmap(clip=rect, dpi=150)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(out_path))
            return True
        finally:
            doc.close()
    except Exception as exc:
        log.warning("table-region render failed for %s page=%s bbox=%s: %s", pdf_path, page_idx, bbox, exc)
        return False


def _render_picture_png(
    pdf_path: Path,
    page: int,
    bbox: Optional[tuple],
    out_path: Path,
) -> bool:
    """Render a PictureItem bbox to PNG. Returns True on success.

    Delegates to _render_table_png — bbox→PNG rendering is identical
    regardless of the source item type.
    """
    if bbox is None or any(v is None for v in bbox):
        return False
    return _render_table_png(pdf_path, page, bbox, out_path)


def _extract_table_body(item) -> List[List[str]]:
    """Pull a 2-D list of cell strings out of a Docling TableItem.

    Docling exposes parsed cells via item.data.grid (list of rows of
    cell objects with .text). Older builds use .data.table_cells with
    explicit row/col indices; handle both shapes.
    """
    data = getattr(item, "data", None)
    if data is None:
        return []
    grid = getattr(data, "grid", None)
    if grid:
        return [[(getattr(c, "text", "") or "").strip() for c in row] for row in grid]
    cells = getattr(data, "table_cells", None) or []
    if not cells:
        return []
    rows: Dict[int, Dict[int, str]] = {}
    for c in cells:
        r = getattr(c, "start_row_offset_idx", None)
        col = getattr(c, "start_col_offset_idx", None)
        if r is None or col is None:
            continue
        rows.setdefault(r, {})[col] = (getattr(c, "text", "") or "").strip()
    if not rows:
        return []
    max_col = max(max(row.keys()) for row in rows.values())
    return [[rows[r].get(c, "") for c in range(max_col + 1)] for r in sorted(rows.keys())]


def _norm_header(text: str) -> str:
    """Normalize a header for matching: lowercase, collapse whitespace, drop
    trailing punctuation (Docling sometimes keeps a trailing ':' on a header)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(":;.- ").strip()


def _recover_heading_levels(blocks: List[Dict[str, Any]], headers_path: Path) -> None:
    """Assign real heading levels to block headings, in place.

    The per-item parse tags every heading level 1. headers.json carries the
    hierarchy-postprocessor's corrected levels. Both are in document order, so a
    forward two-pointer alignment matches each heading block to its reference
    entry. Headings absent from the reference (instructor pseudo-headers the
    safety net demoted out of the hierarchy) are nested one level below the last
    matched header, turning their parent into a real subtree for the chunker.
    """
    if not headers_path.exists():
        return
    try:
        ref = json.loads(headers_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return
    ref_norm = [(_norm_header(h.get("text", "")), int(h.get("level", 1) or 1)) for h in ref]

    j = 0
    last_level = 1
    for b in blocks:
        if b.get("type") != "heading":
            continue
        bt = _norm_header(b.get("text", ""))
        matched: Optional[int] = None
        for k in range(j, len(ref_norm)):
            if ref_norm[k][0] == bt:
                matched = ref_norm[k][1]
                j = k + 1
                break
        if matched is not None:
            b["level"] = matched
            last_level = matched
        else:
            b["level"] = min(6, last_level + 1)


def _normalize_heading_levels(blocks: List[Dict[str, Any]]) -> int:
    """Re-level heading blocks so the hierarchy is skip-free, in place.

    Records each heading's pre-normalization level as ``raw_level`` (so the
    bronze ``header_levels_normalized`` check can report how many skips were
    repaired) then overwrites ``level`` with the stack-based tree depth. A
    single H1->H3 skip used to abort the whole document at the bronze gate;
    this guarantees the no-skip invariant deterministically without ever
    dropping content. Returns the number of forward skips repaired.
    """
    from tamubot.ingestion.validation.header_hierarchy import (
        count_level_skips,
        normalize_heading_levels,
    )

    headings = [b for b in blocks if b.get("type") == "heading"]
    if not headings:
        return 0
    raw_levels = [int(b.get("level", 1) or 1) for b in headings]
    repaired = count_level_skips(raw_levels)
    normalized = normalize_heading_levels(raw_levels)
    for b, raw, lvl in zip(headings, raw_levels, normalized):
        b["raw_level"] = raw
        b["level"] = lvl
    return repaired


def docling_to_blocks(
    pdf_path: Union[str, Path],
    output_dir: Path,
    converter: Optional[DocumentConverter] = None,
    apply_hierarchy: bool = False,
) -> List[Dict[str, Any]]:
    """Convert a PDF and emit RAG-Anything-shape blocks.

    Side effect: writes markdown + headers.json via the underlying convert()
    so callers get the legacy artifacts too without a second Docling run.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if converter is None:
        converter = create_converter()

    result_obj = converter.convert(str(pdf_path))
    stem = pdf_path.stem
    blocks: List[Dict[str, Any]] = []
    idx = 0
    table_render_idx = 0
    tables_dir = output_dir / "tables" / stem
    picture_render_idx = 0
    pictures_dir = output_dir / "pictures" / stem

    # First pass: collect table bboxes per page. Docling with
    # do_table_structure=False emits each table cell as a separate TextItem
    # in addition to the TableItem itself. We use the table bboxes in the
    # second pass to drop those duplicate cell TextItems.
    items = list(result_obj.document.iterate_items())
    table_bboxes_by_page: Dict[int, List[tuple]] = {}
    for item, _level in items:
        if isinstance(item, TableItem):
            tb = _bbox_tuple(item)
            if tb is None or any(v is None for v in tb):
                continue
            table_bboxes_by_page.setdefault(_page_idx(item), []).append(tb)

    _last_text_key: tuple[int, str] | None = None  # (page_idx, normalized_text)
    # block_id -> Docling bbox top (bottom-origin); used to place recovered URLs.
    tops_by_blockid: Dict[str, float] = {}

    for item, _level in items:
        page = _page_idx(item)
        bbox = _bbox_tuple(item)
        block_id = Parser.block_id(stem, page, bbox, idx)

        if isinstance(item, TitleItem):
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            _last_text_key = None
            blocks.append(
                {
                    "type": "heading",
                    "text": text,
                    "level": 1,
                    "page_idx": page,
                    "block_id": block_id,
                }
            )
        elif isinstance(item, SectionHeaderItem):
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            _last_text_key = None
            blocks.append(
                {
                    "type": "heading",
                    "text": text,
                    "level": int(getattr(item, "level", 1) or 1),
                    "page_idx": page,
                    "block_id": block_id,
                }
            )
        elif isinstance(item, PictureItem):
            png_path = pictures_dir / f"picture_{picture_render_idx:03d}.png"
            rendered = _render_picture_png(pdf_path, page, bbox, png_path)
            picture_render_idx += 1
            _last_text_key = None
            blocks.append(
                {
                    "type": "image",
                    "img_path": str(png_path.resolve()) if rendered else "",
                    "image_caption": _caption_text(item, result_obj.document),
                    "image_footnote": "",
                    "page_idx": page,
                    "block_id": block_id,
                }
            )
        elif isinstance(item, TableItem):
            png_path = tables_dir / f"table_{table_render_idx:03d}.png"
            rendered = _render_table_png(pdf_path, page, bbox, png_path)
            table_render_idx += 1
            _last_text_key = None
            blocks.append(
                {
                    "type": "table",
                    "img_path": str(png_path.resolve()) if rendered else "",
                    "table_caption": _caption_text(item, result_obj.document),
                    "table_footnote": "",
                    "table_body": _extract_table_body(item),
                    "page_idx": page,
                    "block_id": block_id,
                }
            )
        elif isinstance(item, TextItem):
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            # Skip text items that are actually shredded table cells. Docling
            # emits these alongside the TableItem when do_table_structure=False;
            # the GFM markdown returned by silver_modal already covers them.
            if _inside_any_table(bbox, page, table_bboxes_by_page):
                continue
            norm = " ".join(text.split())  # collapse whitespace for dedup key
            key = (page, norm)
            if key == _last_text_key:
                continue
            _last_text_key = key
            blocks.append(
                {
                    "type": "text",
                    "text": text,
                    "page_idx": page,
                    "block_id": block_id,
                }
            )
        else:
            continue

        # A block was appended this iteration; record its top for URL placement.
        if bbox and bbox[1] is not None:
            tops_by_blockid[block_id] = bbox[1]
        idx += 1

    # URL recovery: Docling drops PDF link annotations; PyMuPDF still has them.
    # Wrap matching link-text in place, else insert the dropped URL line by its
    # vertical position so it propagates to chunks without masquerading as a
    # neighbouring field's value.
    urls_by_page, page_heights = _recover_urls_by_page(pdf_path)
    wrapped, appended = _inject_urls_into_blocks(blocks, urls_by_page, tops_by_blockid, page_heights, stem=stem)
    if wrapped or appended:
        log.info(
            "v6b URL recovery for %s: wrapped %d link(s) in place, appended %d dropped URL line(s)",
            stem,
            wrapped,
            appended,
        )

    # Re-attach `Label: value` values Docling dropped (e.g. ISBN numbers) — the
    # right-cell FID_CONTENT_DROPPED class. Runs after URL recovery so a recovered
    # URL already counts as present and isn't re-added.
    values = _recover_dropped_label_values(pdf_path, blocks)
    if values:
        log.info("v6b label-value recovery for %s: re-attached %d dropped value(s)", stem, values)

    convert(pdf_path, output_dir, converter=converter, apply_hierarchy=apply_hierarchy)

    # Docling's per-item parse (above) tags every heading H1; the convert() call
    # just wrote a headers.json whose levels come from the hierarchy postprocessor
    # (TOC + numbering + font clustering) and safety net — the corrected view.
    # Recover those levels into the blocks so the merged markdown (and thus the
    # chunker's section tree and phase2 header_path) reflect real hierarchy.
    if apply_hierarchy:
        _recover_heading_levels(blocks, output_dir / f"{stem}.headers.json")
    # Always re-level to a skip-free hierarchy (records raw_level for the
    # observability check). Deterministic, content-preserving — replaces the
    # old blocking-ERROR gate that dropped whole documents on a single skip.
    repaired = _normalize_heading_levels(blocks)
    if repaired:
        log.info("v6b heading re-leveling: repaired %d level-skip(s) in %s", repaired, stem)
    return blocks


class DoclingBlockParser(Parser):
    """RAG-Anything Parser interface over the existing Docling converter."""

    def __init__(self, converter: Optional[DocumentConverter] = None) -> None:
        self._converter = converter

    def check_installation(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter as _DC  # noqa: F401

            return True
        except ImportError:
            return False

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if output_dir is None:
            raise ValueError("DoclingBlockParser requires output_dir (markdown + headers sidecar are persisted)")
        return docling_to_blocks(
            pdf_path,
            Path(output_dir),
            converter=self._converter,
            apply_hierarchy=bool(kwargs.get("apply_hierarchy", False)),
        )

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf(file_path, output_dir=output_dir, method=method, lang=lang, **kwargs)
        raise NotImplementedError(f"DoclingBlockParser only supports .pdf, got {suffix!r}")


register_parser("docling", DoclingBlockParser)
