"""Docling → RAG-Anything block list adapter.

Wraps the existing :mod:`tamubot.ingestion.converters.docling_converter` and
re-emits its parsed document as a `List[Dict[str, Any]]` in the block format
defined by :class:`tamubot.vendor.raganything.parser.Parser`.

This is the seam that lets v6b feed Docling output into RAG-Anything-shaped
modal processors and chunkers without modifying v5 code.
"""

from __future__ import annotations

import logging
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


def _recover_urls_by_page(pdf_path: Path) -> Dict[int, List[tuple]]:
    """Return {1-indexed_page: [(link_text, uri), ...]} from PDF link annotations.

    Docling drops URL annotations during text extraction; v6's Gemini-VLM
    bronze keeps them because URLs are visually rendered. We use PyMuPDF's
    page.get_links() to recover them post-hoc — it's annotation-level access,
    no OCR or LLM needed.
    """
    try:
        import pymupdf
    except ImportError:
        log.warning("pymupdf not installed; URL recovery disabled")
        return {}

    out: Dict[int, List[tuple]] = {}
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        log.warning("URL recovery: pymupdf failed to open %s: %s", pdf_path, exc)
        return {}
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
                entries.append((text, uri))
            if entries:
                out[pno] = entries
    finally:
        doc.close()
    return out


def _inject_urls_into_blocks(
    blocks: List[Dict[str, Any]],
    urls_by_page: Dict[int, List[tuple]],
) -> int:
    """Mutate text blocks to wrap URL link-text in markdown links.

    Walks blocks in order; for each text block, finds URLs whose page matches
    and whose link_text appears in the block content; replaces the first
    unwrapped occurrence with [text](uri). Marks each URL consumed so the
    same URL isn't applied twice to the same link_text.

    Returns the number of URLs successfully injected.
    """
    if not urls_by_page:
        return 0

    # Per-page queue of (link_text, uri) entries still to consume.
    pending = {pno: list(items) for pno, items in urls_by_page.items()}
    injected = 0

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
        for link_text, uri in queue:
            if not link_text:
                continue
            # Avoid double-wrapping if the URL is already a markdown link.
            already = f"]({uri})" in text or f"[{link_text}]" in text
            if not already and link_text in text:
                text = text.replace(link_text, f"[{link_text}]({uri})", 1)
                injected += 1
            else:
                remaining.append((link_text, uri))
        pending[page] = remaining
        block["text"] = text

    return injected


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

        idx += 1

    # URL recovery: Docling drops PDF link annotations; PyMuPDF still has
    # them. Wrap matching link-text in text blocks with markdown links so
    # they propagate to chunks.
    urls_by_page = _recover_urls_by_page(pdf_path)
    injected = _inject_urls_into_blocks(blocks, urls_by_page)
    if injected:
        log.info("v6b URL recovery: injected %d markdown links into %s blocks", injected, stem)

    convert(pdf_path, output_dir, converter=converter, apply_hierarchy=apply_hierarchy)
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
