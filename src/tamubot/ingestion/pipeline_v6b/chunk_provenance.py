"""Bridge v6b semantic chunks back to the bronze blocks they were built from,
then to those blocks' page geometry — so a viewer can highlight, on the raw PDF,
exactly where each chunk's text came from.

Why a read-time bridge instead of threading IDs through the pipeline:
``chunker_v4.chunk_semantic`` reassembles chunk text from a parsed section tree
(``_node_text``) and exposes no char offsets, and its token budget is
``len(text)/4`` — so injecting ``<!--blk:ID-->`` sentinels into the merged
markdown would both break header detection (``^#`` must start the line) and
inflate token counts, shifting chunk boundaries. Instead we exploit the one
invariant that *does* hold: a chunk's ``content`` literally contains its
constituent blocks' text (headers re-rendered as ``## text``, bodies verbatim).
A monotonic forward sweep matches each block to the chunk that contains it.

Geometry: the bronze adapter now persists ``bbox`` per block (Docling
bottom-left coords). For files parsed before that change, ``resolve_annotations``
falls back to PyMuPDF ``search_for`` on the block text, so the viewer works on
already-processed corpora without re-running Docling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

# Streamlit-pdf-viewer annotation = {page, x, y, width, height, color} with
# (x, y) the TOP-LEFT corner, page 1-indexed — matching Docling's page_no.
Annotation = dict[str, Any]
Block = dict[str, Any]
Chunk = dict[str, Any]

_WS_RE = re.compile(r"\s+")
# Markdown the chunker adds around a block's own text — stripped before matching
# so "## Grading Policy" still matches the block heading "Grading Policy".
_MD_NOISE_RE = re.compile(r"^[#>\s|*_-]+")

_PREFIX_CHARS = 40  # fallback match window for a block split across two chunks


def normalize(text: str) -> str:
    """Whitespace-collapsed, lowercased form for substring matching. Treats the
    NBSP Docling emits (``\\xa0``) as a normal space so block and chunk text align."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip().lower()


def block_match_key(block: Block) -> str:
    """The normalized text used to locate a block inside a chunk's content.

    Heading/text blocks match on their text; tables on caption or first non-empty
    cell; images on caption. Returns "" when nothing usable is present (the block
    is then left unaligned rather than mis-attached)."""
    btype = block.get("type")
    if btype in ("heading", "text"):
        return normalize(_MD_NOISE_RE.sub("", block.get("text", "")))
    if btype == "table":
        cap = block.get("table_caption") or ""
        if cap.strip():
            return normalize(cap)
        for row in block.get("table_body") or []:
            for cell in row:
                if cell and cell.strip():
                    return normalize(cell)
        return ""
    if btype == "image":
        return normalize(block.get("image_caption", ""))
    return ""


def align_chunks_to_blocks(blocks: list[Block], chunks: list[Chunk]) -> list[list[Block]]:
    """Map each chunk to the ordered list of blocks that compose it.

    ``chunk_semantic`` traverses a section *tree*, so chunk order does NOT track
    raw block order (a "Catalog Description" sub-section can surface as chunk 0
    while page-1 metadata blocks precede it in the block list). A monotonic
    cursor therefore strands whole chunks; instead each block is matched against
    *all* chunks and attached to the first whose content contains its text.
    Blocks belong to exactly one chunk and we match on the block's full text, so
    cross-chunk collisions are rare; a generic short key (< 12 chars) skips the
    prefix fallback to avoid false hits. Blocks that match no chunk (deduped
    table cells, dropped fragments) are skipped. Returns a list parallel to
    ``chunks``; within each, blocks stay in document (block-list) order.
    """
    result: list[list[Block]] = [[] for _ in chunks]
    norm_chunks = [normalize(c.get("content", "")) for c in chunks]
    for block in blocks:
        key = block_match_key(block)
        if not key:
            continue
        found: Optional[int] = next((j for j, nc in enumerate(norm_chunks) if key in nc), None)
        if found is None:
            short = key[:_PREFIX_CHARS]
            if len(short) >= 12:  # too short a prefix yields false positives
                found = next((j for j, nc in enumerate(norm_chunks) if short in nc), None)
        if found is not None:
            result[found].append(block)
    return result


# ── Geometry → viewer annotations ────────────────────────────────────────────


def _docling_bbox_to_annotation(bbox: list, page: int, page_height: float, color: str) -> Optional[Annotation]:
    """Convert a Docling bottom-left ``(l, t, r, b)`` bbox (t > b, origin
    bottom-left) into a top-left viewer annotation. Mirrors the flip in
    ``docling_block_adapter._render_table_png``."""
    if not bbox or any(v is None for v in bbox):
        return None
    left, top, right, bottom = bbox
    width = right - left
    height = top - bottom
    if width <= 1 or height <= 1:
        return None
    return {
        "page": page,
        "x": round(left, 2),
        "y": round(page_height - top, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "color": color,
    }


def _search_annotations(pdf_page, text: str, page: int, color: str, max_rects: int = 6) -> list[Annotation]:
    """Recover boxes for a block with no persisted bbox via PyMuPDF text search.

    Searches the first non-trivial line (``search_for`` is unreliable on long
    multi-line strings) and returns up to ``max_rects`` top-left rects."""
    line = next((ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 6), "")
    if not line:
        return []
    try:
        rects = pdf_page.search_for(line[:120])
    except Exception:
        return []
    out: list[Annotation] = []
    for r in rects[:max_rects]:
        out.append(
            {
                "page": page,
                "x": round(r.x0, 2),
                "y": round(r.y0, 2),
                "width": round(r.x1 - r.x0, 2),
                "height": round(r.y1 - r.y0, 2),
                "color": color,
            }
        )
    return out


def resolve_annotations(pdf_path: str | Path, blocks: list[Block], color: str = "#e63946") -> list[Annotation]:
    """Build viewer highlight boxes for a set of blocks.

    Prefers each block's persisted Docling ``bbox``; falls back to a PyMuPDF
    text search for blocks parsed before bbox persistence. Tolerant — any block
    that can't be located is skipped rather than raising, so a partial highlight
    still renders."""
    import pymupdf

    annotations: list[Annotation] = []
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return annotations
    try:
        page_heights = {i + 1: doc[i].rect.height for i in range(len(doc))}
        for block in blocks:
            page = int(block.get("page_idx") or 0)
            if page < 1 or page > len(doc):
                continue
            bbox = block.get("bbox")
            if bbox:
                ann = _docling_bbox_to_annotation(bbox, page, page_heights[page], color)
                if ann:
                    annotations.append(ann)
                    continue
            # No usable bbox → search the page for the block's text.
            text = block.get("text") or block.get("table_caption") or block.get("image_caption") or ""
            if text:
                annotations.extend(_search_annotations(doc[page - 1], text, page, color))
    finally:
        doc.close()
    return annotations
