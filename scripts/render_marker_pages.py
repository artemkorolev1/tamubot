"""Render PDF pages containing <!-- image --> markers to PNG.

Used by the `recover-images` skill. For each bronze markdown file with image
markers, this script maps every marker to the PDF page it came from (via the
Docling headers sidecar), renders just those pages (+ the next page for
content spillover) to PNG, and emits a manifest mapping each marker to its
page image and surrounding context.

Usage:
    python scripts/render_marker_pages.py --stem 202611_STAT_624_600_34058
    python scripts/render_marker_pages.py --dept STAT --all
    python scripts/render_marker_pages.py --dept STAT --all --min-markers 2

Output layout (per stem):
    data/syllabi/<DEPT>/v5/.image_recovery_work/<stem>/
        page_03.png
        page_04.png
        manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

IMAGE_MARKER = "<!-- image -->"
MIN_MARKERS_DEFAULT = 2
RENDER_DPI = 150
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _dept_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot derive department from stem: {stem}")
    return parts[1]


def _v5_root(dept: str) -> Path:
    return Path("data/syllabi") / dept / "v5"


def _normalize_header(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(":").lower()


def _build_md_header_index(md_text: str, sidecar: list[dict]) -> list[tuple[int, str, int | None]]:
    """Return [(line_no, header_text, page)] for every header in the markdown.

    page is derived from the sidecar by normalized text match; falls back to
    the previous-resolved page when a markdown header isn't in the sidecar
    (common for ### subheaders that Docling didn't emit as SectionHeaderItems).
    """
    sidecar_by_text: dict[str, int] = {}
    for entry in sidecar:
        key = _normalize_header(entry["text"])
        if key and key not in sidecar_by_text:
            sidecar_by_text[key] = entry["page"]

    result: list[tuple[int, str, int | None]] = []
    last_page: int | None = None
    for i, line in enumerate(md_text.splitlines(), 1):
        m = HEADER_RE.match(line)
        if not m:
            continue
        text = m.group(2)
        page = sidecar_by_text.get(_normalize_header(text))
        if page is not None:
            last_page = page
        result.append((i, text, page if page is not None else last_page))
    return result


def _marker_lines(md_text: str) -> list[int]:
    return [i for i, line in enumerate(md_text.splitlines(), 1) if line.strip() == IMAGE_MARKER]


def _page_for_marker(line_no: int, headers: list[tuple[int, str, int | None]], total_pages: int) -> int:
    """Find the nearest preceding header's page; default to page 1 if none."""
    page = 1
    for hl, _, hp in headers:
        if hl < line_no and hp is not None:
            page = hp
        elif hl >= line_no:
            break
    return min(page, total_pages)


def _context_around(md_text: str, line_no: int, before: int = 5, after: int = 5) -> dict:
    lines = md_text.splitlines()
    lo = max(0, line_no - 1 - before)
    hi = min(len(lines), line_no + after)
    return {
        "before": lines[lo : line_no - 1],
        "after": lines[line_no:hi],
    }


def _render_page(pdf_doc: fitz.Document, page_num: int, out_path: Path) -> None:
    """Render a 1-indexed page to PNG."""
    if out_path.exists():
        return
    page = pdf_doc[page_num - 1]
    matrix = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(out_path)


def process_stem(stem: str, *, force: bool = False) -> dict:
    """Render pages and emit manifest for one stem. Returns the manifest."""
    dept = _dept_from_stem(stem)
    v5 = _v5_root(dept)
    bronze_md = v5 / "bronze" / f"{stem}.md"
    sidecar_path = v5 / "bronze" / f"{stem}.headers.json"
    raw_pdf = v5 / "raw" / f"{stem}.pdf"
    work_dir = v5 / ".image_recovery_work" / stem

    if not bronze_md.exists():
        raise FileNotFoundError(bronze_md)
    if not raw_pdf.exists():
        raise FileNotFoundError(raw_pdf)

    md_text = bronze_md.read_text(encoding="utf-8")
    markers = _marker_lines(md_text)

    sidecar_data: list[dict] = []
    if sidecar_path.exists():
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_data = loaded["headers"] if isinstance(loaded, dict) else loaded

    headers = _build_md_header_index(md_text, sidecar_data)

    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(raw_pdf)
    total_pages = pdf_doc.page_count

    pages_to_render: set[int] = set()
    marker_entries: list[dict] = []
    for idx, line_no in enumerate(markers):
        page = _page_for_marker(line_no, headers, total_pages)
        pages = [page]
        if page + 1 <= total_pages:
            pages.append(page + 1)
        for p in pages:
            pages_to_render.add(p)
        marker_entries.append(
            {
                "marker_index": idx,
                "line_no": line_no,
                "pages": pages,
                "context": _context_around(md_text, line_no),
            }
        )

    rendered: dict[int, str] = {}
    for p in sorted(pages_to_render):
        out_path = work_dir / f"page_{p:02d}.png"
        if force and out_path.exists():
            out_path.unlink()
        _render_page(pdf_doc, p, out_path)
        rendered[p] = str(out_path)

    pdf_doc.close()

    for entry in marker_entries:
        entry["page_pngs"] = [rendered[p] for p in entry["pages"]]

    manifest = {
        "stem": stem,
        "dept": dept,
        "bronze_md": str(bronze_md),
        "raw_pdf": str(raw_pdf),
        "silver_out": str(v5 / "silver" / "01_image_recovery" / f"{stem}.md"),
        "total_pages": total_pages,
        "markers": marker_entries,
        "rendered_pages": rendered,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def discover_stems(dept: str, min_markers: int) -> list[str]:
    bronze_dir = _v5_root(dept) / "bronze"
    stems: list[str] = []
    for md in sorted(bronze_dir.glob("*.md")):
        n = md.read_text(encoding="utf-8").count(IMAGE_MARKER)
        if n >= min_markers:
            stems.append(md.stem)
    return stems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", help="Single stem to process")
    ap.add_argument("--dept", help="Department code (required with --all)")
    ap.add_argument("--all", action="store_true", help="Process all bronze files for --dept with ≥ --min-markers")
    ap.add_argument("--min-markers", type=int, default=MIN_MARKERS_DEFAULT)
    ap.add_argument("--force", action="store_true", help="Re-render existing PNGs")
    args = ap.parse_args()

    if args.stem:
        m = process_stem(args.stem, force=args.force)
        print(f"{args.stem}: {len(m['markers'])} markers across {len(m['rendered_pages'])} pages")
        for entry in m["markers"]:
            print(f"  marker #{entry['marker_index']} (line {entry['line_no']}) → pages {entry['pages']}")
        return 0

    if args.all and args.dept:
        stems = discover_stems(args.dept, args.min_markers)
        print(f"Found {len(stems)} {args.dept} stems with ≥{args.min_markers} markers")
        for s in stems:
            try:
                m = process_stem(s, force=args.force)
                print(f"  {s}: {len(m['markers'])} markers, {len(m['rendered_pages'])} pages rendered")
            except Exception as e:
                print(f"  {s}: ERROR {e}", file=sys.stderr)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
