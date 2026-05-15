"""Standalone Docling header→page extractor.

Opens each PDF with Docling, walks document items, emits a `<stem>.headers.json`
sidecar with [{text, level, page}, ...]. Used by metadata_enrichment.py as the
canonical header→page mapping (replaces fragile PyMuPDF text-matching).

Designed to run independently of full pipeline conversion — does not write
markdown, only the sidecar. Skips files that already have a sidecar unless
--force is passed.

Default scope: only files where image_recovery was skipped (Gemini didn't
modify the markdown, so Docling's items match the markdown content).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import SectionHeaderItem, TitleItem


def find_pdf(stem: str, raw_root: Path) -> Path | None:
    """Locate the source PDF (stripping _HP and version suffix)."""
    import re

    lookup = stem.removesuffix("_HP")
    lookup = re.sub(r"_v\d{3}$", "", lookup)
    for pdf in raw_root.rglob(f"{lookup}.pdf"):
        return pdf
    return None


def extract_headers(pdf_path: Path, converter: DocumentConverter) -> list[dict]:
    """Return a list of {text, level, page} dicts for every header in the PDF."""
    result = converter.convert(str(pdf_path))
    doc = result.document
    headers: list[dict] = []
    for item, _level in doc.iterate_items():
        if isinstance(item, (TitleItem, SectionHeaderItem)):
            text = (item.text or "").strip()
            if not text:
                continue
            level = getattr(item, "level", None)
            if isinstance(item, TitleItem):
                level = 1
            page = item.prov[0].page_no if item.prov else None
            headers.append({"text": text, "level": level, "page": page})
    return headers


def gemini_skipped_stems(dept: str) -> list[str]:
    """Return stems whose image_recovery_log status is 'skipped'."""
    log_path = Path(f"data/syllabi/{dept}/logs/filter_image_recovery_log.csv")
    if not log_path.exists():
        return []
    skipped: dict[str, str] = {}
    with log_path.open() as fh:
        for row in csv.DictReader(fh):
            stem = (row.get("file") or "").removesuffix(".md")
            status = row.get("status", "")
            if stem:
                skipped[stem] = status  # last-wins
    return sorted(s for s, st in skipped.items() if st == "skipped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="CSCE")
    ap.add_argument(
        "--sidecar-dir",
        default=None,
        help="Where to write sidecars. Default: data/syllabi/<DEPT>/bronze/",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing sidecars")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process all CSCE stems, not just Gemini-skipped ones",
    )
    ap.add_argument(
        "--stem",
        action="append",
        default=None,
        help="Process only these specific stems (repeatable)",
    )
    args = ap.parse_args()

    if args.stem:
        targets = sorted(args.stem)
    elif args.all:
        # All chunk files
        chunk_dir = Path(f"data/syllabi/{args.dept}/silver/06_chunk")
        targets = sorted(f.stem for f in chunk_dir.glob("*.json"))
    else:
        targets = gemini_skipped_stems(args.dept)

    if not targets:
        print("No targets found.", file=sys.stderr)
        return 1

    print(f"Extracting Docling sidecars for {len(targets)} files ({args.dept}):")
    for s in targets:
        print(f"  {s}")
    print()

    raw_root = Path("tamu_data/raw")
    sidecar_dir = Path(args.sidecar_dir or f"data/syllabi/{args.dept}/bronze")
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Docling converter...")
    opts = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        do_picture_classification=False,
        do_picture_description=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    summary: list[tuple[str, str, int]] = []
    t0 = time.monotonic()
    for i, stem in enumerate(targets, 1):
        sidecar_path = sidecar_dir / f"{stem}.headers.json"
        if sidecar_path.exists() and not args.force:
            print(f"[{i}/{len(targets)}] {stem}: sidecar exists, skipping (use --force to overwrite)")
            summary.append((stem, "skipped-existing", 0))
            continue

        pdf = find_pdf(stem, raw_root)
        if not pdf:
            print(f"[{i}/{len(targets)}] {stem}: PDF NOT FOUND")
            summary.append((stem, "no-pdf", 0))
            continue

        t1 = time.monotonic()
        try:
            headers = extract_headers(pdf, converter)
        except Exception as exc:
            print(f"[{i}/{len(targets)}] {stem}: ERROR {exc}")
            summary.append((stem, f"error: {exc}", 0))
            continue
        elapsed = time.monotonic() - t1

        with_page = sum(1 for h in headers if h["page"] is not None)
        sidecar_path.write_text(json.dumps(headers, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{i}/{len(targets)}] {stem}: {len(headers)} headers ({with_page} with page), "
            f"{elapsed:.1f}s → {sidecar_path.name}"
        )
        summary.append((stem, "ok", len(headers)))

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")
    ok = sum(1 for _, st, _ in summary if st == "ok")
    print(f"  Sidecars written: {ok}/{len(targets)}")
    failures = [(s, st) for s, st, _ in summary if st not in ("ok", "skipped-existing")]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for s, st in failures:
            print(f"    {s}: {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
