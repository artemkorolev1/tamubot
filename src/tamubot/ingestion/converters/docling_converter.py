"""Docling-based PDF-to-markdown converter with ML layout detection."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTextSerializer,
)
from docling_core.types.doc.document import SectionHeaderItem, TitleItem

from tamubot.ingestion._vendor.hierarchical.postprocessor import ResultPostprocessor

log = logging.getLogger(__name__)

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_INDENTED_HEADER_RE = re.compile(r"^(\s+)(#{1,6}\s+.+)$")
_ANCHORED_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _apply_hierarchy_safety_net(markdown: str, original_header_texts_lower: set[str]) -> str:
    """Reconcile postprocessor output with Docling's pre-postprocessor header set.

    1. Dedent indented header lines (postprocessor sometimes emits
       ``    ## Heading`` which breaks ``^#`` matching).
    2. Demote H1/H2 lines whose text was never a SectionHeaderItem in the
       original Docling output — these are postprocessor false positives.
    """
    out: list[str] = []
    for line in markdown.splitlines():
        m = _INDENTED_HEADER_RE.match(line)
        if m:
            line = m.group(2)
        hm = _ANCHORED_HEADER_RE.match(line)
        if hm:
            level = len(hm.group(1))
            text = hm.group(2).strip()
            if level <= 2 and text.lower() not in original_header_texts_lower:
                line = text
        out.append(line)
    return "\n".join(out)


@dataclass
class ConvertResult:
    markdown: str
    output_path: Path
    timing_s: float
    header_count: int
    hierarchy_depth: dict[int, int] = field(default_factory=dict)


class _CustomTextSerializer(MarkdownTextSerializer):
    """Map SectionHeaderItem.level directly to markdown # count (not level+1)."""

    def _format_heading(
        self,
        text: str,
        item: Union[TitleItem, SectionHeaderItem],
    ) -> str:
        if isinstance(item, TitleItem):
            num_hashes = 1
        else:
            num_hashes = max(1, item.level)
        return f"{'#' * num_hashes} {text}"


def create_converter() -> DocumentConverter:
    """Create a Docling DocumentConverter (expensive — call once, reuse)."""
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,  # was False — see plan v6b-table-content
        do_picture_classification=False,
        do_picture_description=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_chart_extraction=False,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )


def _count_headers(markdown: str) -> tuple[int, dict[int, int]]:
    """Return (total_header_count, {level: count})."""
    depth: dict[int, int] = {}
    count = 0
    for m in HEADER_RE.finditer(markdown):
        level = len(m.group(1))
        depth[level] = depth.get(level, 0) + 1
        count += 1
    return count, depth


def convert(
    pdf_path: Path,
    output_dir: Path,
    converter: DocumentConverter | None = None,
    apply_hierarchy: bool = False,
) -> ConvertResult:
    """Convert a PDF to markdown using Docling.

    When ``apply_hierarchy`` is True, runs ``ResultPostprocessor`` to infer
    heading levels (TOC + numbering + font clustering) and applies a safety
    net that dedents indented headers and demotes postprocessor false
    positives back to body text.
    """
    if converter is None:
        converter = create_converter()

    t0 = time.monotonic()

    result = converter.convert(str(pdf_path))

    # Snapshot Docling's pre-postprocessor header set + text→page map. The
    # safety net needs the original set to detect false promotions; the
    # text→page map lets the final sidecar look up pages for headers that
    # the postprocessor may have moved around.
    orig_headers_lower: set[str] = set()
    text_to_page: dict[str, int | None] = {}
    for item, _level in result.document.iterate_items():
        text_attr = getattr(item, "text", None)
        if text_attr:
            t = text_attr.strip()
            if t:
                text_to_page.setdefault(
                    t.lower(),
                    item.prov[0].page_no if item.prov else None,
                )
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            t = (text_attr or "").strip()
            if t:
                orig_headers_lower.add(t.lower())

    if apply_hierarchy:
        # Infer heading hierarchy (Docling outputs flat H1 by default). Modifies
        # result.document in place using TOC, numbering, and font-size clustering.
        ResultPostprocessor(result, source=str(pdf_path)).process()

    # Serialize with custom heading formatter. Use model_construct to skip
    # Pydantic validation, which can fail on some hierarchy states.
    try:
        serializer = MarkdownDocSerializer(
            doc=result.document,
            text_serializer=_CustomTextSerializer(),
        )
    except Exception:
        log.debug("Pydantic validation failed, using model_construct to skip it")
        serializer = MarkdownDocSerializer.model_construct(
            doc=result.document,
            text_serializer=_CustomTextSerializer(),
            table_serializer=MarkdownDocSerializer.model_fields["table_serializer"].default,
            picture_serializer=MarkdownDocSerializer.model_fields["picture_serializer"].default,
            key_value_serializer=MarkdownDocSerializer.model_fields["key_value_serializer"].default,
            form_serializer=MarkdownDocSerializer.model_fields["form_serializer"].default,
            fallback_serializer=MarkdownDocSerializer.model_fields["fallback_serializer"].default,
            list_serializer=MarkdownDocSerializer.model_fields["list_serializer"].default,
            inline_serializer=MarkdownDocSerializer.model_fields["inline_serializer"].default,
            meta_serializer=MarkdownDocSerializer.model_fields["meta_serializer"].default,
            annotation_serializer=MarkdownDocSerializer.model_fields["annotation_serializer"].default,
            params=MarkdownParams(),
        )
    markdown = serializer.serialize().text

    # Fix HTML encoding artifacts
    markdown = markdown.replace("&amp;", "&")

    if apply_hierarchy:
        markdown = _apply_hierarchy_safety_net(markdown, orig_headers_lower)

    timing_s = time.monotonic() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text(markdown, encoding="utf-8")

    # Sidecar: page mapping for each header in the final markdown. Consumed by
    # filters.metadata_enrichment to anchor chunks to PDF pages reliably
    # (avoids fragile PyMuPDF text-matching that breaks on line-wrapped headers).
    headers: list[dict] = []
    if apply_hierarchy:
        # Final markdown may differ from result.document after the safety net,
        # so derive headers from markdown and look up page from the snapshot.
        for line in markdown.splitlines():
            m = _ANCHORED_HEADER_RE.match(line)
            if not m:
                continue
            level = len(m.group(1))
            text = m.group(2).strip()
            headers.append({"text": text, "level": level, "page": text_to_page.get(text.lower())})
    else:
        for item, _level in result.document.iterate_items():
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                text = (item.text or "").strip()
                if not text:
                    continue
                item_level = 1 if isinstance(item, TitleItem) else getattr(item, "level", 1)
                page = item.prov[0].page_no if item.prov else None
                headers.append({"text": text, "level": item_level, "page": page})
    sidecar_path = output_dir / f"{pdf_path.stem}.headers.json"
    sidecar_path.write_text(json.dumps(headers, indent=2, ensure_ascii=False), encoding="utf-8")

    header_count, hierarchy_depth = _count_headers(markdown)

    return ConvertResult(
        markdown=markdown,
        output_path=output_path,
        timing_s=timing_s,
        header_count=header_count,
        hierarchy_depth=hierarchy_depth,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDF to markdown via Docling")
    parser.add_argument("pdf", type=Path, help="Input PDF file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    args = parser.parse_args()

    if not args.pdf.is_file():
        print(f"Error: {args.pdf} is not a file", file=sys.stderr)
        sys.exit(1)

    print("Loading Docling converter...")
    conv = create_converter()
    print(f"Converting {args.pdf.name}...")
    res = convert(args.pdf, args.output_dir, converter=conv)
    print(f"  Output: {res.output_path}")
    print(f"  Time:   {res.timing_s:.1f}s")
    print(f"  Headers: {res.header_count}  depth: {res.hierarchy_depth}")
