"""Docling-based PDF-to-markdown converter with ML layout detection."""

from __future__ import annotations

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

try:
    from hierarchical.postprocessor import ResultPostprocessor  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — optional private module
    ResultPostprocessor = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


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
        do_table_structure=False,
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
) -> ConvertResult:
    """Convert a PDF to markdown using Docling + hierarchical postprocessor."""
    if converter is None:
        converter = create_converter()

    t0 = time.monotonic()

    result = converter.convert(str(pdf_path))

    # Reconstruct heading hierarchy from PDF outline / TOC (optional private module)
    if ResultPostprocessor is not None:
        ResultPostprocessor(result, source=str(pdf_path)).process()

    # Serialize with custom heading formatter.
    # The hierarchical postprocessor can leave the document hierarchy in a state
    # that fails Pydantic validation, so use model_construct to skip validation.
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

    timing_s = time.monotonic() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text(markdown, encoding="utf-8")

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
