"""silver_structured: NuExtract3 structured extraction over v6b bronze markdown.

Text path by default (a pilot showed text + the split-grading schema matches
vision quality at a fraction of the cost). Vision is a *narrow fallback*, used
only when the bronze markdown is empty/tiny — i.e. scanned/image-only PDFs where
Docling produced no usable text. Output: silver/05_structured/<stem>.json.
"""

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.rag.models_v4 import SyllabusExtract

# Below this many non-whitespace chars, the markdown is too thin to extract from
# (image-only PDF) → fall back to vision on the page renders.
_MIN_TEXT_CHARS = 200
_VISION_MAX_PAGES = 8
_VISION_DPI = 150


def needs_vision(markdown: str, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """True when the markdown is too sparse to extract from (likely a scan)."""
    return len(markdown.strip()) < min_chars


def apply_course_code_fix(extract: SyllabusExtract, stem: str) -> SyllabusExtract:
    """Prepend the department when the model returns a bare number (e.g. '615' →
    'STAT 615'). The dept is authoritative from the filename, not the model."""
    cc = (extract.course_code or "").strip()
    if cc and not any(ch.isalpha() for ch in cc):
        extract.course_code = f"{dept_from_stem(stem)} {cc}"
    return extract


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, str, dict)):
        return len(value) > 0
    return True


def merge_extracts(extracts: list[SyllabusExtract]) -> SyllabusExtract:
    """Field-wise merge across per-page vision results: keep the first populated
    value seen for each field. Empty list → an empty SyllabusExtract."""
    merged = SyllabusExtract()
    for field in SyllabusExtract.model_fields:
        for ex in extracts:
            value = getattr(ex, field)
            if _is_populated(value):
                setattr(merged, field, value)
                break
    return merged


def _extract_via_vision(extractor, pdf_path) -> SyllabusExtract:
    import fitz  # pymupdf
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    page_extracts: list[SyllabusExtract] = []
    for page_index in range(min(len(doc), _VISION_MAX_PAGES)):
        pix = doc[page_index].get_pixmap(dpi=_VISION_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_extracts.append(extractor.extract_image(img))
    return merge_extracts(page_extracts)


def _compute_silver_structured(context: AssetExecutionContext) -> MaterializeResult:
    extractor = context.resources.nuextract.get_extractor()
    stem = context.partition_key
    dept = dept_from_stem(stem)

    md_path = paths.bronze_md_path(stem)
    if not md_path.exists():
        raise FileNotFoundError(f"v6b bronze md missing for {stem!r}: {md_path}")
    markdown = md_path.read_text(encoding="utf-8")

    used_vision = False
    if needs_vision(markdown):
        pdf = paths.raw_path(stem)
        if pdf.exists():
            used_vision = True
            extract = _extract_via_vision(extractor, pdf)
        else:
            extract = extractor.extract_text(markdown)
    else:
        extract = extractor.extract_text(markdown)

    extract = apply_course_code_fix(extract, stem)

    out_path = paths.silver_structured_path(stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(extract.model_dump_json(indent=2), encoding="utf-8")

    return MaterializeResult(
        metadata={
            "stem": stem,
            "dept": dept,
            "used_vision_fallback": used_vision,
            "course_code": extract.course_code or "",
            "instructor": extract.instructor_name or "",
            "meeting_slots": len(extract.meeting_schedule),
            "assessment_weights": len(extract.assessment_weights),
            "letter_grade_cutoffs": len(extract.letter_grade_cutoffs),
            "learning_outcomes": len(extract.learning_outcomes),
            "output_path": MetadataValue.path(str(out_path)),
            "sha256": hash_file(out_path),
        }
    )


silver_structured = asset(
    name="v6b_silver_structured",
    deps=[AssetKey("v6b_bronze_blocks")],
    partitions_def=stem_partitions,
    code_version=code_version_of(_compute_silver_structured),
    group_name="v6b_silver",
    description="NuExtract3 structured extraction (text path; vision fallback on empty md).",
    required_resource_keys={"nuextract"},
)(_compute_silver_structured)
