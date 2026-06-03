"""silver_structured: TAMU Gemini structured extraction over v6b bronze markdown.

Primary path: TAMU Gemini 2.5 Flash via ``tools.llm.call_llm`` — prompted
extraction of SyllabusExtract fields from clean markdown. Free, retry-hardened,
and (unlike NuExtract) prompt-able for the audit pain points around policy text,
dual-section grading, and long learning-outcome lists.

Vision is a narrow fallback for scanned/image-only PDFs where the bronze
markdown is empty; it routes per-page renders through NuExtract3 (4-bit). The
NuExtract resource is only touched in that fallback, so the model is never
loaded on a pure-text department.

Output: silver/05_structured/<stem>.json.

The same per-stem extraction is driven two ways, both through ``process_stems``:
the Dagster asset (one partition, or a whole department in one single-run
backfill) and ``scripts/run_silver_structured_batch.py``.
"""

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from dagster import (
    AssetExecutionContext,
    AssetKey,
    BackfillPolicy,
    Failure,
    MaterializeResult,
    MetadataValue,
    asset,
)

from tamubot.ingestion.pipeline_v5.util import code_version_of, dept_from_stem, hash_file
from tamubot.ingestion.pipeline_v6b import paths
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions
from tamubot.rag.models_v4 import SyllabusExtract
from tamubot.rag.tools.llm import call_llm

# Below this many non-whitespace chars, the markdown is too thin to extract from
# (image-only PDF) → fall back to vision on the page renders.
_MIN_TEXT_CHARS = 200
_VISION_MAX_PAGES = 8
_VISION_DPI = 150

# Bounded concurrency for TAMU calls. The gateway is stable in practice; this is
# conservative and can be ramped up if throughput becomes the bottleneck.
_GEMINI_MAX_WORKERS = 4

# Per-stem retry budget for degenerate Gemini output (empty fields, no
# identifiers). The TAMU gateway returns degenerate shapes ~20% of the time
# even at temperature=0 — see metadata_enrichment.py for the same mitigation.
_MAX_GEMINI_ATTEMPTS = 3

# Output cap. Long syllabi with many learning outcomes can produce ~2k tokens
# of JSON; 8192 leaves headroom and well exceeds the 4096 SSE-empty-response
# floor noted in rag/CLAUDE.md.
_GEMINI_MAX_TOKENS = 8192


# The prompt explicitly addresses the audit pain points found on the prior
# NuExtract output: policy fields defaulting to university boilerplate;
# assessment_weights non-deterministic across dual-section syllabi;
# learning_outcomes dropping bullets from long lists; F-cutoff semantics
# flipping between "F starts at 59" and the correct "F is the 0-floor ceiling".
SILVER_STRUCTURED_PROMPT = """\
You are a university syllabus parser. Extract structured information from the syllabus markdown.

Return ONLY a single JSON object — no markdown fences, no commentary. The object MUST have exactly these fields (use null or [] when a field is not stated in the syllabus):

{
  "course_code":               "string|null",
  "course_title":              "string|null",
  "instructor_name":           "string|null",
  "instructor_email":          "string|null",
  "credit_hours":              integer|null,
  "meeting_schedule":          [{"day": "string|null", "time": "string|null", "location": "string|null"}, ...],
  "assessment_weights":        [{"component": "string", "weight_pct": number}, ...],
  "letter_grade_cutoffs":      [{"grade": "string", "min_percent": number}, ...],
  "prerequisites":             ["string", ...],
  "learning_outcomes":         ["string", ...],
  "attendance_policy":         "string|null",
  "academic_integrity_policy": "string|null"
}

EXTRACTION RULES:
- GROUNDING: Use only text that appears in the syllabus. Do not infer, add domain knowledge, or paraphrase.
- COURSE CODE: Always include a space between the department letters and the course number ("STAT 615", not "STAT615" or bare "615").
- POLICY FIELDS (attendance_policy, academic_integrity_policy): TAMU syllabi typically contain BOTH a university-wide boilerplate paragraph AND a course-specific rule (e.g. "you may miss up to two seminars"). Prefer the course-specific rule. If both are essential, concatenate the course-specific rule first, then a blank line, then the boilerplate. If only boilerplate is present, use it. Capture the FULL paragraph verbatim — do not truncate mid-sentence.
- LETTER GRADE CUTOFFS: F is the bottom of an A/B/C/D/F ladder — the ceiling, not a "minimum". When the syllabus says "F: 0-59" or "F < 60", encode it as {"grade": "F", "min_percent": 0}. Never encode F with a non-zero min_percent.
- ASSESSMENT WEIGHTS: List every distinct graded component with its percentage. For dual-section or split-grading syllabi (e.g. "Section 600: Homework 40%, Exams 60%; Section 700: Homework 30%, Exams 70%"), emit ONE row per (section, component) and disambiguate via the component name: {"component": "Homework (Sec 600)", "weight_pct": 40}. Never sum across sections.
- LEARNING OUTCOMES: Preserve every bullet from the syllabus's stated learning outcomes section verbatim. Do not summarize, paraphrase, or drop entries to fit. If the syllabus lists 7 outcomes, the array MUST have 7 entries.
- PREREQUISITES: One short string per distinct prerequisite (e.g. "STAT 614", "Approval of instructor").
- TYPOS: Preserve typos in the source (e.g. "presentaion") — do not silently correct.
- MISSING: Use null / [] when the field is not stated. Do not fabricate.

Return ONLY the JSON object."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def parse_syllabus_json(raw: str) -> SyllabusExtract:
    """Parse raw model output into a SyllabusExtract. Tolerates ``` fences,
    surrounding prose, and minor structural slips (dropped braces, trailing
    commas) via json-repair. Raises ValueError if no JSON object is present."""
    text = _strip_json_fence(raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in model output: {raw[:200]!r}") from None
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            from json_repair import repair_json
            data = json.loads(repair_json(candidate))
    return SyllabusExtract.model_validate(data)


def _is_degenerate(extract: SyllabusExtract) -> bool:
    # TAMU gateway sometimes returns the schema with nulls only. Detect via the
    # union of identity fields — if none are populated, the call effectively
    # failed and a retry is warranted.
    return not (extract.course_code or extract.instructor_name or extract.course_title)


def extract_text_via_gemini(markdown: str) -> SyllabusExtract:
    """Send the bronze markdown to TAMU Gemini and return a parsed extract.

    Retries up to ``_MAX_GEMINI_ATTEMPTS`` on parse errors or degenerate output
    (empty identifier fields). ``call_llm`` already handles transport-level
    retries, so this loop only fights JSON-shape failures."""
    last: SyllabusExtract | None = None
    last_exc: Exception | None = None
    for _ in range(_MAX_GEMINI_ATTEMPTS):
        try:
            result = call_llm(
                [
                    {"role": "system", "content": SILVER_STRUCTURED_PROMPT},
                    {"role": "user", "content": markdown},
                ],
                temperature=0.0,
                max_tokens=_GEMINI_MAX_TOKENS,
                json_mode=True,
            )
            last = parse_syllabus_json(result.text)
        except Exception as exc:  # noqa: BLE001 — surface only after all retries
            last_exc = exc
            continue
        if not _is_degenerate(last):
            return last
    if last is not None:
        # All attempts parsed but were degenerate; surface the last shape rather
        # than the exception (caller can decide if empty-identifier is fatal).
        return last
    assert last_exc is not None
    raise last_exc


# ── Tier 1 deterministic post-processors ─────────────────────────────────────


_COURSE_CODE_SPACE_RE = re.compile(r"^([A-Z]{2,5})(\d+[A-Z]?)$")
# Stray "apostrophe-newline-apostrophe" sentence-glue seen in the NuExtract era;
# kept as a safety net for the rare Gemini run that leaks similar artifacts.
_APO_GLUE_RE = re.compile(r"['‘’]\s*\n\s*['‘’]")
# Trailing JSON-syntax artifacts inside a string value.
_TRAILING_JUNK_RE = re.compile(r"[}\"'‘’]+\s*$")


def _fix_course_code(code: str | None, stem: str) -> str | None:
    """Normalize course_code: prepend dept for bare numbers, insert missing space.

    Examples:
        "615"     → "STAT 615"   (dept inferred from the stem)
        "STAT615" → "STAT 615"   (missing space inserted)
        "STAT 615" → unchanged
        None / "" → unchanged
    """
    if not code:
        return code
    cc = code.strip()
    if not any(ch.isalpha() for ch in cc):
        return f"{dept_from_stem(stem)} {cc}"
    m = _COURSE_CODE_SPACE_RE.match(cc)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return cc


def _fix_f_cutoff(cutoffs):
    """When an A/B/C/D ladder is present, F is the 0-floor ceiling — its
    min_percent must be 0. Models occasionally encode it as the previous grade's
    boundary (e.g. 59), which downstream reads as "F starts at 59"."""
    if not cutoffs:
        return cutoffs
    grades = {(c.grade or "").strip().upper() for c in cutoffs}
    if "F" in grades and grades & {"A", "B", "C", "D"}:
        for c in cutoffs:
            if (c.grade or "").strip().upper() == "F" and c.min_percent not in (None, 0, 0.0):
                c.min_percent = 0.0
    return cutoffs


def _clean_policy(text: str | None) -> str | None:
    """Strip JSON-syntax artifacts that occasionally leak into string values."""
    if not text:
        return text
    cleaned = _APO_GLUE_RE.sub("\n\n", text)
    cleaned = _TRAILING_JUNK_RE.sub("", cleaned)
    return cleaned.strip() or None


def clean_extract(extract: SyllabusExtract, stem: str) -> SyllabusExtract:
    """Deterministic post-processor for known extraction artifacts. Idempotent —
    applying twice is the same as once."""
    extract.course_code = _fix_course_code(extract.course_code, stem)
    extract.letter_grade_cutoffs = _fix_f_cutoff(extract.letter_grade_cutoffs)
    extract.attendance_policy = _clean_policy(extract.attendance_policy)
    extract.academic_integrity_policy = _clean_policy(extract.academic_integrity_policy)
    return extract


# ── Vision fallback (NuExtract, per-page) ────────────────────────────────────


def needs_vision(markdown: str, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """True when the markdown is too sparse to extract from (likely a scan)."""
    return len(markdown.strip()) < min_chars


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, str, dict)):
        return len(value) > 0
    return True


def _dedupe_key(item: object) -> str:
    """Order-preserving dedupe key for a list item (str / pydantic model / dict)."""
    dump = getattr(item, "model_dump_json", None)
    if callable(dump):
        return dump()
    try:
        return json.dumps(item, sort_keys=True, default=str)
    except TypeError:
        return str(item)


def merge_extracts(extracts: list[SyllabusExtract]) -> SyllabusExtract:
    """Field-wise merge across per-page vision results.

    Scalar fields keep the first populated value. **List** fields
    (learning_outcomes, assessment_weights, meeting_schedule, …) are concatenated
    across pages with order-preserving dedupe — otherwise a list that spans pages
    would be truncated to page 1's entries (FID_CONTENT_DROPPED on multi-page
    scans). Empty list → an empty SyllabusExtract."""
    merged = SyllabusExtract()
    for field_name in SyllabusExtract.model_fields:
        first = next(
            (getattr(ex, field_name) for ex in extracts if _is_populated(getattr(ex, field_name))),
            None,
        )
        if first is None:
            continue
        if isinstance(first, list):
            combined: list = []
            seen: set[str] = set()
            for ex in extracts:
                for item in getattr(ex, field_name) or []:
                    key = item if isinstance(item, str) else _dedupe_key(item)
                    if key in seen:
                        continue
                    seen.add(key)
                    combined.append(item)
            setattr(merged, field_name, combined)
        else:
            setattr(merged, field_name, first)
    return merged


def _all_populated(extract: SyllabusExtract) -> bool:
    return all(_is_populated(getattr(extract, f)) for f in SyllabusExtract.model_fields)


def _extract_via_vision(nuextract, pdf_path) -> SyllabusExtract:
    import fitz  # pymupdf
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    page_extracts: list[SyllabusExtract] = []
    for page_index in range(min(len(doc), _VISION_MAX_PAGES)):
        pix = doc[page_index].get_pixmap(dpi=_VISION_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_extracts.append(nuextract.extract_image(img))
        # Each page is a ~30s vision call; stop once every field is filled rather
        # than always rendering all 8 pages.
        if _all_populated(merge_extracts(page_extracts)):
            break
    return merge_extracts(page_extracts)


# ── Per-stem driver ──────────────────────────────────────────────────────────


def extract_stem(stem: str, nuextract_getter: Callable[[], object] | None = None) -> tuple[SyllabusExtract, bool]:
    """Run extraction over one stem's bronze markdown.

    Returns (extract, used_vision). The vision fallback only fires when the
    markdown is too thin AND a source PDF exists; ``nuextract_getter`` is a
    zero-arg callable that returns the loaded NuExtract extractor, called only
    on the fallback path so a pure-text department never pays the model-load
    cost."""
    md_path = paths.bronze_md_path(stem)
    if not md_path.exists():
        raise FileNotFoundError(f"v6b bronze md missing for {stem!r}: {md_path}")
    markdown = md_path.read_text(encoding="utf-8")
    if needs_vision(markdown):
        pdf = paths.raw_path(stem)
        if pdf.exists() and nuextract_getter is not None:
            extract = _extract_via_vision(nuextract_getter(), pdf)
            return clean_extract(extract, stem), True
    extract = extract_text_via_gemini(markdown)
    return clean_extract(extract, stem), False


# ── Batch driver ─────────────────────────────────────────────────────────────


@dataclass
class BatchSummary:
    """Outcome of a ``process_stems`` run."""

    ok: int = 0
    errors: int = 0
    vision: int = 0
    elapsed_s: float = 0.0
    failed: list[str] = field(default_factory=list)


def process_stems(
    stems: list[str],
    *,
    nuextract_getter: Callable[[], object] | None = None,
    force: bool = False,
    log: Callable[..., None] = print,
    on_done: Callable[[str, SyllabusExtract, bool], None] | None = None,
) -> BatchSummary:
    """Extract every stem, writing silver/05_structured/<stem>.json.

    TAMU Gemini drives the text path in bounded-concurrency parallel calls. The
    vision fallback runs sequentially (per-page calls are slow and rare). Stems
    whose output already exists are skipped unless ``force``.

    ``on_done(stem, extract, used_vision)`` fires once per successful stem
    (after its JSON is written). Per-stem failures are collected in the returned
    summary rather than raised, so one bad doc doesn't abort the batch.
    """
    summary = BatchSummary()
    pending = [s for s in stems if force or not paths.silver_structured_path(s).exists()]
    log(f"{len(stems)} stems ({len(stems) - len(pending)} already done, {len(pending)} to run)")
    if not pending:
        return summary

    def finish(stem: str, extract: SyllabusExtract, used_vision: bool) -> None:
        out = paths.silver_structured_path(stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(extract.model_dump_json(indent=2), encoding="utf-8")
        summary.ok += 1
        if used_vision:
            summary.vision += 1
        if on_done is not None:
            on_done(stem, extract, used_vision)

    def fail(stem: str, exc: Exception, label: str) -> None:
        summary.errors += 1
        summary.failed.append(stem)
        log(f"  {label} {stem}: ERROR {type(exc).__name__}: {exc}")

    # Classify pending stems: text vs vision fallback.
    text_items: list[tuple[str, str]] = []
    vision_stems: list[str] = []
    for stem in pending:
        md_path = paths.bronze_md_path(stem)
        if not md_path.exists():
            fail(stem, FileNotFoundError(str(md_path)), "[bronze]")
            continue
        md = md_path.read_text(encoding="utf-8")
        if needs_vision(md) and paths.raw_path(stem).exists():
            vision_stems.append(stem)
        else:
            text_items.append((stem, md))

    t0 = time.time()

    if text_items:
        n = len(text_items)
        with ThreadPoolExecutor(max_workers=_GEMINI_MAX_WORKERS) as pool:
            futures = {pool.submit(extract_text_via_gemini, md): stem for stem, md in text_items}
            for done, fut in enumerate(as_completed(futures), 1):
                stem = futures[fut]
                try:
                    ex = fut.result()
                except Exception as exc:  # noqa: BLE001
                    fail(stem, exc, f"[gemini {done}/{n}]")
                    continue
                ex = clean_extract(ex, stem)
                finish(stem, ex, False)
                log(f"  [gemini {done}/{n}] {stem}: {ex.course_code or '-'}")

    for stem in vision_stems:
        if nuextract_getter is None:
            fail(stem, RuntimeError("vision fallback needed but no NuExtract extractor available"), "[vision]")
            continue
        try:
            extract, used_vision = extract_stem(stem, nuextract_getter)
        except Exception as exc:  # noqa: BLE001
            fail(stem, exc, "[vision]")
            continue
        finish(stem, extract, used_vision)

    summary.elapsed_s = time.time() - t0
    log(
        f"[done] {summary.ok} ok, {summary.errors} errors, "
        f"{summary.vision} via vision in {summary.elapsed_s / 60:.1f} min"
    )
    return summary


# ── Dagster asset ────────────────────────────────────────────────────────────


def _compute_silver_structured(context: AssetExecutionContext) -> MaterializeResult:
    stems = list(context.partition_keys)
    nuextract_resource = context.resources.nuextract

    def record(stem: str, extract: SyllabusExtract, used_vision: bool) -> None:
        out_path = paths.silver_structured_path(stem)
        context.add_asset_metadata(
            {
                "dept": dept_from_stem(stem),
                "used_vision_fallback": used_vision,
                "course_code": extract.course_code or "",
                "instructor": extract.instructor_name or "",
                "meeting_slots": len(extract.meeting_schedule),
                "assessment_weights": len(extract.assessment_weights),
                "letter_grade_cutoffs": len(extract.letter_grade_cutoffs),
                "learning_outcomes": len(extract.learning_outcomes),
                "output_path": MetadataValue.path(str(out_path)),
                "sha256": hash_file(out_path),
            },
            partition_key=stem,
        )

    summary = process_stems(
        stems,
        nuextract_getter=nuextract_resource.get_extractor,
        log=context.log.info,
        on_done=record,
    )

    # Fail loud: a failed stem wrote no JSON, so its partition must not go green.
    # Successful stems are already on disk → a skip-done retry re-runs only the
    # failures, and a fully clean run turns the whole range green.
    if summary.failed:
        raise Failure(
            description=(f"{len(summary.failed)} of {len(stems)} stem(s) failed: {', '.join(summary.failed)}")
        )

    return MaterializeResult(
        metadata={"backend": "tamu_gemini", "vision_fallback_count": summary.vision},
    )


silver_structured = asset(
    name="v6b_silver_structured",
    deps=[AssetKey("v6b_bronze_blocks")],
    partitions_def=stem_partitions,
    backfill_policy=BackfillPolicy.single_run(),
    code_version=code_version_of(_compute_silver_structured),
    group_name="v6b_silver",
    description="TAMU Gemini structured extraction (text path; NuExtract vision fallback for scanned PDFs).",
    required_resource_keys={"nuextract"},
)(_compute_silver_structured)
