"""Filter: enrich markdown syllabi with course metadata and summary.

Extracts metadata from three sources:
  1. Filename — course_id, section, term, crn (deterministic)
  2. LLM — instructor, TAs, meeting_times, location, credit_hours
  3. Scraped metadata JSON — syllabus_url, doc_id (Simple Syllabus only)

Also generates a COURSE_SUMMARY keyword index via LLM.

Unlike other filters (markdown → markdown), this outputs JSON per file.

CLI: python -m tamubot.ingestion.filters.metadata_enrichment input_dir/ output_dir/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tamubot.core import config
from tamubot.ingestion.filters.base import FilterResult
from tamubot.ingestion.pipeline_v4 import code_to_term

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_ROOT = Path("tamu_data/raw")
MAX_RETRIES = 2
MODEL = config.TAMU_MODEL

METADATA_PROMPT = """\
You are a university syllabus parser. Extract ONLY the following metadata fields from the text below.

Return ONLY valid JSON with this exact structure (omit fields you cannot find):
{
  "instructor": {
    "name": "...",
    "email": "...",
    "office": "...",
    "office_hours": "..."
  },
  "teaching_assistants": [{"name": "...", "email": "..."}],
  "meeting_times": "...",
  "location": "...",
  "credit_hours": "..."
}

Rules:
- Extract only what is explicitly stated. Do not infer or hallucinate.
- Return null for fields not found.
- Output must be valid JSON only — no markdown fences, no explanation.
"""

SUMMARY_PROMPT = """\
You are a university syllabus parser. Generate a COURSE_SUMMARY — a RAG-optimized \
keyword index for the syllabus below.

Use this strict format:

  {course_id} [/ <crosslist>] | <Full Course Title> | {term}
  Instructor: <Full Name> | <email>
  Meets: <days and times> | <location>
  Grading: <component [weight%], component [weight%], ...> — list every graded \
component with its percentage weight exactly as stated in the syllabus. Use the \
format "Component [XX%]". Include the grade scale if provided (e.g., A: 90-100).
  Topics: <ALL named concepts, techniques, algorithms, models, and skills from the \
ENTIRE syllabus — course description, learning outcomes, AND weekly schedule — merged \
into one comma-separated list. Rare/specialized terms first, broader ones last. \
No duplicates.>
  Tools: <ALL software, platforms, applications, environments, and services that \
students USE or interact with for coursework — include LMS platforms (Canvas, \
Blackboard), submission tools (Turnitin), modeling/diagramming tools (Visio, Lucidchart), \
project management tools (MS Project, Jira), programming environments, simulation \
software, cloud platforms, and any other named technology. Omit only: personal \
communication channels (email, phone), test-taking hardware (webcam, Chromebooks), \
and university support resources (library tech bar, OAL lab). If a tool is mentioned \
in the syllabus text, include it.>
  Prerequisites: <exact course codes and standing; omit if none stated>

Rules:
- GROUNDING: Use ONLY terms and phrases that appear explicitly in the source text. \
Do NOT add domain knowledge or infer anything not written. If "Python" is not in \
the text, it is not in the summary.
- TOPICS IS EVERYTHING: Topics is the single comprehensive concept list — merge what \
would otherwise be topics, methods, skills, and niche into Topics. Do not create \
separate fields for these.
- NO DUPLICATION: No term appears twice anywhere in the summary. Tools must not \
repeat terms already in Topics.
- RARE TERMS FIRST: List specialized/rare terms before generic ones in Topics. \
Rare terms provide stronger keyword signal for search.
- BE THOROUGH: Extract ALL named algorithms, models, techniques, tools, and skills \
from the entire syllabus — do not truncate.
- EXCLUDE from Topics: academic integrity terms (plagiarism, complicity, fabrication, \
cheating), grading process terms (late penalties, regrading, test cases, grade appeals, \
incomplete grades), and software environment setup steps (installing libraries, \
configuring environments, executing compilers, running programs). These are course \
policies, not knowledge areas.
- Retain ALL proper nouns, theorem names, algorithm names, drug names, and tool \
names verbatim.
- NEVER write narrative: "students will learn", "designed to", "upon completion", \
"this course", "you will", "gain experience", or any similar framing.
- Use declarative noun phrases only.
- If a field has no data in the source text, omit that line entirely.
- Target 300–450 tokens total.
- Output the summary text only — no JSON, no markdown fences.
"""

# ── JSON / LLM helpers ────────────────────────────────────────────────────────


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw.strip()


def _sanitize_json(raw: str) -> str:
    raw = raw.replace("\x00", "")
    raw = re.sub(r"\\([^\"\\/bfnrtu0-9])", r"\\\\\1", raw)
    raw = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    return raw


def _clean_replacement_chars(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.replace("\ufffd", "-")
    if isinstance(obj, dict):
        return {k: _clean_replacement_chars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_replacement_chars(v) for v in obj]
    return obj


def _llm_call(client: Any, prompt: str, text: str) -> tuple[str, str]:
    """Make an LLM call with retries. Returns (raw_response, error)."""
    user_message = f"{prompt}\n\n---\n\nSYLLABUS TEXT:\n{text}"
    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.0,
                max_tokens=4096,
                stream=True,
            )
            raw = "".join(c.choices[0].delta.content or "" for c in stream).strip()
            return raw, ""
        except Exception as e:
            if attempt == MAX_RETRIES:
                return "", str(e)
            print(f"    WARN: LLM attempt {attempt + 1} failed ({e}), retrying...")
    return "", "unknown error"


def dedup_course_summary(content: str) -> str:
    """Remove duplicate terms in COURSE_SUMMARY.

    Deduplicates within Topics, strips Tools terms already in Topics,
    and drops legacy fields the model might emit.
    """

    def parse_terms(s: str) -> list[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    def norm(t: str) -> str:
        return t.lower().strip()

    lines = content.split("\n")
    result_lines: list[str] = []
    topics_seen: set[str] = set()

    for line in lines:
        m = re.match(r"^(\s*)(Topics):\s*(.*)", line)
        if m:
            indent, _, terms_str = m.groups()
            unique = []
            for t in parse_terms(terms_str):
                if norm(t) not in topics_seen:
                    topics_seen.add(norm(t))
                    unique.append(t)
            result_lines.append(f"{indent}Topics: {', '.join(unique)}")
            continue

        m = re.match(r"^(\s*)(Tools):\s*(.*)", line)
        if m:
            indent, _, terms_str = m.groups()
            unique = [t for t in parse_terms(terms_str) if norm(t) not in topics_seen]
            if unique:
                result_lines.append(f"{indent}Tools: {', '.join(unique)}")
            continue

        if re.match(r"^\s*(Skills|Methods|Tools/Platforms|Niche|Schedule):", line):
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


# Generic filler terms to strip from Topics
_GENERIC_TOPICS = {
    "course overview",
    "course introduction",
    "introduction",
    "final review",
    "basic ideas of statistical learning",
}


def _strip_generic_topics(content: str) -> str:
    """Remove generic filler terms from the Topics line."""
    lines = content.split("\n")
    result: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*Topics:\s*)(.*)", line)
        if m:
            prefix, terms_str = m.groups()
            terms = [t.strip() for t in terms_str.split(",") if t.strip()]
            filtered = [t for t in terms if t.lower() not in _GENERIC_TOPICS]
            result.append(f"{prefix}{', '.join(filtered)}")
        else:
            result.append(line)
    return "\n".join(result)


def _fix_summary_header(content: str, course_id: str) -> str:
    """Ensure the first line of the summary starts with the course_id."""
    if not content or not course_id:
        return content
    lines = content.split("\n")
    if lines and course_id not in lines[0]:
        # Prepend course_id to existing header
        lines[0] = f"{course_id} {lines[0].lstrip('| ')}"
    return "\n".join(lines)


def _clean_summary_locations(content: str) -> str:
    """Normalize location artifacts in the summary Meets line."""
    lines = content.split("\n")
    result: list[str] = []
    for line in lines:
        if line.startswith("Meets:"):
            line = re.sub(r"ONLINE\s+ONLINE", "Online", line, flags=re.IGNORECASE)
        result.append(line)
    return "\n".join(result)


def _count_topics(content: str) -> int:
    """Count number of terms in the Topics line."""
    for line in content.split("\n"):
        if line.strip().startswith("Topics:"):
            terms = line.split(":", 1)[1]
            return len([t for t in terms.split(",") if t.strip()])
    return 0


def _is_truncated(content: str) -> bool:
    """Check if the summary appears truncated (ends mid-phrase)."""
    stripped = content.rstrip()
    if not stripped:
        return False
    # Ends with a preposition, article, or comma
    last_word = stripped.split()[-1].lower() if stripped.split() else ""
    if stripped.endswith(","):
        return True
    if last_word in ("in", "of", "the", "a", "an", "and", "or", "for", "to", "with", "from"):
        return True
    return False


def _normalize_location(location: str | None) -> str | None:
    """Clean up common location artifacts."""
    if not location:
        return location
    # Deduplicate "ONLINE ONLINE" → "Online"
    if re.match(r"^ONLINE\s+ONLINE$", location, re.IGNORECASE):
        return "Online"
    return location


# ── Header-to-page mapping ───────────────────────────────────────────────────

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def find_source_pdf(stem: str) -> Path | None:
    """Locate the source PDF for a given file stem in tamu_data/raw/."""
    # Strip _HP suffix and version suffix (_vNNN) for lookup
    lookup_stem = stem.removesuffix("_HP")
    lookup_stem = re.sub(r"_v\d{3}$", "", lookup_stem)
    for pdf in RAW_ROOT.rglob(f"{lookup_stem}.pdf"):
        return pdf
    return None


def map_headers_to_pages(markdown: str, pdf_path: Path) -> list[dict]:
    """Map each markdown header to its page in the source PDF.

    Uses PyMuPDF to extract text per page, then matches header text
    sequentially (each header must be on the same or later page).
    """
    import fitz  # PyMuPDF

    headers = [(len(m.group(1)), m.group(2).strip()) for m in _HEADER_RE.finditer(markdown)]
    if not headers:
        return []

    doc = fitz.open(str(pdf_path))
    page_texts = [page.get_text() for page in doc]
    doc.close()

    def _find_page(header_text: str, min_page: int = 0) -> int | None:
        h_words = header_text.split()
        for i in range(min_page, len(page_texts)):
            pt = page_texts[i]
            if len(h_words) >= 2:
                pattern = r"\s+".join(re.escape(w) for w in h_words)
                if re.search(pattern, pt, re.IGNORECASE):
                    return i + 1
            else:
                if re.search(r"\b" + re.escape(header_text) + r"\b", pt, re.IGNORECASE):
                    return i + 1
        return None

    results = []
    min_page = 0
    for level, header_text in headers:
        page_num = _find_page(header_text, min_page)
        if page_num is not None:
            min_page = page_num - 1
        results.append({"text": header_text, "level": level, "page": page_num})

    return results


# ── Filename parsing ──────────────────────────────────────────────────────────


def parse_filename(stem: str) -> dict[str, str]:
    """Parse ``{term_code}_{subject}_{course}_{section}_{crn}`` into metadata.

    Returns dict with course_id, section, term, crn, term_code.
    """
    parts = stem.split("_")
    if len(parts) < 5:
        return {}
    term_code = parts[0]
    subject = parts[1]
    course = parts[2]
    section = parts[3]
    crn = parts[4]
    try:
        term = code_to_term(term_code)
    except Exception:
        term = term_code
    return {
        "course_id": f"{subject} {course}",
        "section": section,
        "term": term,
        "crn": crn,
        "term_code": term_code,
    }


# ── LLM extraction ───────────────────────────────────────────────────────────


def extract_metadata_llm(markdown: str, client: Any) -> tuple[dict, str]:
    """Extract instructor/TA/schedule metadata via LLM."""
    raw, error = _llm_call(client, METADATA_PROMPT, markdown)
    if error:
        return {}, error
    try:
        raw = _strip_fences(raw)
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = json.loads(_sanitize_json(raw))
        return _clean_replacement_chars(meta), ""
    except Exception as e:
        return {}, str(e)


def generate_course_summary(
    markdown: str,
    course_id: str,
    term: str,
    client: Any,
) -> tuple[str, str]:
    """Generate a COURSE_SUMMARY keyword index via LLM.

    Retries once if the result has fewer than 5 topics or is truncated.
    Applies post-processing: dedup, strip generic terms, fix header.
    """
    prompt = SUMMARY_PROMPT.replace("{course_id}", course_id).replace("{term}", term)
    raw, error = _llm_call(client, prompt, markdown)
    if error:
        return "", error
    raw = _strip_fences(raw)
    summary = dedup_course_summary(raw)

    # Retry once if quality is poor
    if _count_topics(summary) < 5 or _is_truncated(summary):
        print("    WARN: summary quality low, retrying...")
        raw2, error2 = _llm_call(client, prompt, markdown)
        if not error2:
            raw2 = _strip_fences(raw2)
            candidate = dedup_course_summary(raw2)
            if _count_topics(candidate) > _count_topics(summary):
                summary = candidate

    summary = _strip_generic_topics(summary)
    summary = _fix_summary_header(summary, course_id)
    summary = _clean_summary_locations(summary)
    return summary, ""


# ── Scraped metadata ─────────────────────────────────────────────────────────


def load_scraped_metadata() -> dict[str, dict]:
    """Build {stem: {syllabus_url, doc_id}} from metadata JSON files."""
    lookup: dict[str, dict] = {}
    # Old layout: simple_syllabus_*/simple_syllabus_metadata.json
    for meta_file in sorted(RAW_ROOT.glob("simple_syllabus_*/simple_syllabus_metadata.json")):
        try:
            entries = json.loads(meta_file.read_text(encoding="utf-8"))
            for filename, info in entries.items():
                stem = filename.removesuffix(".pdf")
                lookup[stem] = info
        except Exception as e:
            print(f"  WARN: Could not load {meta_file}: {e}")
    # New layout: simple_syllabus/<DEPT>/<level>/<Term>/metadata.json
    for meta_file in sorted(RAW_ROOT.glob("simple_syllabus/**/metadata.json")):
        try:
            entries = json.loads(meta_file.read_text(encoding="utf-8"))
            for filename, info in entries.items():
                stem = filename.removesuffix(".pdf")
                lookup[stem] = info
        except Exception as e:
            print(f"  WARN: Could not load {meta_file}: {e}")
    # Flat metadata file
    flat = RAW_ROOT / "simple_syllabus_metadata.json"
    if flat.exists():
        try:
            entries = json.loads(flat.read_text(encoding="utf-8"))
            for filename, info in entries.items():
                stem = filename.removesuffix(".pdf")
                lookup[stem] = info
        except Exception as e:
            print(f"  WARN: Could not load {flat}: {e}")
    return lookup


def detect_source(stem: str) -> str:
    """Determine if a syllabus came from simple_syllabus or howdy_portal."""
    # _HP suffix is a definitive marker for Howdy Portal files
    if stem.endswith("_HP"):
        return "howdy_portal"
    # Strip version suffix for PDF lookup
    lookup_stem = re.sub(r"_v\d{3}$", "", stem)
    for source_dir in RAW_ROOT.glob("howdy_portal/**/*.pdf"):
        if source_dir.stem == lookup_stem:
            return "howdy_portal"
    for source_dir in RAW_ROOT.glob("simple_syllabus/**/*.pdf"):
        if source_dir.stem == lookup_stem:
            return "simple_syllabus"
    return "unknown"


# ── Course type & format classification ──────────────────────────────────────

_COURSE_TYPE_MAP: dict[str, str] = {
    "681": "seminar",
    "684": "internship",
    "689": "special_topics",
    "691": "research",
}


def classify_course_type(course_id: str) -> str:
    """Classify course type from course_id (e.g. 'ISEN 681' → 'seminar')."""
    parts = course_id.split()
    if len(parts) >= 2:
        return _COURSE_TYPE_MAP.get(parts[1], "course")
    return "course"


def detect_format(location: str | None, meeting_times: str | None) -> str:
    """Determine if a section is online or in-person from location/meeting info."""
    combined = f"{location or ''} {meeting_times or ''}".upper()
    if "ONLINE" in combined or "REMOTE" in combined or "N/A" in combined:
        return "online"
    return "in-person"


# ── Main enrichment logic ────────────────────────────────────────────────────


def enrich_file(md_path: Path, output_dir: Path, client: Any, scraped_meta: dict[str, dict]) -> dict[str, Any]:
    """Enrich a single markdown file.

    Returns dict with keys: source_file, pipeline_version, source, course_type,
    course_metadata, course_summary, and a top-level ``_log`` (excluded from the
    written JSON, used only for report generation).
    """
    stem = md_path.stem
    markdown = md_path.read_text(encoding="utf-8")

    # Step 1: Filename parsing
    filename_meta = parse_filename(stem)
    filename_parsed = bool(filename_meta)
    course_id = filename_meta.get("course_id", "")
    section = filename_meta.get("section", "")
    term = filename_meta.get("term", "")
    crn = filename_meta.get("crn", "")

    # Step 2: LLM metadata extraction
    llm_meta, llm_meta_error = extract_metadata_llm(markdown, client)

    # Step 3: LLM course summary
    summary, llm_summary_error = generate_course_summary(
        markdown,
        course_id,
        term,
        client,
    )

    # Step 4: Scraped metadata injection
    # Strip _HP and version suffixes to match scraped metadata keys
    lookup_stem = stem.removesuffix("_HP")
    lookup_stem = re.sub(r"_v\d{3}$", "", lookup_stem)
    scraped = scraped_meta.get(lookup_stem, scraped_meta.get(stem, {}))
    scraped_found = bool(scraped)
    source = detect_source(stem)

    if source == "howdy_portal":
        syllabus_url = None
    else:
        syllabus_url = scraped.get("syllabus_url") or None

    # Step 5: Classify course type and delivery format
    course_type = classify_course_type(course_id)
    location = _normalize_location(llm_meta.get("location"))
    meeting_times = llm_meta.get("meeting_times")
    delivery_format = detect_format(location, meeting_times)

    # Build course_metadata
    course_metadata = {
        "course_id": course_id,
        "section": section,
        "term": term,
        "crn": crn,
        "instructor": llm_meta.get("instructor"),
        "teaching_assistants": llm_meta.get("teaching_assistants", []),
        "meeting_times": meeting_times,
        "location": location,
        "format": delivery_format,
        "credit_hours": llm_meta.get("credit_hours"),
        "syllabus_url": syllabus_url,
    }

    # Step 6: Header-to-page mapping
    pdf_path = find_source_pdf(stem)
    if pdf_path:
        headers = map_headers_to_pages(markdown, pdf_path)
    else:
        headers = [
            {"text": m.group(2).strip(), "level": len(m.group(1)), "page": None} for m in _HEADER_RE.finditer(markdown)
        ]

    result = {
        "source_file": stem,
        "pipeline_version": "v4",
        "source": source,
        "course_type": course_type,
        "course_metadata": course_metadata,
        "course_summary": summary,
        "headers": headers,
    }

    # Internal log — not written to JSON, used for report generation
    result["_log"] = {
        "filename_parsed": filename_parsed,
        "llm_metadata_status": "error" if llm_meta_error else "ok",
        "llm_metadata_error": llm_meta_error or None,
        "llm_summary_status": "error" if llm_summary_error else "ok",
        "llm_summary_error": llm_summary_error or None,
        "scraped_meta_found": scraped_found,
        "pdf_found": pdf_path is not None,
        "header_count": len(headers),
        "headers_with_page": sum(1 for h in headers if h["page"] is not None),
    }

    # Write output — exclude internal _log from persisted JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.json"
    persisted = {k: v for k, v in result.items() if not k.startswith("_")}
    out_path.write_text(json.dumps(persisted, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


# ── BaseFilter protocol ──────────────────────────────────────────────────────


class MetadataEnrichmentFilter:
    """Metadata enrichment filter for the v4 pipeline."""

    name = "metadata_enrichment"

    def apply(
        self,
        input_dir: Path,
        output_dir: Path,
        config_dict: dict[str, Any],
    ) -> FilterResult:
        client = config.get_tamu_client()
        scraped_meta = load_scraped_metadata()
        md_files = sorted(input_dir.glob("*.md"))

        result = FilterResult(input_count=len(md_files))
        for md_file in md_files:
            print(f"    Enriching {md_file.stem}...")
            try:
                enrichment = enrich_file(md_file, output_dir, client, scraped_meta)
                log = enrichment["_log"]
                result.modified_count += 1
                result.log_entries.append(
                    {
                        "file": md_file.stem,
                        "status": "ok",
                        "llm_metadata_status": log["llm_metadata_status"],
                        "llm_summary_status": log["llm_summary_status"],
                    }
                )
            except Exception as e:
                result.log_entries.append(
                    {
                        "file": md_file.stem,
                        "status": "error",
                        "error": str(e),
                    }
                )
                print(f"    ERROR: {md_file.stem}: {e}")

        return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich markdown syllabi with course metadata and summary.",
    )
    parser.add_argument("input", type=Path, help="Input dir or single .md file")
    parser.add_argument("output", type=Path, help="Output directory for JSON files")
    args = parser.parse_args()

    client = config.get_tamu_client()
    scraped_meta = load_scraped_metadata()

    if args.input.is_file():
        md_files = [args.input]
    elif args.input.is_dir():
        md_files = sorted(args.input.glob("*.md"))
    else:
        print(f"ERROR: {args.input} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    print(f"Enriching {len(md_files)} file(s)...")
    args.output.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        print(f"  {md_file.stem}...")
        try:
            result = enrich_file(md_file, args.output, client, scraped_meta)
            log = result["_log"]
            meta_status = log["llm_metadata_status"]
            summary_status = log["llm_summary_status"]
            hdrs = f"headers={log['header_count']} (pages={log['headers_with_page']})"
            print(f"    metadata={meta_status}  summary={summary_status}  {hdrs}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
