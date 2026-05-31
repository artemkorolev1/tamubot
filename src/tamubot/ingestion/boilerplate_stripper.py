"""
tamubot.ingestion.boilerplate_stripper

Deterministic pre-stripper for TAMU syllabus PDFs.
Removes known institutional boilerplate sections BEFORE the LLM sees the text,
based on section-header matching against a frequency-derived registry.

Usage (module):
    from tamubot.ingestion.boilerplate_stripper import strip_pdf
    cleaned_text, strip_log = strip_pdf(pdf_path)

Batch validation (CLI — run from repo root):
    python -m tamubot.ingestion.boilerplate_stripper
    python -m tamubot.ingestion.boilerplate_stripper --input-dirs tamu_data/raw/simple_syllabus_20260305
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

# ── Boilerplate registry ──────────────────────────────────────────────────────
# Derived from header frequency analysis across 317 CSCE/ISEN syllabi.
# Each entry: lowercase header string → category label.
# A PDF section whose header (lowercased, stripped) matches → stripped.

BOILERPLATE_REGISTRY: dict[str, list[str]] = {
    # Document / department masthead lines (pure structural noise)
    "DEPT_HEADER": [
        "college of engineering",
        "college of arts and sciences",  # Simple Syllabus banner above STAT/other A&S depts
        "computer science and engineering",
        "industrial and systems engineering",
        "statistics",  # H2 dept-name banner (STAT corpus); validator-flagged across files
        "college of engineering industrial and systems engineering",
        "college of engineering computer science and engineering",
        "texas a&m at galveston",
    ],
    # Standard TAMU-wide legal/policy sections (100% verbatim across all files)
    "TAMU_POLICY": [
        "university policies",
        "academic integrity statement and policy",
        "academic integrity statements and policy",  # plural variant seen in some PDFs
        "notice of nondiscrimination",
        "non-discrimination policy",  # alternate phrasing (3+ files)
        "civil rights, free speech, and title ix policies",
        "americans with disabilities act (ada) policy",
        "pregnancy accommodations",
        "statement on mental health and wellness",
        # FERPA header — wraps across lines in different ways across PDFs
        "statement on the family educational rights and privacy act (ferpa)",
        "statement on the family educational rights and privacy act",  # no trailing (ferpa)
        "statement on the family educational rights and",
        "privacy act (ferpa)",
        "(ferpa)",  # orphaned second wrap line
        "free speech and civil discourse",
        "additional university policies for galveston campus",
        "additional university policies for the galveston campus",  # "the" variant
        "college and department policies",
        "university attendance policy",
        "makeup work policy",
        "course evaluation",  # HelioCampus eval boilerplate
        "ai statement",  # Institutional AI-use policy (appears verbatim across syllabi)
        "title ix and statement on limits to confidentiality",  # Title IX boilerplate (34/100 files)
        "texas a&m at college station",  # Location header inside boilerplate blocks
        "attendance policy",  # Generic attendance policy (distinct from "university attendance policy")
        "americans with disabilities act (ada) policy statement",  # Variant with "Statement" suffix (HP files)
        "student observances for religious holy days",  # Religious observance boilerplate
        "mental health and well-being",  # Variant of "Statement on Mental Health and Wellness"
        "mental health and wellness",  # Short variant without "Statement on" prefix
        "additional departmental, college, and university policies",  # Catch-all policy pointer
        "campus-specific policies",  # Campus-specific policy block
        "optional syllabus statements",  # HP "Optional Syllabus Statements" wrapper containing nested boilerplate (Free Speech, AI, etc.) as plain text
        "recording statement",  # TAMU canned recording-permissions paragraph (validator-flagged, 2+ files)
        "optional course information items",  # HP wrapper for optional template fields, 3 HP files
        "course material policy",  # Variant of "course material and copyright" (verbatim TAMU template)
        "course auditing",  # TAMU registrar audit policy (CSCE 605)
        # NOTE: "attendance certification" intentionally NOT stripped — section contains
        # course-specific deadline date (financial-aid-relevant) buried in otherwise-template text.
        "incomplete grade policy",  # TAMU I-grade policy (CSCE 605)
        "professionalism and respect",  # Generic respect/conduct boilerplate (CSCE 605)
    ],
    # WRAPPER: TAMU-template section *containers* whose header is boilerplate but
    # which often nest course-specific H3 subsections (e.g. "## College and
    # Department Policies > ### Homework Policy"). Strip the header line only;
    # children are kept so nested course content survives.
    "WRAPPER": [
        "college and department policies",  # often wraps Homework Policy, Exam Policy, etc.
        "additional departmental, college, and university policies",
        "additional university policies for galveston campus",
        "additional university policies for the galveston campus",
        "campus-specific policies",
        "optional syllabus statements",
    ],
    # IT helpdesk / tool support blocks (zero course-specific content)
    "TECH_SUPPORT": [
        "technology support",
        "canvas lms technical support",
        "technology services (it) - main campus",
        "perusall technical support",
        "peerceptiv technical support",
        "github technical support",
        "peer teacher central",
        "university writing center",
        "learning resources",
        "gradescope technical support",  # Gradescope helpdesk info
    ],
    # Additional institutional template sections confirmed across files
    "INSTITUTIONAL": [
        "course copyright",
        "plagiarism and copyright",  # variant seen in 3+ files
        "respect for all",
        "absence documentation",
        "accessibility statement",  # Disability accommodation boilerplate
        "late days table",  # Template table repeated verbatim
        "course material and copyright",  # Variant of "course copyright" (ISEN 689)
        "plagiarism policy",  # Standalone plagiarism policy section (HP files)
        "plagiarism",  # Bare "Plagiarism" header (STAT corpus, no "Policy" suffix)
        "copyright notice",  # Bare "Copyright notice" header (STAT corpus)
    ],
}

# Flat lookup: lowercase header text → category label
_HEADER_MAP: dict[str, str] = {h: cat for cat, headers in BOILERPLATE_REGISTRY.items() for h in headers}

ALL_BOILERPLATE_HEADERS: frozenset[str] = frozenset(_HEADER_MAP)

# Known TAMU boilerplate that appears at body font size (not detectable by font
# metadata).  ONLY include long, specific phrases where false-positive risk is
# negligible.  Do NOT add short generic words like "Reporting" or "Fabrication".
BODY_BOILERPLATE_HEADERS: list[str] = [
    "aggie honor code",
    "preferred name and preferred gender pronouns",
    "submission of work policy",
    "regrading policy",
    "course copyright statement",
    "course copyright",
    "course material and copyright",
]


# Sentence-level boilerplate that bleeds into course-specific sections.
# Be conservative: only strip phrases that are verbatim TAMU template text and
# that contribute no course-specific info. Avoid stripping phrases that just
# *reference* Student Rule 7 alongside course-specific guidance.
SENTENCE_STRIPS: list[str] = [
    r"The university views class attendance and participation as an individual student responsibility[^.]*\.",
    r"Please refer to Student Rule 7 in its entirety for information about excused absences[^.]*\.",
    r"Please refer to Student Rule 7 in its entirety for information about makeup work[^.]*\.",
]

_SENTENCE_STRIP_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in SENTENCE_STRIPS]

# Howdy Portal repeats this 3-4 line page-header block on every page (lands in
# extracted markdown as a table-leakage fragment). Match across line breaks.
_HP_PAGE_HEADER_RE = re.compile(
    r"^\|?\s*TEXAS A&M\s*\n\s*UNIVERSITY\.?\s*\n(?:\s*\n)?\s*Course Syllabus\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_template_sentences(text: str) -> tuple[str, int]:
    """Remove stray TAMU-template sentences anywhere they appear in body text.

    Returns ``(new_text, sentence_strip_count)``.
    """
    total = 0
    for pat in _SENTENCE_STRIP_PATTERNS:
        text, n = pat.subn("", text)
        total += n
    text, n = _HP_PAGE_HEADER_RE.subn("", text)
    total += n
    # Collapse runs of 3+ blank lines that the strip can leave behind.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Tidy doubled spaces left by mid-paragraph strips.
    text = re.sub(r"  +", " ", text)
    return text, total


_LEADING_NUMBER_RE = re.compile(r"^[\d.]+\s+")


def classify_header(text: str) -> str | None:
    """Return boilerplate category label if text matches registry, else None."""
    # Collapse internal whitespace (tabs/multiple spaces) so headers like
    # "University\tPolicies" match the registry key "university policies".
    norm = re.sub(r"\s+", " ", text.lower().strip()).rstrip(":")
    result = _HEADER_MAP.get(norm)
    if result:
        return result
    # Strip leading section numbers (e.g. "6.1 ATTENDANCE POLICY" -> "attendance policy")
    stripped = _LEADING_NUMBER_RE.sub("", norm)
    if stripped != norm:
        return _HEADER_MAP.get(stripped)
    return None


def strip_markdown_boilerplate(md_text: str) -> tuple[str, list[dict]]:
    """Strip boilerplate sections from Markdown text (e.g. from pymupdf4llm).

    Detects headers by '#' prefix instead of font metadata.

    Returns:
        filtered_md  — Markdown with boilerplate sections removed
        strip_log    — list of {header, type, chars, header_level} dicts
    """

    sections: list[tuple[str | None, int, list[str]]] = []
    cur_header: str | None = None
    cur_level: int = 0
    cur_lines: list[str] = []

    for line in md_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            sections.append((cur_header, cur_level, cur_lines))
            cur_header = m.group(2).strip()
            cur_level = len(m.group(1))
            cur_lines = []
        else:
            cur_lines.append(line)
    sections.append((cur_header, cur_level, cur_lines))

    kept: list[str] = []
    strip_log: list[dict] = []

    for header, level, content_lines in sections:
        bp_type = classify_header(header) if header else None
        content = "\n".join(content_lines).strip()

        if bp_type:
            strip_log.append(
                {
                    "header": header,
                    "type": bp_type,
                    "chars": len(header or "") + len(content),
                    "header_level": level,
                    "content": content,
                }
            )
        else:
            if header:
                kept.append(f"{'#' * level} {header}")
            if content:
                kept.append(content)

    return "\n\n".join(kept), strip_log


def strip_body_level_boilerplate(text: str) -> tuple[str, list[dict]]:
    """Strip boilerplate sections whose headers appear at body font size.

    These headers have no font-size prefix in annotated text and therefore
    cannot be caught by strip_font_annotated_boilerplate.  We match only
    long, unambiguous phrases listed in BODY_BOILERPLATE_HEADERS.

    A block starts when a line (stripped, lowercased) exactly matches an entry.
    It ends at the next font-annotated header line ([Xpt...] prefix) or EOF.

    Returns:
        filtered_text  — text with body-level boilerplate blocks removed
        strip_log      — list of {header, type, chars, content} dicts
    """
    import re as _re

    FONT_PREFIX = _re.compile(r"^\[(\d+\.?\d*)pt")
    _BP_SET = {h.lower() for h in BODY_BOILERPLATE_HEADERS}

    lines = text.splitlines()
    out_lines: list[str] = []
    strip_log: list[dict] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_lower = line.strip().lower().rstrip(":")

        if stripped_lower in _BP_SET:
            # Collect block until next font-annotated header or EOF
            header_text = line.strip()
            block: list[str] = []
            j = i + 1
            while j < len(lines):
                if FONT_PREFIX.match(lines[j]):
                    break
                block.append(lines[j])
                j += 1
            content = "\n".join(block).strip()
            strip_log.append(
                {
                    "header": header_text,
                    "type": "BODY_BOILERPLATE",
                    "chars": len(header_text) + len(content),
                    "content": content,
                }
            )
            i = j  # skip header + body block
        else:
            out_lines.append(line)
            i += 1

    return "\n".join(out_lines), strip_log


# ── Core strip function ───────────────────────────────────────────────────────


def strip_pdf(pdf_path: Path) -> tuple[str, list[dict]]:
    """
    Extract text from a PDF and remove known boilerplate sections.

    Strategy:
      1. Extract all text lines with font metadata via PyMuPDF.
      2. Detect section headers: bold flag OR font size > body size + 1.5pt.
      3. Group lines into (header, content) sections.
      4. Strip sections whose header matches BOILERPLATE_REGISTRY.
      5. Return joined non-boilerplate text + a strip log.

    Returns:
        cleaned_text  — text with boilerplate removed, ready for LLM
        strip_log     — list of {header, type, chars, page_start}
    """
    doc = fitz.open(str(pdf_path))

    raw_lines: list[dict] = []
    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                raw_lines.append(
                    {
                        "text": text,
                        "size": max(s.get("size", 0) for s in spans),
                        "bold": any(s.get("flags", 0) & 16 for s in spans),
                        "page": page_num,
                    }
                )
    doc.close()

    if not raw_lines:
        return "", []

    # Body font = most common rounded font size
    body_size: int = Counter(round(ln["size"]) for ln in raw_lines).most_common(1)[0][0]

    # Segment into sections
    sections: list[tuple[str | None, int, list[str]]] = []
    cur_header: str | None = None
    cur_page: int = 1
    cur_lines: list[str] = []

    for ln in raw_lines:
        is_header = (ln["bold"] or ln["size"] >= body_size + 0.5) and 3 <= len(ln["text"]) <= 120
        if is_header:
            sections.append((cur_header, cur_page, cur_lines))
            cur_header = ln["text"]
            cur_page = ln["page"]
            cur_lines = []
        else:
            cur_lines.append(ln["text"])
    sections.append((cur_header, cur_page, cur_lines))

    # Filter: strip boilerplate, keep the rest
    kept: list[str] = []
    strip_log: list[dict] = []

    for header, page, lines in sections:
        bp_type = classify_header(header) if header else None
        content = "\n".join(lines)

        if bp_type:
            strip_log.append(
                {
                    "header": header,
                    "type": bp_type,
                    "chars": len(header or "") + len(content),
                    "page_start": page,
                    "content": content,
                }
            )
        else:
            if header:
                kept.append(header)
            if content:
                kept.append(content)

    return "\n\n".join(kept), strip_log


def pdf_to_annotated_markdown(pdf_path: Path) -> tuple[str, dict]:
    """Extract PDF text with inline font annotations for every non-body line.

    Body text (most common font size, non-bold) is emitted as plain text.
    All other lines are prefixed with their font metadata so a downstream LLM
    can identify structural headers from both typography AND content:

        [16.0pt bold] College of Engineering
        [13.0pt bold] Course Description
        This is body text at 11.0pt — no prefix.
        [11.0pt bold] Note:

    body_size is detected at 0.5pt resolution to avoid integer-rounding errors
    (e.g. a PDF with 10.5pt body text must not collapse to 10pt).

    Returns:
        annotated_text — font-annotated plain text (no Markdown headers)
        stats          — {page_count, body_size, annotated_lines, plain_lines,
                         char_count_in, char_count_out}
    """

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count

    raw_lines: list[dict] = []
    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                raw_lines.append(
                    {
                        "text": text,
                        "size": max(s.get("size", 0) for s in spans),
                        "bold": any(s.get("flags", 0) & 16 for s in spans),
                        "page": page_num,
                    }
                )
    doc.close()

    char_count_in = sum(len(ln["text"]) for ln in raw_lines)

    if not raw_lines:
        return "", {
            "page_count": page_count,
            "body_size": 0,
            "annotated_lines": 0,
            "plain_lines": 0,
            "char_count_in": 0,
            "char_count_out": 0,
        }

    # 0.5pt resolution prevents collapsing nearby sizes (e.g. 10.5 → 10).
    # Weight by character count, not line count: short table-cell lines (e.g. 10pt)
    # would otherwise beat out longer body paragraphs and corrupt the body-size
    # estimate — causing the stripper to treat all body-font content as headers.
    char_weights: dict[float, int] = {}
    for ln in raw_lines:
        bucket = round(ln["size"] * 2) / 2
        char_weights[bucket] = char_weights.get(bucket, 0) + len(ln["text"])
    body_size: float = max(char_weights, key=char_weights.__getitem__)

    annotated_count = plain_count = 0
    out_lines: list[str] = []

    for ln in raw_lines:
        text = ln["text"]
        # Round size to 1 decimal for clean display
        size = round(ln["size"], 1)
        size_bucket = round(ln["size"] * 2) / 2  # same resolution as body detection
        bold = ln["bold"]

        is_body = (size_bucket == body_size) and not bold

        if is_body:
            out_lines.append(text)
            plain_count += 1
        else:
            bold_tag = " bold" if bold else ""
            out_lines.append(f"[{size}pt{bold_tag}] {text}")
            annotated_count += 1

    # Embed body font size as a parseable header comment for downstream functions
    header = f"<!-- body_font:{body_size}pt -->"
    annotated_text = header + "\n" + "\n".join(out_lines)

    return annotated_text, {
        "page_count": page_count,
        "body_size": body_size,
        "annotated_lines": annotated_count,
        "plain_lines": plain_count,
        "char_count_in": char_count_in,
        "char_count_out": len(annotated_text),
    }


def strip_font_annotated_boilerplate(text: str) -> tuple[str, list[dict]]:
    """Strip boilerplate from font-annotated text produced by pdf_to_annotated_markdown.

    Annotated lines have the form: [Xpt bold] Header Text  or  [Xpt] Header Text
    Plain lines (no prefix) are body text. Each annotated line starts a new
    section; its text content is matched against BOILERPLATE_REGISTRY.

    Returns:
        filtered_text — annotated text with boilerplate sections removed
        strip_log     — list of {header, type, chars, font_size, bold, content}
    """
    import re as _re

    FONT_PREFIX = _re.compile(r"^\[(\d+\.?\d*)pt( bold)?\] (.+)$")
    BODY_HEADER = _re.compile(r"^<!-- body_font:(\d+\.?\d*)pt -->$")

    # Preserve the body_font comment line, skip it for section parsing
    all_lines = text.splitlines()
    body_comment = ""
    body_size = 0.0
    parse_lines = all_lines
    if all_lines and (bm := BODY_HEADER.match(all_lines[0])):
        body_comment = all_lines[0]
        body_size = float(bm.group(1))
        parse_lines = all_lines[1:]

    sections: list[tuple[str | None, dict, list[str]]] = []
    cur_header: str | None = None
    cur_meta: dict = {}
    cur_lines: list[str] = []

    for line in parse_lines:
        m = FONT_PREFIX.match(line)
        if m:
            size = float(m.group(1))
            bold = m.group(2) is not None
            text = m.group(3).strip()
            # Start a new section if:
            #   (a) line is clearly above body size (normal structural header), OR
            #   (b) its text exactly matches the boilerplate registry — handles PDFs
            #       where sub-headers sit BELOW body font size (e.g. 10.5pt bold in a
            #       13pt body doc) but are still known institutional boilerplate.
            is_structural = body_size == 0.0 or size >= body_size + 1.0
            is_known_bp = classify_header(text) is not None
            if is_structural or is_known_bp:
                sections.append((cur_header, cur_meta, cur_lines))
                cur_header = text
                cur_meta = {"font_size": size, "bold": bold}
                cur_lines = []
            else:
                cur_lines.append(line)
        else:
            cur_lines.append(line)
    sections.append((cur_header, cur_meta, cur_lines))

    kept: list[str] = []
    strip_log: list[dict] = []

    for header, meta, content_lines in sections:
        bp_type = classify_header(header) if header else None
        content = "\n".join(content_lines).strip()
        font_size = meta.get("font_size", 0.0)
        bold = meta.get("bold", False)

        if bp_type:
            strip_log.append(
                {
                    "header": header,
                    "type": bp_type,
                    "chars": len(header or "") + len(content),
                    "font_size": font_size,
                    "bold": bold,
                    "content": content,
                }
            )
        else:
            if header:
                bold_tag = " bold" if bold else ""
                kept.append(f"[{font_size}pt{bold_tag}] {header}")
            if content:
                kept.append(content)

    # Re-attach the body_font comment so annotated_to_clean_markdown can use it
    if body_comment:
        kept.insert(0, body_comment)

    first_pass_text = "\n".join(kept)

    # Second pass: strip body-level boilerplate (headers with no font prefix)
    second_pass_text, body_strip_log = strip_body_level_boilerplate(first_pass_text)
    strip_log.extend(body_strip_log)

    return second_pass_text, strip_log


def annotated_to_clean_markdown(annotated_text: str) -> str:
    """Convert font-annotated text to clean Markdown using adaptive header detection.

    Strategy:
      1. Parse <!-- body_font:Xpt --> header to get the document body font size.
      2. Only consider annotated lines with size > body_size as header candidates
         (filters out sub-body text like footnotes at smaller sizes).
      3. Among candidates, discard configs with count > 60 — they're over-represented
         body-variant text, not structural headers (e.g. 188 lines at 11pt in ISEN_620).
      4. Sort remaining viable configs by size desc, bold as tie-breaker.
         Take at most 2 → mapped to ## (top) and ### (second).
      5. Top config gets promoted to # only if it appears <= 3 times in the doc
         (rare → likely a document title, not a repeated section header).
      6. Lines longer than 120 chars are always plain text regardless of font.
    """
    import re as _re
    from collections import Counter as _Counter

    FONT_PREFIX = _re.compile(r"^\[(\d+\.?\d*)pt( bold)?\] (.+)$")
    BODY_HEADER = _re.compile(r"^<!-- body_font:(\d+\.?\d*)pt -->$")

    # ── Parse body font size from first line ──────────────────────────────────
    lines = annotated_text.splitlines()
    body_size = 0.0
    start_idx = 0
    if lines:
        m = BODY_HEADER.match(lines[0])
        if m:
            body_size = float(m.group(1))
            start_idx = 1  # skip header comment line

    # ── Pass 1: count each (size, bold) config above body size ───────────────
    config_counts: _Counter = _Counter()
    for line in lines[start_idx:]:
        m = FONT_PREFIX.match(line)
        if m:
            size = float(m.group(1))
            bold = m.group(2) is not None
            if size > body_size:  # only candidates above body size
                config_counts[(size, bold)] += 1

    # ── Assign header levels ──────────────────────────────────────────────────
    MAX_HEADER_COUNT = 60  # configs appearing more often are body-variant text

    viable = sorted(
        [(cfg, cnt) for cfg, cnt in config_counts.items() if cnt <= MAX_HEADER_COUNT],
        key=lambda x: (x[0][0], x[0][1]),  # sort by (size, bold) descending
        reverse=True,
    )

    level_map: dict[tuple[float, bool], int] = {}
    for i, (cfg, _cnt) in enumerate(viable[:2]):  # at most ## and ###
        level_map[cfg] = i + 2  # 2=##, 3=###

    # Top config → # only if it's a rare title (appears very few times)
    if viable and viable[0][1] <= 3:
        level_map[viable[0][0]] = 1  # promote to #

    # ── Pass 2: emit clean markdown ───────────────────────────────────────────
    out_lines: list[str] = []
    for line in lines[start_idx:]:
        m = FONT_PREFIX.match(line)
        if m:
            size = float(m.group(1))
            bold = m.group(2) is not None
            text = m.group(3)
            level = level_map.get((size, bold))
            if level and len(text) <= 120:
                out_lines.append(f"{'#' * level} {text}")
            else:
                out_lines.append(text)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def render_stripped_markdown(stem: str, strip_log: list[dict]) -> str:
    """Format stripped sections as readable Markdown for human review.

    Handles strip_log entries from both strip_markdown_boilerplate
    (has header_level) and strip_font_annotated_boilerplate (has font_size/bold).
    """
    total_chars = sum(e.get("chars", 0) for e in strip_log)
    lines: list[str] = [
        f"# Stripped Content: {stem}",
        f"_{len(strip_log)} sections stripped, {total_chars} chars total_",
        "",
        "---",
    ]
    for entry in strip_log:
        header = entry.get("header", "(no header)")
        bp_type = entry.get("type", "?")
        chars = entry.get("chars", 0)
        content = entry.get("content", "")

        # Font-annotated log entry
        if "font_size" in entry:
            font_size = entry["font_size"]
            bold = entry.get("bold", False)
            bold_tag = " bold" if bold else ""
            meta_str = f"font: {font_size}pt{bold_tag} | chars: {chars}"
        else:
            level = entry.get("header_level", 2)
            meta_str = f"chars: {chars} | header_level: {level}"

        lines.append("")
        lines.append(f"## [{bp_type}] {header}")
        lines.append(f"_{meta_str}_")
        lines.append("")
        if content:
            lines.append(content)

    return "\n".join(lines)


# ── Batch validation ──────────────────────────────────────────────────────────


def batch_validate(pdf_paths: list[Path], log_dir: Path) -> dict:
    """
    Run strip_pdf on every PDF, write per-file JSON logs, return a summary.

    Per-file log: tamu_data/logs/boilerplate_strip/<stem>.json
    Summary:      tamu_data/logs/boilerplate_strip/_summary.json
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    header_counter: Counter = Counter()
    type_counter: Counter = Counter()

    for i, pdf in enumerate(pdf_paths, 1):
        print(f"[{i}/{len(pdf_paths)}] {pdf.name} ...", end=" ", flush=True)
        try:
            cleaned, strip_log = strip_pdf(pdf)

            # Original char count (re-open for raw text)
            doc = fitz.open(str(pdf))
            original = "\n".join(p.get_text() for p in doc)
            doc.close()

            chars_orig = len(original)
            chars_rem = len(cleaned)
            chars_stripped = sum(e["chars"] for e in strip_log)
            reduction = round(chars_stripped / max(chars_orig, 1) * 100, 1)

            result: dict = {
                "file": pdf.name,
                "chars_original": chars_orig,
                "chars_remaining": chars_rem,
                "chars_stripped": chars_stripped,
                "reduction_pct": reduction,
                "sections_stripped": strip_log,
            }
            print(f"-{reduction}%  ({len(strip_log)} sections stripped)")

            for entry in strip_log:
                header_counter[entry["header"].lower()] += 1
                type_counter[entry["type"]] += 1

        except Exception as e:
            result = {"file": pdf.name, "error": str(e)}
            print(f"ERROR: {e}")

        results.append(result)
        with open(log_dir / f"{pdf.stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    ok = [r for r in results if "error" not in r]
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_files": len(results),
        "files_ok": len(ok),
        "avg_reduction_pct": round(sum(r["reduction_pct"] for r in ok) / max(len(ok), 1), 1),
        "stripped_by_type": dict(type_counter.most_common()),
        "stripped_header_frequency": dict(header_counter.most_common()),
    }
    with open(log_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description="Batch boilerplate strip — dry-run validation across all PDFs")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            "tamu_data/raw/simple_syllabus_20260305",
            "tamu_data/raw/simple_syllabus_20260304",
        ],
    )
    parser.add_argument("--log-dir", default="tamu_data/logs/boilerplate_strip")
    args = parser.parse_args()

    # Collect + deduplicate by stem (later dir in list wins)
    seen: dict[str, Path] = {}
    for d in reversed(args.input_dirs):
        for p in sorted(Path(d).glob("*.pdf")):
            seen[p.stem] = p
    pdfs = sorted(seen.values(), key=lambda p: p.stem)

    print(f"PDFs:    {len(pdfs)}")
    print(f"Log dir: {args.log_dir}\n")

    summary = batch_validate(pdfs, Path(args.log_dir))

    total = summary["total_files"]
    print(f"\n{'=' * 65}")
    print(f"Files processed : {total}  ({summary['files_ok']} ok)")
    print(f"Avg reduction   : {summary['avg_reduction_pct']}%")

    print("\nStripped sections by type:")
    for bp_type, count in summary["stripped_by_type"].items():
        print(f"  {bp_type:<20} {count:>5} occurrences")

    print("\nStripped header frequency (top 40):")
    print(f"  {'Header':<55} {'Files':>6}  {'%':>6}")
    print(f"  {'-' * 55}  {'------':>6}  {'------':>6}")
    for header, count in list(summary["stripped_header_frequency"].items())[:40]:
        print(f"  {header:<55} {count:>6}  {count / total * 100:>5.1f}%")

    print(f"\nPer-file logs : {args.log_dir}/")
    print(f"Summary       : {args.log_dir}/_summary.json")
