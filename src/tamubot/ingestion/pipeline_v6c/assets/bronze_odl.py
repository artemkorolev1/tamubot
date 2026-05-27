"""bronze_odl_markdown + bronze_odl_headers_sidecar — extract via opendataloader-pdf.

Java-backed extractor (veraPDF stack). Local-only mode; no hybrid OCR.

Standalone `odl_convert()` is callable outside Dagster so the bake-off
harness can reuse it without materializing assets — mirrors the
vlm_convert() shape from pipeline_v6.

veraPDF drops one char from repeated-letter runs on some Simple Syllabus PDFs
("College"→"Colege", "Meeting"→"Meting") and sometimes injects a space inside
the run ("TOOLS"→"T OLS"). `_repair_letter_drops` reverses both using
pypdfium2's extraction as ground-truth vocabulary — pypdfium2 extracts the
same PDFs cleanly.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from tamubot.ingestion.pipeline_v5.schemas import HeaderEntry, HeadersSidecar


@dataclass
class OdlConvertResult:
    markdown: str
    headers: list[HeaderEntry]
    timing_s: float
    page_count: int = 0
    raw_json: dict = field(default_factory=dict)
    repair_word_fixes: int = 0
    repair_merge_fixes: int = 0


_WORD_RE = re.compile(r"[A-Za-z]+")
_TOKEN_SPLIT_RE = re.compile(r"([A-Za-z]+)")
_SHORT_WORD_RE = re.compile(r"^[A-Za-z]{1,2}$")
_LONGER_WORD_RE = re.compile(r"^[A-Za-z]{2,}$")


def _collapse_dups(s: str) -> str:
    """Collapse consecutive identical chars to one. 'College' → 'Colege'."""
    if not s:
        return s
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def _build_vocab(text: str) -> dict[str, set[str]]:
    """collapse_dups(lower(word)) → set of original-case words seen in clean text."""
    vocab: dict[str, set[str]] = {}
    for word in _WORD_RE.findall(text):
        vocab.setdefault(_collapse_dups(word.lower()), set()).add(word)
    return vocab


def _match_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:].lower()
    return replacement.lower()


def _pypdfium_text(pdf_path: Path) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return "\n".join(p.get_textpage().get_text_range() for p in pdf)
    finally:
        pdf.close()


def _repair_letter_drops(md: str, vocab: dict[str, set[str]]) -> tuple[str, int, int]:
    """Two-pass: (1) word-level letter restoration, (2) conservative space-injection merge.

    Strictly-longer canonical required for both passes — if the vocab only has a
    candidate equal in length to the input, the input stays. This protects against
    trivial collisions while still catching the common SS bug shape (one letter
    dropped from a duplicated-letter run, optionally with a space substituted in).
    """
    n_word = 0
    n_merge = 0

    def fix_word(m: re.Match) -> str:
        nonlocal n_word
        w = m.group(0)
        key = _collapse_dups(w.lower())
        candidates = vocab.get(key, ())
        # Skip if w is already a real word — vocab evidence that this form exists.
        # Prevents false fixes like "to" → "too", "in" → "inn", "wel" → "well" (when "wel"
        # is a real abbreviation in the doc).
        wl = w.lower()
        if any(c.lower() == wl for c in candidates):
            return w
        longer = sorted((c for c in candidates if len(c) > len(w)), key=lambda c: (-len(c), c))
        if not longer:
            return w
        n_word += 1
        return _match_case(w, longer[0])

    md1 = _WORD_RE.sub(fix_word, md)

    # Pass 2: scan tokens explicitly. Don't use re.sub here — a non-fixable
    # candidate ("of co") would consume its chars and block a real fix on
    # the next token boundary ("co mon" → "common").
    tokens = _TOKEN_SPLIT_RE.split(md1)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        # Look for: word ([1-2 letters]) + " " separator + word ([2+ letters])
        if (
            i + 2 < len(tokens)
            and _SHORT_WORD_RE.match(tokens[i] or "")
            and tokens[i + 1] == " "
            and _LONGER_WORD_RE.match(tokens[i + 2] or "")
        ):
            a, b = tokens[i], tokens[i + 2]
            combined = (a + b).lower()
            combined_key = _collapse_dups(combined)
            candidates = vocab.get(combined_key, ())
            # Skip if the merged form (a+b) is already a real vocab entry — no fix needed.
            # Also skip when the merged candidate doesn't strictly add letters (no real gain).
            if not any(c.lower() == combined for c in candidates):
                longer = sorted(
                    (c for c in candidates if len(c) > len(a) + len(b)),
                    key=lambda c: (-len(c), c),
                )
                if longer:
                    out.append(_match_case(a + b, longer[0]))
                    i += 3
                    n_merge += 1
                    continue
        out.append(tokens[i])
        i += 1
    md2 = "".join(out)
    return md2, n_word, n_merge


_LABEL_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 &/,'\"().-]{1,55}:\s*$")


def _promote_label_headings(md: str) -> str:
    """Promote orphaned bold 'Label:' lines to H2 headings.

    Heuristic: a line is a label heading when ALL of these hold:
      - matches ^[A-Z][...]:$ (capitalized, ends with colon, <60 chars total)
      - has a blank line immediately above (or is first line)
      - has a blank line immediately below (or is last line)
      - is not already a markdown heading

    This catches the dominant v6c failure mode where opendataloader-pdf
    extracts the syllabus's bold section labels (Prerequisites, Learning
    Objectives, Attendance Policy, …) as plain paragraphs.

    We deliberately do NOT promote inline labels like 'Office: BLOC 417'
    — those have content after the colon on the same line and are filtered
    by the strict ``:\\s*$`` end-anchor.
    """
    lines = md.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _LABEL_HEADING_RE.match(stripped):
            out.append(line)
            continue
        # Skip if already a heading
        if stripped.startswith("#"):
            out.append(line)
            continue
        prev_blank = i == 0 or lines[i - 1].strip() == ""
        next_blank = i == len(lines) - 1 or lines[i + 1].strip() == ""
        if prev_blank and next_blank:
            out.append(f"## {stripped}")
        else:
            out.append(line)
    return "\n".join(out)


def _walk_headings(node, out: list[HeaderEntry]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "heading":
            text = (node.get("content") or "").strip()
            level = node.get("heading level")
            page = node.get("page number")
            if text and isinstance(level, int):
                out.append(HeaderEntry(text=text, level=level, page=page))
        for v in node.values():
            _walk_headings(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk_headings(item, out)


def odl_convert(
    pdf_path: Path,
    *,
    output_dir: Path | None = None,
    repair_text: bool = True,
) -> OdlConvertResult:
    """Run opendataloader-pdf on a single PDF and return parsed result.

    Writes intermediate `.md` and `.json` into `output_dir` (or a temp dir if
    None). Callers wanting persistence pass output_dir; the bake-off Dagster
    asset reads-and-rewrites into the v6c bronze paths.

    `repair_text=True` (default) runs a pypdfium2-vocabulary substitution pass
    over the markdown to undo veraPDF's letter-drop / space-injection corruption
    on Simple Syllabus PDFs. Also repairs headings extracted from the ODL JSON.
    """
    import tempfile

    import opendataloader_pdf

    work_dir = output_dir if output_dir is not None else Path(tempfile.mkdtemp(prefix="odl_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    opendataloader_pdf.convert(
        input_path=[str(pdf_path)],
        output_dir=str(work_dir),
        format="markdown,json",
    )
    elapsed = time.time() - t0

    md_path = work_dir / f"{pdf_path.stem}.md"
    json_path = work_dir / f"{pdf_path.stem}.json"
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    raw = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}

    n_word = n_merge = 0
    if repair_text and md:
        vocab = _build_vocab(_pypdfium_text(pdf_path))
        md, n_word, n_merge = _repair_letter_drops(md, vocab)
        # Persist the repaired markdown so downstream readers see corrected text.
        md_path.write_text(md, encoding="utf-8")
    else:
        vocab = {}

    if md:
        md = _promote_label_headings(md)
        md_path.write_text(md, encoding="utf-8")

    headers: list[HeaderEntry] = []
    _walk_headings(raw, headers)
    if vocab:
        for h in headers:
            h.text, _, _ = _repair_letter_drops(h.text, vocab)

    return OdlConvertResult(
        markdown=md,
        headers=headers,
        timing_s=elapsed,
        page_count=int(raw.get("number of pages") or 0),
        raw_json=raw,
        repair_word_fixes=n_word,
        repair_merge_fixes=n_merge,
    )


def _bronze_compute(context):
    """Dagster compute: ODL-convert one stem's raw PDF, write markdown + sidecar + meta."""
    from dagster import AssetKey, MaterializeResult, MetadataValue

    from tamubot.ingestion.pipeline_v5.util import dept_from_stem, hash_file
    from tamubot.ingestion.pipeline_v6c import paths as v6c_paths

    stem = context.partition_key
    dept = dept_from_stem(stem)
    pdf = v6c_paths.raw_path(stem)
    if not pdf.exists():
        raise FileNotFoundError(f"raw PDF missing for stem {stem!r}: {pdf}")

    bronze_dir = v6c_paths.bronze_md_path(stem).parent
    bronze_dir.mkdir(parents=True, exist_ok=True)

    result = odl_convert(pdf, output_dir=bronze_dir)

    md_out = v6c_paths.bronze_md_path(stem)
    md_out.write_text(result.markdown, encoding="utf-8")

    sidecar_out = v6c_paths.bronze_sidecar_path(stem)
    sidecar = HeadersSidecar(headers=result.headers)
    sidecar_out.write_text(sidecar.model_dump_json(indent=2), encoding="utf-8")

    odl_json_out = v6c_paths.bronze_odl_json_path(stem)
    odl_json_out.write_text(json.dumps(result.raw_json, indent=2, ensure_ascii=False), encoding="utf-8")

    meta_out = v6c_paths.bronze_meta_path(stem)
    meta_out.write_text(
        json.dumps(
            {
                "stem": stem,
                "dept": dept,
                "converter": "opendataloader-pdf",
                "timing_s": round(result.timing_s, 2),
                "page_count": result.page_count,
                "header_count": len(result.headers),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_sha = hash_file(md_out)
    token_estimate = len(result.markdown) // 4

    md_meta = {
        "stem": stem,
        "dept": dept,
        "output_path": MetadataValue.path(str(md_out)),
        "converter": "opendataloader-pdf",
        "timing_s": round(result.timing_s, 2),
        "page_count": result.page_count,
        "header_count": len(result.headers),
        "token_count_estimate": token_estimate,
        "sha256": md_sha,
        "preview": MetadataValue.md(result.markdown[:1000]),
    }
    sidecar_meta = {
        "stem": stem,
        "dept": dept,
        "output_path": MetadataValue.path(str(sidecar_out)),
        "header_entries": len(result.headers),
    }

    yield MaterializeResult(asset_key=AssetKey("v6c_bronze_markdown"), metadata=md_meta)
    yield MaterializeResult(asset_key=AssetKey("v6c_bronze_headers_sidecar"), metadata=sidecar_meta)
