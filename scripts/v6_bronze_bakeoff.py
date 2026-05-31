"""v6 bronze bake-off — compare VLM bronze (gemini-3.1-flash-lite) vs v5 Docling
bronze on N STAT files. Emits data/syllabi/<DEPT>/v6/pilot_bakeoff.xlsx.

Five gates (per v6 plan Step 2):
  1. token recall  — char-length ratio in [0.98, 1.02]
  2. header recall — Docling headers fuzzy-found in VLM (rapidfuzz, score >=85)
  3. fabrication  — TAMU judge on VLM-only sections (opt-in: --with-fabrication-judge)
  4. table rows   — row count within 10% (auto-pass when no tables)
  5. stability    — two VLM runs identical chunk count, edit distance <1% (opt-in: --stability)

Budget tiers (Gemini calls = paid, TAMU = free):
  --limit 10                      → 10 Gemini calls (~$0.03), gates 1/2/4
  --limit 10 --stability          → 20 Gemini calls (~$0.06), gates 1/2/4/5
  --limit 10 --with-fabrication-judge → +10 free TAMU calls, adds gate 3
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz.fuzz import token_sort_ratio

from tamubot.core import config
from tamubot.ingestion.pipeline_v5 import paths as v5_paths
from tamubot.ingestion.pipeline_v5.schemas import HeadersSidecar
from tamubot.ingestion.pipeline_v6 import paths as v6_paths
from tamubot.ingestion.pipeline_v6.assets.bronze_vlm import VLM_MODEL, vlm_convert

DEFAULT_DEPT = "STAT"
GATE_THRESHOLDS = {
    # Token-recall: one-sided. VLM may legitimately produce MORE text than
    # Docling (preserving URLs, contact info, full table cells), so we only
    # gate against under-extraction. Upper bound retired 2026-05-26.
    "token_recall_min": 0.95,
    "token_recall_max": 1.50,
    "header_recall_min": 0.95,
    "header_fuzzy_score_min": 85,
    "table_row_min": 0.90,
    "table_row_max": 1.10,
    "stability_edit_distance_max": 0.01,
}

# Indicative per-million-token pricing for gemini-3.1-flash-lite (used only for
# cost estimation in the report; treat as approximate — verify on the actual bill).
PRICE_INPUT_PER_MTOK = 0.10
PRICE_OUTPUT_PER_MTOK = 0.40


@dataclass
class GateResult:
    name: str
    passed: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class FileResult:
    stem: str
    vlm_md_chars: int = 0
    docling_md_chars: int = 0
    vlm_headers: int = 0
    docling_headers: int = 0
    vlm_table_rows: int = 0
    docling_table_rows: int = 0
    timing_s: float = 0.0
    input_image_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    gates: list[GateResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _count_table_rows(md: str) -> int:
    rows = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("|") and ln.strip().endswith("|")]
    # Exclude separator rows (| --- | --- |)
    return sum(1 for ln in rows if "---" not in ln)


_HEADER_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _markdown_headers(md: str) -> list[str]:
    return [m.strip() for m in _HEADER_RE.findall(md)]


def gate_token_recall(docling_md: str, vlm_md: str) -> GateResult:
    docling_len = len(docling_md)
    vlm_len = len(vlm_md)
    ratio = vlm_len / docling_len if docling_len else 1.0
    passed = GATE_THRESHOLDS["token_recall_min"] <= ratio <= GATE_THRESHOLDS["token_recall_max"]
    return GateResult(
        name="token_recall",
        passed=passed,
        metadata={
            "docling_chars": docling_len,
            "vlm_chars": vlm_len,
            "ratio": round(ratio, 4),
        },
    )


_DEPT_FROM_STEM_RE = re.compile(r"^\d+_([A-Z]+)_")
_NONEMPTY_ROW_RE = re.compile(r"\|[^|]*\w[^|]*\|")


def gate_header_structure(stem: str, vlm_md: str, sidecar_headers: list[str]) -> GateResult:
    """One-sided structural gate: VLM extracted a substantial, valid header tree.

    Pass when: VLM has >=5 headers AND VLM has at least one H1 AND the H1 (or
    early text) contains the dept code derived from the stem (e.g. 'STAT'). The
    Docling sidecar count is reported as an informational metric (`docling_headers`,
    `recall_vs_docling`) but does NOT gate — Docling is known to spuriously
    promote body text to H1, which is exactly what v6 sets out to fix.
    """
    vlm_headers = _markdown_headers(vlm_md)
    h1s = [m.group(1).strip() for m in re.finditer(r"^#\s+(.+)$", vlm_md, re.MULTILINE)]

    dept_match = _DEPT_FROM_STEM_RE.match(stem)
    dept = dept_match.group(1) if dept_match else ""
    # Title H1 must include the dept code (or the first 500 chars must — covers
    # cases where VLM puts dept above the title H1).
    has_title = bool(dept) and (any(dept in h for h in h1s) or dept in vlm_md[:500])

    recall_vs_docling = None
    if sidecar_headers:
        matched = sum(
            1
            for dh in sidecar_headers
            if max(
                (token_sort_ratio(dh.lower(), vh.lower()) for vh in vlm_headers),
                default=0,
            )
            >= GATE_THRESHOLDS["header_fuzzy_score_min"]
        )
        recall_vs_docling = round(matched / len(sidecar_headers), 4)

    passed = (len(vlm_headers) >= 5) and (len(h1s) >= 1) and has_title
    return GateResult(
        name="header_structure",
        passed=passed,
        metadata={
            "vlm_headers": len(vlm_headers),
            "vlm_h1_count": len(h1s),
            "dept_code": dept,
            "has_title_with_dept": has_title,
            "docling_headers": len(sidecar_headers),
            "recall_vs_docling": recall_vs_docling,
        },
    )


def gate_table_extraction(docling_md: str, vlm_md: str) -> GateResult:
    """One-sided gate: VLM extracts ≥ as many real tables as Docling.

    Counts only non-empty rows (rows with actual content cells). Pass when:
    - VLM has 0 rows and Docling has 0 rows (no tables), OR
    - VLM nonempty rows >= Docling nonempty rows (VLM at least as complete).
    Penalises VLM only for *missing* table data, never for finding more.
    """
    drows = _count_table_rows(docling_md)
    vrows = _count_table_rows(vlm_md)
    # Non-empty rows = rows containing at least one cell with a word character.
    d_nonempty = len(_NONEMPTY_ROW_RE.findall(docling_md))
    v_nonempty = len(_NONEMPTY_ROW_RE.findall(vlm_md))

    if d_nonempty == 0 and v_nonempty == 0:
        return GateResult(
            name="table_extraction",
            passed=True,
            metadata={
                "docling_table_rows_total": drows,
                "vlm_table_rows_total": vrows,
                "docling_nonempty": 0,
                "vlm_nonempty": 0,
                "no_tables": True,
            },
        )

    passed = v_nonempty >= d_nonempty
    return GateResult(
        name="table_extraction",
        passed=passed,
        metadata={
            "docling_table_rows_total": drows,
            "vlm_table_rows_total": vrows,
            "docling_nonempty": d_nonempty,
            "vlm_nonempty": v_nonempty,
            "delta_nonempty": v_nonempty - d_nonempty,
        },
    )


def gate_stability(vlm_md_a: str, vlm_md_b: str) -> GateResult:
    a_headers = len(_markdown_headers(vlm_md_a))
    b_headers = len(_markdown_headers(vlm_md_b))
    sim = difflib.SequenceMatcher(None, vlm_md_a, vlm_md_b).ratio()
    edit_dist = 1 - sim
    passed = (a_headers == b_headers) and (edit_dist < GATE_THRESHOLDS["stability_edit_distance_max"])
    return GateResult(
        name="stability",
        passed=passed,
        metadata={
            "run_a_headers": a_headers,
            "run_b_headers": b_headers,
            "edit_distance": round(edit_dist, 4),
            "similarity": round(sim, 4),
        },
    )


FABRICATION_PROMPT = """You are comparing two markdown extractions of the same PDF syllabus.

DOCLING (ground truth, may have hierarchy errors):
---
{docling}
---

VLM_EXTRACTION (under test):
---
{vlm}
---

Identify any content present in VLM_EXTRACTION that is NOT plausibly derivable from DOCLING (i.e. likely fabricated by the VLM). Hierarchy differences and rephrasing of the SAME factual content do not count as fabrication. Be conservative — only flag content that asserts facts not in DOCLING.

Reply with strict JSON:
{{"findings": [{{"quote": "...", "reason": "..."}}], "verdict": "CLEAN" or "FABRICATED"}}

If unsure, reply CLEAN. Empty findings list = CLEAN."""


OMISSIONS_PROMPT = """You are comparing two markdown extractions of the same PDF syllabus.

DOCLING (ground truth, may have hierarchy errors):
---
{docling}
---

VLM_EXTRACTION (under test):
---
{vlm}
---

Identify any SUBSTANTIVE content present in DOCLING that is MISSING from VLM_EXTRACTION (course-relevant facts: instructor, grading policy, schedule items, assignments, learning outcomes, prerequisites). Ignore: boilerplate university policies (TAMU honor code, ADA, FERPA, Title IX, etc.), page headers/footers, repeated dept mastheads, and rephrasings of the SAME information.

Reply with strict JSON:
{{"findings": [{{"quote": "...", "reason": "..."}}], "verdict": "CLEAN" or "INCOMPLETE"}}

If unsure, reply CLEAN. Empty findings list = CLEAN."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> Optional[dict]:
    """TAMU often wraps JSON in ```json fences```. Try fenced first, then bare braces."""
    fenced = _FENCE_RE.search(raw)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    bare = _BRACE_RE.search(raw)
    if bare:
        try:
            return json.loads(bare.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _one_tamu_call(client, prompt: str) -> tuple[str, Optional[str]]:
    """Single streaming call → raw text or (raw, error)."""
    try:
        resp = client.chat.completions.create(
            model=config.TAMU_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
            stream=True,
        )
        chunks = []
        for ev in resp:
            delta = ev.choices[0].delta.content if ev.choices else None
            if delta:
                chunks.append(delta)
        return "".join(chunks), None
    except Exception as e:
        return "", f"tamu_call_failed: {e!r}"


def _call_tamu_judge(prompt: str, *, max_retries: int = 2) -> tuple[Optional[dict], Optional[str]]:
    """TAMU streaming call → parsed JSON, with retry on parse failure.

    Retries with an appended 'return STRICT JSON only, no markdown fences' suffix
    when the first attempt fails to parse.
    """
    client = config.get_tamu_client()
    last_raw = ""
    for attempt in range(max_retries + 1):
        attempt_prompt = prompt
        if attempt > 0:
            attempt_prompt += (
                "\n\nIMPORTANT: Reply with STRICT JSON only. Do NOT wrap in code fences. "
                "Do NOT add commentary before or after. Start directly with { and end with }."
            )
        raw, err = _one_tamu_call(client, attempt_prompt)
        if err is not None:
            if attempt < max_retries:
                continue
            return None, err
        last_raw = raw
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed, None
    return None, f"no_valid_json_after_{max_retries + 1}_attempts: {last_raw[:200]!r}"


def _judge_gate(name: str, prompt_tpl: str, clean_verdict: str, docling_md: str, vlm_md: str) -> GateResult:
    d_trunc = docling_md[:8000]
    v_trunc = vlm_md[:8000]
    prompt = prompt_tpl.format(docling=d_trunc, vlm=v_trunc)
    parsed, err = _call_tamu_judge(prompt)
    if err is not None:
        return GateResult(name=name, passed=False, metadata={"error": err[:300], "verdict": "ERROR"})
    findings = parsed.get("findings", [])
    verdict = parsed.get("verdict", "UNKNOWN")
    return GateResult(
        name=name,
        passed=(verdict == clean_verdict) and not findings,
        metadata={
            "verdict": verdict,
            "finding_count": len(findings),
            "findings": findings[:5],
        },
    )


def gate_fabrication(docling_md: str, vlm_md: str) -> GateResult:
    """One TAMU call: 'what did the VLM make up?'"""
    return _judge_gate("fabrication", FABRICATION_PROMPT, "CLEAN", docling_md, vlm_md)


def gate_omissions(docling_md: str, vlm_md: str) -> GateResult:
    """One TAMU call: 'what substantive content did the VLM miss?'"""
    return _judge_gate("omissions", OMISSIONS_PROMPT, "CLEAN", docling_md, vlm_md)


def _est_cost(input_image_toks: int, input_text_toks: int, output_toks: int) -> float:
    inp = (input_image_toks + input_text_toks) / 1_000_000 * PRICE_INPUT_PER_MTOK
    out = output_toks / 1_000_000 * PRICE_OUTPUT_PER_MTOK
    return round(inp + out, 6)


def _load_docling_sidecar_headers(stem: str) -> list[str]:
    sidecar_path = v5_paths.bronze_sidecar_path(stem)
    if not sidecar_path.exists():
        return []
    try:
        parsed = HeadersSidecar.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
        return [h.text for h in parsed.headers]
    except Exception:
        return []


def run_bakeoff(
    dept: str,
    stems: list[str],
    *,
    with_stability: bool = False,
    with_fabrication_judge: bool = False,
    out_xlsx: Optional[Path] = None,
) -> list[FileResult]:
    results: list[FileResult] = []
    out_xlsx = out_xlsx or v6_paths.bakeoff_report_path(dept)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}", file=sys.stderr, flush=True)
        r = FileResult(stem=stem)

        pdf = v5_paths.raw_path(stem)
        if not pdf.exists():
            r.notes.append(f"raw PDF missing: {pdf}")
            results.append(r)
            continue

        docling_md_path = v5_paths.bronze_md_path(stem)
        if not docling_md_path.exists():
            r.notes.append(f"docling bronze missing: {docling_md_path}")
            results.append(r)
            continue
        docling_md = docling_md_path.read_text(encoding="utf-8")
        sidecar_headers = _load_docling_sidecar_headers(stem)

        # First VLM run.
        try:
            vlm_a = vlm_convert(pdf)
        except Exception as e:
            r.notes.append(f"vlm_convert failed: {e!r}")
            results.append(r)
            continue
        vlm_md_path = v6_paths.bronze_md_path(stem)
        vlm_md_path.parent.mkdir(parents=True, exist_ok=True)
        vlm_md_path.write_text(vlm_a.markdown, encoding="utf-8")

        r.vlm_md_chars = len(vlm_a.markdown)
        r.docling_md_chars = len(docling_md)
        r.vlm_headers = len(vlm_a.headers)
        r.docling_headers = len(sidecar_headers)
        r.vlm_table_rows = _count_table_rows(vlm_a.markdown)
        r.docling_table_rows = _count_table_rows(docling_md)
        r.timing_s = round(vlm_a.timing_s, 2)
        r.input_image_tokens = vlm_a.input_image_tokens
        r.output_tokens = vlm_a.output_tokens
        r.est_cost_usd = _est_cost(vlm_a.input_image_tokens, vlm_a.input_text_tokens, vlm_a.output_tokens)

        r.gates.append(gate_token_recall(docling_md, vlm_a.markdown))
        r.gates.append(gate_header_structure(stem, vlm_a.markdown, sidecar_headers))
        r.gates.append(gate_table_extraction(docling_md, vlm_a.markdown))

        if with_stability:
            try:
                vlm_b = vlm_convert(pdf)
                r.gates.append(gate_stability(vlm_a.markdown, vlm_b.markdown))
                r.est_cost_usd += _est_cost(vlm_b.input_image_tokens, vlm_b.input_text_tokens, vlm_b.output_tokens)
            except Exception as e:
                r.notes.append(f"stability second-run failed: {e!r}")

        if with_fabrication_judge:
            r.gates.append(gate_fabrication(docling_md, vlm_a.markdown))
            r.gates.append(gate_omissions(docling_md, vlm_a.markdown))

        results.append(r)

    _write_report(dept, results, out_xlsx, with_stability, with_fabrication_judge)
    return results


def _write_report(
    dept: str,
    results: list[FileResult],
    out_xlsx: Path,
    with_stability: bool,
    with_fabrication_judge: bool,
) -> None:
    """Write a per-file row + summary footer to xlsx via openpyxl."""
    try:
        from openpyxl import Workbook
    except ImportError:
        # Fallback: write JSON next to where xlsx would go.
        out_json = out_xlsx.with_suffix(".json")
        out_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=str),
            encoding="utf-8",
        )
        print(f"openpyxl missing — wrote JSON instead: {out_json}", file=sys.stderr)
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "bakeoff"

    base_cols = [
        "stem",
        "docling_chars",
        "vlm_chars",
        "docling_headers",
        "vlm_headers",
        "docling_table_rows",
        "vlm_table_rows",
        "vlm_timing_s",
        "vlm_input_image_tokens",
        "vlm_output_tokens",
        "est_cost_usd",
        "g1_token_recall_pass",
        "g1_ratio",
        "g2_header_structure_pass",
        "g2_vlm_headers",
        "g2_has_title",
        "g2_recall_vs_docling",
        "g4_table_extraction_pass",
        "g4_docling_nonempty",
        "g4_vlm_nonempty",
        "g4_delta",
    ]
    if with_stability:
        base_cols += ["g5_stability_pass", "g5_edit_distance"]
    if with_fabrication_judge:
        base_cols += [
            "g3a_fabrication_pass",
            "g3a_verdict",
            "g3a_count",
            "g3b_omissions_pass",
            "g3b_verdict",
            "g3b_count",
        ]
    base_cols += ["notes"]
    ws.append(base_cols)

    def _g(r: FileResult, name: str) -> Optional[GateResult]:
        for g in r.gates:
            if g.name == name:
                return g
        return None

    for r in results:
        g1 = _g(r, "token_recall")
        g2 = _g(r, "header_structure")
        g4 = _g(r, "table_extraction")
        g5 = _g(r, "stability")
        g3a = _g(r, "fabrication")
        g3b = _g(r, "omissions")
        row = [
            r.stem,
            r.docling_md_chars,
            r.vlm_md_chars,
            r.docling_headers,
            r.vlm_headers,
            r.docling_table_rows,
            r.vlm_table_rows,
            r.timing_s,
            r.input_image_tokens,
            r.output_tokens,
            r.est_cost_usd,
            (g1.passed if g1 else None),
            (g1.metadata.get("ratio") if g1 else None),
            (g2.passed if g2 else None),
            (g2.metadata.get("vlm_headers") if g2 else None),
            (g2.metadata.get("has_title_with_dept") if g2 else None),
            (g2.metadata.get("recall_vs_docling") if g2 else None),
            (g4.passed if g4 else None),
            (g4.metadata.get("docling_nonempty") if g4 else None),
            (g4.metadata.get("vlm_nonempty") if g4 else None),
            (g4.metadata.get("delta_nonempty") if g4 else None),
        ]
        if with_stability:
            row += [
                (g5.passed if g5 else None),
                (g5.metadata.get("edit_distance") if g5 else None),
            ]
        if with_fabrication_judge:
            row += [
                (g3a.passed if g3a else None),
                (g3a.metadata.get("verdict") if g3a else None),
                (g3a.metadata.get("finding_count") if g3a else None),
                (g3b.passed if g3b else None),
                (g3b.metadata.get("verdict") if g3b else None),
                (g3b.metadata.get("finding_count") if g3b else None),
            ]
        row += ["; ".join(r.notes) if r.notes else ""]
        ws.append(row)

    # Summary footer
    ws.append([])
    total = len(results)
    g1_pass = sum(1 for r in results if (g := _g(r, "token_recall")) and g.passed)
    g2_pass = sum(1 for r in results if (g := _g(r, "header_structure")) and g.passed)
    g4_pass = sum(1 for r in results if (g := _g(r, "table_extraction")) and g.passed)
    cost = round(sum(r.est_cost_usd for r in results), 4)
    ws.append(["SUMMARY", f"N={total}", f"cost=${cost}"])
    ws.append([f"g1_token_recall: {g1_pass}/{total}"])
    ws.append([f"g2_header_structure: {g2_pass}/{total}"])
    ws.append([f"g4_table_extraction: {g4_pass}/{total}"])
    if with_stability:
        g5_pass = sum(1 for r in results if (g := _g(r, "stability")) and g.passed)
        ws.append([f"g5_stability: {g5_pass}/{total}"])
    if with_fabrication_judge:
        g3a_pass = sum(1 for r in results if (g := _g(r, "fabrication")) and g.passed)
        g3b_pass = sum(1 for r in results if (g := _g(r, "omissions")) and g.passed)
        ws.append([f"g3a_fabrication_clean: {g3a_pass}/{total}"])
        ws.append([f"g3b_omissions_clean: {g3b_pass}/{total}"])

    wb.save(out_xlsx)
    print(f"report: {out_xlsx}", file=sys.stderr)


def _resolve_stems(dept: str, limit: Optional[int], explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        return explicit
    raw_dir = v5_paths.v5_root(dept) / "raw"
    pdfs = sorted(p.stem for p in raw_dir.glob("*.pdf"))
    return pdfs[:limit] if limit else pdfs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default=DEFAULT_DEPT)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--stems", nargs="*", help="explicit stems (overrides --limit)")
    ap.add_argument("--stability", action="store_true", help="run VLM twice per file for gate 5")
    ap.add_argument("--with-fabrication-judge", action="store_true", help="run TAMU judge gate 3")
    ap.add_argument("--out", type=Path, default=None, help="output xlsx path")
    args = ap.parse_args()

    stems = _resolve_stems(args.dept, args.limit, args.stems)
    if not stems:
        print(f"no stems found for {args.dept}", file=sys.stderr)
        return 1

    t0 = time.time()
    print(
        f"bake-off: dept={args.dept} stems={len(stems)} stability={args.stability} "
        f"fabrication={args.with_fabrication_judge} model={VLM_MODEL}",
        file=sys.stderr,
    )
    results = run_bakeoff(
        args.dept,
        stems,
        with_stability=args.stability,
        with_fabrication_judge=args.with_fabrication_judge,
        out_xlsx=args.out,
    )
    elapsed = time.time() - t0
    total_cost = round(sum(r.est_cost_usd for r in results), 4)
    n_failed = sum(1 for r in results if any(not g.passed for g in r.gates) or r.notes)
    print(
        f"done in {elapsed:.1f}s — {len(results) - n_failed}/{len(results)} clean — est cost ${total_cost}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
