from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tamubot.core import config

CATEGORIES = [
    "content_preservation",
    "strip_completeness",
    "structural_coherence",
    "metadata_accuracy",
]

SYSTEM_PROMPT = """You are a quality checker for processed university syllabus documents.
The document below was converted from PDF to markdown, then had institutional boilerplate
sections stripped. Audit it for problems in 4 categories. Return ONLY valid JSON.

{
  "content_preservation": [
    "Specific description of course content that appears to be missing or was incorrectly removed"
  ],
  "strip_completeness": [
    "Exact header or section that is institutional boilerplate but was NOT stripped"
  ],
  "structural_coherence": [
    "Specific structural problem: broken table, orphaned header, wrong heading level, etc."
  ],
  "metadata_accuracy": [
    "Specific metadata field that is missing, incorrect, or malformed (e.g. instructor name, credit hours, CRN)"
  ]
}

Rules:
- Each array entry must be a specific, actionable finding — not a vague observation.
- For content_preservation: name the exact content that was lost (e.g. "Grading breakdown table is missing").
- For strip_completeness: quote the exact header text still present (e.g. "Section 'Americans with Disabilities Act' is boilerplate and should be removed").
- For structural_coherence: describe exactly what is wrong (e.g. "Table in Course Schedule has empty header row '|    |'").
- For metadata_accuracy: name the field and what is wrong (e.g. "Instructor email is missing").
- Empty arrays are fine — only report real problems.
- Boilerplate = university-wide institutional text (ADA, Title IX, attendance policy, etc.). Course-specific sections like grading policy, schedule, late work policy are NOT boilerplate — even if they reference university rules (e.g. "Student Rule 7").
- A section that contains course-specific policy details (grading breakdown, late penalties, makeup exam terms) is NOT boilerplate regardless of whether it also cites a university rule.
- Do NOT flag missing sections that were correctly stripped as boilerplate."""

PROPOSAL_PROMPT = """You are reviewing a processed university syllabus markdown document for quality.
Based on the issues found, propose specific fixes. Return ONLY valid JSON.

{
  "new_false_positives": ["header text that should NOT be a header — demote to body text"],
  "new_boilerplate": ["header text of sections that are institutional boilerplate and should be stripped entirely"],
  "text_cleanups": ["regex pattern or literal string that should be removed/replaced in the output"],
  "reasoning": "brief explanation of why each proposal is needed"
}

Rules:
- new_false_positives: headers that are NOT real section headers (e.g. location names, material types, short labels promoted by the PDF parser)
- new_boilerplate: ONLY university-wide institutional text that appears verbatim across ALL syllabi. NEVER add course-specific sections like grading policy, schedule, late work policy.
- text_cleanups: artifacts like HTML comments, encoding errors, placeholder tags
- Be conservative. Only propose additions you are confident about."""


@dataclass
class ValidationResult:
    file_stem: str
    findings: dict[str, list[str]]
    timing_s: float
    proposals: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_issues(self) -> int:
        return sum(len(v) for v in self.findings.values())

    @property
    def issue_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.findings.items()}


def _build_user_prompt(markdown: str, metadata: dict | None) -> str:
    parts = [f"<document>\n{markdown}\n</document>"]
    if metadata:
        parts.append(f"\n<metadata>\n{json.dumps(metadata, indent=2)}\n</metadata>")
    return "\n".join(parts)


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    client = config.get_tamu_client()
    stream = client.chat.completions.create(
        model=config.TAMU_MODEL,
        temperature=0.0,
        stream=True,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    # TAMU gateway always returns SSE — collect streamed chunks
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream).strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def _get_proposals(markdown: str, all_findings: list[str]) -> dict[str, list[str]]:
    """Ask the LLM to propose registry additions and file cleanups."""
    user_prompt = "## Issues found\n\n" + "\n".join(f"- {i}" for i in all_findings) + f"\n\n## Document\n\n{markdown}"
    try:
        result = _call_llm(PROPOSAL_PROMPT, user_prompt)
        return {
            "new_false_positives": result.get("new_false_positives", []),
            "new_boilerplate": result.get("new_boilerplate", []),
            "text_cleanups": result.get("text_cleanups", []),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception:
        return {}


def _error_result(file_stem: str, error: str, timing: float) -> ValidationResult:
    return ValidationResult(
        file_stem=file_stem,
        findings={cat: [] for cat in CATEGORIES},
        timing_s=timing,
    )


def validate_file(
    markdown_path: Path,
    metadata: dict | None = None,
    output_dir: Path | None = None,
    propose_fixes: bool = True,
    version_label: str | None = None,
) -> ValidationResult:
    stem = markdown_path.stem
    markdown = markdown_path.read_text(encoding="utf-8")
    prompt = _build_user_prompt(markdown, metadata)

    t0 = time.perf_counter()
    result_dict = None
    last_error = ""

    for attempt in range(2):
        try:
            result_dict = _call_llm(SYSTEM_PROMPT, prompt)
            break
        except Exception as e:
            last_error = str(e)

    elapsed = round(time.perf_counter() - t0, 2)

    if result_dict is None:
        vr = _error_result(stem, last_error, elapsed)
        vr.findings.setdefault("_errors", []).append(f"LLM error: {last_error}")
    else:
        findings = {cat: result_dict.get(cat, []) for cat in CATEGORIES}
        # Flatten all findings for proposal generation
        all_issues = [issue for cat_issues in findings.values() for issue in cat_issues]
        proposals = {}
        if propose_fixes and all_issues:
            proposals = _get_proposals(markdown, all_issues)
        vr = ValidationResult(
            file_stem=stem,
            findings=findings,
            timing_s=round(time.perf_counter() - t0, 2),
            proposals=proposals,
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Always write the canonical unversioned file (latest result).
        report_path = output_dir / f"{stem}_validation.json"
        report_path.write_text(json.dumps(asdict(vr), indent=2), encoding="utf-8")
        # Also write a versioned snapshot when a label is provided, so the
        # report builder can show v1/v2/... side-by-side without losing
        # prior runs.
        if version_label:
            vpath = output_dir / f"{stem}_validation_{version_label}.json"
            payload = asdict(vr)
            payload["version"] = version_label
            vpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return vr


# ── Proposal application ─────────────────────────────────────────────────────

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def apply_proposals(
    proposals: dict[str, list[str]],
    markdown_dir: Path,
) -> dict[str, int]:
    """Apply confirmed proposals to markdown files in-place. Returns counts."""
    counts = {"false_positives_demoted": 0, "text_cleanups_applied": 0}

    fp_set = {t.strip().rstrip(":").lower() for t in proposals.get("new_false_positives", [])}
    cleanups = proposals.get("text_cleanups", [])

    for md_path in sorted(markdown_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        out_lines = []

        for line in lines:
            m = _HEADER_RE.match(line)
            if m and m.group(2).strip().rstrip(":").lower() in fp_set:
                out_lines.append(m.group(2).strip())
                counts["false_positives_demoted"] += 1
            else:
                out_lines.append(line)

        result = "\n".join(out_lines)

        for pattern in cleanups:
            before = result
            result = re.sub(re.escape(pattern), "", result)
            if result != before:
                counts["text_cleanups_applied"] += 1

        md_path.write_text(result, encoding="utf-8")

    return counts


def merge_proposals(results: list[ValidationResult]) -> dict[str, list[str]]:
    """Deduplicate proposals across multiple validation results."""
    fp: set[str] = set()
    bp: set[str] = set()
    cl: set[str] = set()
    reasons: list[str] = []

    for vr in results:
        if not vr.proposals:
            continue
        fp.update(vr.proposals.get("new_false_positives", []))
        bp.update(vr.proposals.get("new_boilerplate", []))
        cl.update(vr.proposals.get("text_cleanups", []))
        r = vr.proposals.get("reasoning", "")
        if r:
            reasons.append(f"{vr.file_stem}: {r}")

    return {
        "new_false_positives": sorted(fp),
        "new_boilerplate": sorted(bp),
        "text_cleanups": sorted(cl),
        "reasoning": reasons,
    }


def validate_directory(
    input_dir: Path,
    metadata_dir: Path | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    version_label: str | None = None,
    file_pattern: str = "*.md",
    limit: int | None = None,
) -> list[ValidationResult]:
    md_files = sorted(input_dir.glob(file_pattern))
    if limit:
        md_files = md_files[:limit]
    results: list[ValidationResult] = []
    if report_path is None:
        from tamubot.ingestion.report_writer import get_report

        report_path = get_report()

    for md_path in md_files:
        metadata = None
        if metadata_dir:
            meta_path = metadata_dir / f"{md_path.stem}.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        vr = validate_file(md_path, metadata=metadata, output_dir=output_dir, version_label=version_label)
        results.append(vr)
        counts = vr.issue_counts
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0) or "clean"
        print(f"  {vr.file_stem}: {summary} ({vr.timing_s:.1f}s)")

        if report_path and version_label:
            from tamubot.ingestion.report_writer import update_validation

            update_validation(
                report_path,
                vr.file_stem,
                version_label,
                vr.issue_counts,
                vr.findings,
            )

    if output_dir and results:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "validation_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_stem",
                    "total_issues",
                    *CATEGORIES,
                    "timing_s",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            for vr in results:
                writer.writerow(
                    {
                        "file_stem": vr.file_stem,
                        "total_issues": vr.total_issues,
                        **vr.issue_counts,
                        "timing_s": vr.timing_s,
                    }
                )
        print(f"\nSummary CSV: {csv_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based syllabus quality validation")
    parser.add_argument("input_dir", type=Path, help="Directory of markdown files")
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out = args.output_dir or args.input_dir / "validation"
    results = validate_directory(args.input_dir, args.metadata_dir, out)
    total = sum(r.total_issues for r in results)
    print(f"\n{len(results)} files validated — {total} total issues found")
