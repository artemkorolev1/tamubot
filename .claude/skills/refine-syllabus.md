---
name: refine-syllabus
description: Use when manually fixing or refining a processed syllabus markdown file — reading the file, comparing against source PDF stages, applying structural/content fixes, re-validating, and updating the Excel report
triggers: ["refine syllabus", "fix syllabus", "manually fix", "refine file", "fix markdown", "manual fix"]
---

# Refine Syllabus — Manual QA Fix for Processed Syllabi

Announce: "Using refine-syllabus skill."

Manually fix structural and content errors in processed syllabus markdown files (`data/syllabi/silver/04_hierarchy/*.md`), then validate and update the tracking report.

## Step 0 — Gather Details

Ask the user for ALL of the following before starting. Do not proceed until you have answers:

1. **Which file(s)?** — file stem (e.g. `202641_ISEN_667_700_47617`) or pattern (e.g. "all Fall 2026 ISEN files with 4+ errors")
2. **Source PDF?** — does the user have the original PDF to attach? If yes, ask them to attach it. If not, you will compare against earlier pipeline stages.
3. **Known issues?** — any specific problems the user already knows about (e.g. "duplicate instructor info", "broken table", "missing section"), or should you diagnose from the validation report?
4. **Version label?** — what version to use when appending results to the report (e.g. `v8`, `v9`). Check the report for the latest version and suggest the next one.

## Step 1 — Read and Diagnose

### 1a. Check validation results

```python
import json
from pathlib import Path

val_path = Path("data/syllabi/silver/05_validate") / f"{stem}_validation.json"
if val_path.exists():
    data = json.loads(val_path.read_text())
    for cat, items in data["findings"].items():
        if items:
            print(f"  {cat}: {len(items)}")
            for item in items:
                desc = item.get("description", item.get("finding", str(item)))
                print(f"    - {desc}")
```

### 1b. Check all pipeline stages

Read the file at each stage to understand where errors were introduced:

| Stage | Path |
|---|---|
| Bronze (Docling raw) | `data/syllabi/bronze/{stem}.md` |
| Image recovery (Gemini) | `data/syllabi/silver/01_image_recovery/{stem}.md` |
| False positive filter | `data/syllabi/silver/02_false_positive/{stem}.md` |
| Boilerplate strip | `data/syllabi/silver/03_boilerplate/{stem}.md` |
| Hierarchy (current) | `data/syllabi/silver/04_hierarchy/{stem}.md` |

Check if the file was processed by Gemini image recovery — if so, use the `01_image_recovery` version as the base truth, not the bronze Docling output.

### 1c. Compare against source PDF

If the user attached a PDF, use it as ground truth for content verification. Otherwise, cross-reference the pipeline stages to identify what content existed in the source but was lost or mangled.

## Step 2 — Classify Issues

Separate issues into:

- **Fixable**: structural errors (heading levels, duplicate sections, orphaned labels, broken tables, misplaced content, indentation, image artifacts)
- **Source data limitations**: content missing from the original PDF extraction (e.g. missing instructor details, image-only schedules) — note these but don't fabricate content
- **LLM validator false positives**: validator flagged correct output — note but don't change

Present the diagnosis and proposed fixes to the user. **STOP — get confirmation before editing.**

## Step 3 — Apply Fixes

Common fix patterns (from prior manual QA sessions):

### Heading level fixes
```
#### Chapter 1:  →  ### Chapter 1:     (demoted subsection → proper level)
### Late Work Policy  →  ## Late Work Policy  (promoted to correct H2)
# e. In accordance...  →  e. In accordance...  (heading → list item)
```

### Duplicate content removal
- Duplicate instructor info (email/phone/office appearing twice)
- Duplicate makeup/late work policy sections
- Duplicate "Credit Hours" lines

### Label-value joins
```
Office Hours        →  Office Hours: Thursday 3-5
<blank line>
Thursday 3-5
```

### Content relocation
- Textbook info misplaced in Course Schedule → move to Textbook section
- Meeting times in Course Schedule section → keep but note it's meeting info, not schedule

### Boilerplate removal
- Excused absence enumeration (items 1-7 + a-e sub-items from Student Rule 7)
- Duplicate Course Specific Makeup Work Policy that repeats Late Work Policy

### Artifact cleanup
- Image description artifacts ("The image shows the cover of...")
- `<!-- image -->` markers
- Broken table fragments (`|    |`, `|----|`)
- Non-breaking spaces, HTML entities (`&lt;` → `<`)
- "Spring break" in Fall semester schedules

### Formatting
- Grading breakdowns → bullet lists
- Exam dates → bullet lists
- Excessive indentation cleanup
- Excessive blank lines (3+ → 2)

**For heavy rewrites**: if 5+ issues in a single file, consider rewriting the file from scratch using the source PDF or earliest clean pipeline stage as the base, applying proper markdown structure throughout.

## Step 4 — Run Automated Fixes (Optional)

If the same mechanical pattern affects many files, write or use a script:

```python
# Example: scripts/fix_fall26_isen.py applies 8 automated fix passes
# Label-value joins, duplicate headers, duplicate credit hours,
# image artifacts, double-hyphen lists, image markers, broken tables,
# excessive blank lines
python scripts/fix_fall26_isen.py
```

Only run automated scripts on files NOT already manually fixed — check with the user.

## Step 5 — Validate

Invoke the **task-budget** skill first — each validation is ~1 TAMU API call (~25s).

```python
import sys
sys.path.insert(0, "src")
from tamubot.ingestion.validators.llm_validator import validate_file
from pathlib import Path

hier_dir = Path("data/syllabi/silver/04_hierarchy")
val_dir = Path("data/syllabi/silver/05_validate")

for stem in fixed_stems:
    result = validate_file(hier_dir / f"{stem}.md", output_dir=val_dir)
    print(f"{stem}: {result.total_issues} issues — {result.issue_counts}")
```

## Step 6 — Update Report

Append the new version columns to the Excel report:

```python
import json
sys.path.insert(0, "src")
from tamubot.ingestion.report_writer import update_validation
from pathlib import Path

report = Path("data/syllabi/silver/docling_pipeline_v4_report.xlsx")
val_dir = Path("data/syllabi/silver/05_validate")

for stem in fixed_stems:
    val_file = val_dir / f"{stem}_validation.json"
    data = json.loads(val_file.read_text())
    findings = data["findings"]
    counts = {cat: len(items) for cat, items in findings.items()}
    # Convert finding dicts to description strings
    findings_str = {}
    for cat, items in findings.items():
        findings_str[cat] = [
            item.get("description", item.get("finding", str(item)))
            if isinstance(item, dict) else str(item)
            for item in items
        ]
    update_validation(report, stem, version_label, counts, findings_str)
    print(f"Updated {stem}: {counts}")
```

## Step 7 — Report Results

Present a before/after table:

```
| File | v{N-1} | v{N} | Delta | Key fixes applied |
|------|--------|------|-------|-------------------|
| ...  | ...    | ...  | ...   | ...               |
```

## Key Files

| File | Purpose |
|---|---|
| `data/syllabi/silver/04_hierarchy/*.md` | Files to fix (final pipeline output) |
| `data/syllabi/silver/05_validate/*_validation.json` | Per-file validation results |
| `data/syllabi/silver/docling_pipeline_v4_report.xlsx` | Cross-iteration tracking report |
| `src/tamubot/ingestion/validators/llm_validator.py` | `validate_file(path, output_dir=)` |
| `src/tamubot/ingestion/report_writer.py` | `update_validation(report, stem, version, counts, findings)` |
| `scripts/fix_fall26_isen.py` | Automated mechanical fix script (Fall 2026 ISEN) |

## Gotchas

- **Check Gemini stage first**: if report shows `Gemini Recovered: YES`, use the `01_image_recovery` file as base truth, not the bronze Docling output.
- **LLM validation variance**: re-validating unchanged files can produce +/-1-2 different findings. Don't chase variance on files you didn't change.
- **Don't fabricate content**: if instructor details or schedule content is missing from all pipeline stages, it was never in the PDF. Note it as a source data limitation.
- **TAMU API uses SSE**: set `stream=True` (already handled by `validate_file`). Each call takes ~25s.
- **Non-breaking spaces**: Docling sometimes emits `\u00a0` which looks identical to a space but breaks string matching. Use Python to fix these, not the Edit tool.
- **Report version**: always check what the latest version in the report is before suggesting a new one to avoid collisions.
