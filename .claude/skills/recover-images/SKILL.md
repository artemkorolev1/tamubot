---
name: recover-images
description: Use when running the v5 silver/01_image_recovery stage WITHOUT Gemini — Claude reads PDF page renders directly and rewrites <!-- image --> markers as markdown tables / brief figure descriptions / removed (for logos). Free, no API budget.
triggers: ["recover images", "image recovery", "claude image recovery", "silver image recovery", "replace image markers", "no gemini image recovery"]
---

# Recover Images — Claude-driven (no Gemini)

Announce: "Using recover-images skill."

## When to use

Drop-in replacement for `silver_image_recovery` (which calls Gemini multimodal). Use this when:
- Processing a new department's bronze corpus and you want zero API spend on this stage.
- Reprocessing a single file whose Gemini recovery was poor.
- Any time `<!-- image -->` markers need to become real markdown content.

NOT for: fixing layout outside of marker regions (use `refine-syllabus` for that).

## Inputs / outputs

Per-stem contract is identical to the Dagster asset — downstream filters (`silver_false_positive`, `silver_boilerplate`, etc.) don't care which path produced the file.

| Path | Role |
|---|---|
| `data/syllabi/<DEPT>/v5/bronze/<stem>.md` | Input markdown with `<!-- image -->` markers |
| `data/syllabi/<DEPT>/v5/bronze/<stem>.headers.json` | Sidecar; gives header→page map |
| `data/syllabi/<DEPT>/v5/raw/<stem>.pdf` | Source PDF |
| `data/syllabi/<DEPT>/v5/silver/01_image_recovery/<stem>.md` | **Output** |
| `data/syllabi/<DEPT>/v5/.image_recovery_work/<stem>/` | Temp PNGs + manifest (cleaned at end) |
| `data/syllabi/<DEPT>/v5/logs/manual_edits.csv` | One row per file (mandatory) |
| `data/syllabi/<DEPT>/v5/silver/pipeline_v5_report.xlsx` | Refreshed at batch end |

## Replacement rules (mirrors current Gemini prompt)

For each `<!-- image -->` marker:

1. **Tables** → standard markdown table with `|` delimiters. Preserve column order and all cells visible in the rendered page.
2. **Schedules / week-by-week** → markdown table (these are tables that lost their borders in Docling).
3. **Figures / diagrams** → one brief paragraph describing the figure's content (what it shows, key labels). Don't invent specifics.
4. **Logos / decorative banners / textbook covers** → remove the marker entirely (replace with empty line).
5. **Cannot tell what's there** → leave the marker in place AND log it (see "Recovery failures").

**Preserve everything else verbatim.** Don't reformat surrounding markdown, don't fix typos, don't merge or split paragraphs. Only the marker line is in scope.

## Workflow — single file

```bash
# 1. Render pages containing markers (free, fast)
python scripts/render_marker_pages.py --stem <STEM>
```

Then for that stem:

1. Read `data/syllabi/<DEPT>/v5/.image_recovery_work/<STEM>/manifest.json` → list of markers with line_no, page numbers, and PNG paths.
2. Read the bronze md: `data/syllabi/<DEPT>/v5/bronze/<STEM>.md`.
3. For each marker entry in the manifest, Read the relevant `page_NN.png` files. Identify what's in the marker region using the marker's surrounding `context.before` / `context.after` from the manifest as anchors.
4. Use `Edit` (replace_all=False) on the bronze content to swap each `<!-- image -->` line for the resolved content. The simplest is to read the bronze file into memory, do the substitutions, and `Write` the result to silver.
5. Write output to `data/syllabi/<DEPT>/v5/silver/01_image_recovery/<STEM>.md`.
6. **Log** the recovery (see "Logging" below).
7. Delete the work dir: `rm -rf data/syllabi/<DEPT>/v5/.image_recovery_work/<STEM>`.

## Workflow — batch (recommended for a full department)

```bash
# 1. Render pages for all files with ≥2 markers
python scripts/render_marker_pages.py --dept <DEPT> --all

# 2. Pass-through copy files with <2 markers (logo-only, no recovery needed)
python -c "
import shutil
from pathlib import Path
bronze = Path('data/syllabi/<DEPT>/v5/bronze')
out = Path('data/syllabi/<DEPT>/v5/silver/01_image_recovery')
out.mkdir(parents=True, exist_ok=True)
for md in sorted(bronze.glob('*.md')):
    if md.read_text(encoding='utf-8').count('<!-- image -->') < 2:
        shutil.copy2(md, out / md.name)
        print(f'passthrough: {md.stem}')
"
```

Then process each manifest file in turn (or dispatch parallel subagents — one per stem, each with the SKILL.md context and the stem name). Each stem is independent; no shared state.

After all stems are done:

```bash
# 3. Clean up work dirs
rm -rf data/syllabi/<DEPT>/v5/.image_recovery_work

# 4. Refresh the v5 report (creates file if missing)
python scripts/build_v5_pilot_report.py --dept <DEPT>
```

## Logging — mandatory

Per `iterate-pipeline`'s manual-edits rule, every file recovered via this skill MUST log to `data/syllabi/<DEPT>/v5/logs/manual_edits.csv`. The v5 report builder reads this column on the Summary sheet — without the row, the report can't show why a stem's markers dropped between runs.

```python
import csv, datetime
from pathlib import Path

log = Path(f"data/syllabi/{DEPT}/v5/logs/manual_edits.csv")
log.parent.mkdir(parents=True, exist_ok=True)
new = not log.exists()
with log.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "stem", "stage", "applied_by", "llm_calls", "summary"])
    w.writerow([
        datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        STEM,
        "01_image_recovery",
        "claude-visual",
        0,
        SUMMARY,  # e.g. "Recovered 4 markers: 2 tables, 1 figure description, 1 logo removed"
    ])
```

Pass-through files (markers < 2) do NOT need a log row — they're identical to bronze.

## Report refresh — mandatory at batch end

```bash
python scripts/build_v5_pilot_report.py --dept <DEPT>
```

This is a no-LLM, on-disk aggregator. It:
- Reads bronze + silver/01_image_recovery marker counts directly from .md files
- Derives "img_status" = "recovered" if markers dropped, else "passthrough"
- Picks up the manual_edits.csv rows for the Summary sheet's "Manual Edits" column
- Output: `data/syllabi/<DEPT>/v5/silver/pipeline_v5_report.xlsx`

**Do not skip this** even though no LLM calls happened. The report is the source of truth for stage status; if it's stale, the next iteration session sees a misleading picture.

## Recovery failures — when you can't tell what's there

If a page render is too low-res, the image overlaps a page break, or the content is genuinely undecipherable:
- Leave the `<!-- image -->` marker in place in the output file.
- Note the marker in the log row's summary, e.g. `"Recovered 3 of 4 markers; 1 unresolved (page 5, unclear table)"`.
- Don't invent content. The v5 report flags files with non-zero residual markers in silver/01.

If multiple markers per file are unresolvable, the file is a candidate for `refine-syllabus` follow-up.

## Page-mapping caveats

The helper (`render_marker_pages.py`) maps each marker to the page of the nearest preceding header via the Docling sidecar. Limitations:
- Markers near a page break may render the **previous** page when the actual content is on the next. The helper always renders `page` + `page+1` to cover this.
- `### subheaders` not present in the sidecar inherit the previous resolved page. Usually correct, but check the rendered page actually shows the surrounding context before substituting.
- If you see a marker whose context doesn't match anything on the rendered pages, broaden via `--force` and add page hints by hand: re-render a wider range with `fitz` manually, or fall back to reading the whole PDF via `Read`.

## Cross-references

- [[iterate-pipeline]] — full v5 stage QA loop; mentions this skill as the no-cost path
- [[refine-syllabus]] — for non-marker fixes after recovery
- [[task-budget]] — not needed here (this skill is free), but check before running validate/enrich downstream
