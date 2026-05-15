---
name: iterate-pipeline
description: Use when iteratively improving the v4 ingestion pipeline — onboard a new department, run a pilot, diagnose issues, apply fixes, re-validate, and emit a per-department Excel report matching the ISEN layout
triggers: ["iterate pipeline", "pipeline iteration", "fix pipeline issues", "pipeline QA", "validate pipeline", "improve pipeline", "iterate-pipeline", "process new department", "ingest new department"]
---

# Iterate Pipeline — v4 QA & New-Department Onboarding

Announce: "Using iterate-pipeline skill."

## When to use

- Onboarding a new department (e.g. CSCE was added on top of ISEN — see "New Department Onboarding" below).
- Iteratively improving an existing department's pipeline output: validate → diagnose → fix → re-run affected files → track in Excel.

## Critical paths (v4)

**Per-department medallion tree** (used when `--department` is supplied to `pipeline_v4`):
```
data/syllabi/<DEPT>/raw/      <stem>_vNNN.pdf
data/syllabi/<DEPT>/bronze/   <stem>_vNNN.md     (Docling output)
data/syllabi/<DEPT>/silver/
  01_image_recovery/<stem>_vNNN.md     (Gemini multimodal)
  02_false_positive/<stem>_vNNN.md     (rule-based)
  03_boilerplate/<stem>_vNNN.md        (rule-based; sidecar *_stripped.txt)
  04_hierarchy/<stem>_vNNN.md          (rule-based)
  05_enrich/<stem>_vNNN.json           (LLM metadata + summary; not a pipeline step yet)
  05_validate/<stem>_vNNN_validation.json
  06_chunk/<stem>_vNNN.json
  docling_pipeline_v4_report.xlsx      (per-dept multi-sheet report)
data/syllabi/<DEPT>/gold/     <stem>_vNNN.json
data/syllabi/<DEPT>/logs/     <step>_log.csv
```

**Legacy shared tree** (`data/syllabi/{raw,bronze,silver,gold,logs}/`): holds the ISEN files from before per-dept namespacing. Don't write there for new departments — `pipeline_v4.setup_paths(dept)` re-points the constants automatically when `--department` is given.

| File | Purpose |
|---|---|
| `src/tamubot/ingestion/pipeline_v4.py` | Orchestrator. `setup_paths(dept)` namespaces all roots. `_input_dir_for_step` walks back through silver dirs for steps run in isolation. |
| `src/tamubot/ingestion/filters/{image_recovery,false_positive,boilerplate,hierarchy,metadata_enrichment}.py` | Each filter's `apply()` honors `config["file_pattern"]` and `config["limit"]`. |
| `src/tamubot/ingestion/boilerplate_stripper.py` | `BOILERPLATE_REGISTRY` (dept-agnostic), `BODY_BOILERPLATE_HEADERS`, `_BP_KEYWORDS` (flags new candidates). |
| `src/tamubot/ingestion/filters/image_recovery.py` | `RAW_ROOT` is mutated by `pipeline_v4.setup_paths`. `_find_raw_pdf` falls back to `glob({base_stem}_v???.pdf)` to handle copy/convert version mismatch. |
| `src/tamubot/ingestion/validators/llm_validator.py` | `validate_directory(file_pattern=..., limit=...)`. |
| `src/tamubot/ingestion/ingest.py` | `--v4 --department X` prefers `data/syllabi/<DEPT>/silver/06_chunk/`, falls back to shared for legacy ISEN. |
| `scripts/build_csce_pilot_report.py` | Builds the per-dept multi-sheet Excel report (ISEN-style layout). |

## New Department Onboarding

When a new department needs to be ingested from scratch:

1. **Enable scrapers**: add the dept code to `DEPARTMENTS = {...}` in both `src/tamubot/scraper/spiders/class_search_spider.py:14` and `src/tamubot/scraper/download_simple_syllabus.py:23`. `catalog_spider.py:11` has `DEPT_PATHS` — add catalog paths if not already present.

2. **Scrape**: invoke `scrape-howdy-portal`, `scrape-simple-syllabus`, and `scrape-catalog` skills. Expect Summer/Fall future-term sections to return 0 from Howdy Portal until TAMU registers them, but Simple Syllabus often has them earlier.

3. **Stage HP-only files**: `pipeline_v4.find_pdfs()` only globs `tamu_data/raw/simple_syllabus`. For each term, copy any howdy_portal PDFs whose stems are NOT in the simple_syllabus tree over to it with `_HP` appended to the stem. This matches the existing `detect_source()` convention.

4. **Confirm scope with the user**: per-dept unique-stem counts per term. Future terms may be empty.

5. **Pilot first** — see Pilot section below.

## Pilot First, Always

Before processing the full corpus, run **Phase B on 10 PDFs** via `--limit 10`. Inspect at every stop point. **Never run the LLM-heavy steps (image_recovery, validate, metadata_enrichment) on a corpus you haven't piloted.**

```bash
# Spend nothing first: copy + convert (no LLM)
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit 10 --steps copy,convert -y

# HARD GATE: count files that would actually trigger Gemini before running image_recovery
for f in data/syllabi/<DEPT>/bronze/<term_code>_<DEPT>_*.md; do
  n=$(grep -c '<!-- image -->' "$f")
  [ "$n" -ge 2 ] && echo "$f $n"
done | wc -l
# Then ask the user before running this:
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit 10 --step image_recovery -y

# Rule-based filters (free)
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit 10 --steps false_positive,boilerplate,hierarchy -y

# LLM validate (1-2 calls/file, ~25s each)
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit 10 --step validate -y

# Enrich (NOT a pipeline_v4 step — call CLI directly; 2 LLM calls/file)
# Stage just the pilot files into a temp dir so the standalone CLI only sees them:
TMPIN=$(mktemp -d)
for f in data/syllabi/<DEPT>/silver/04_hierarchy/<term_code>_<DEPT>_*.md; do
  ln -sf "$(realpath "$f")" "$TMPIN/$(basename "$f")"
done
python -m tamubot.ingestion.filters.metadata_enrichment "$TMPIN" data/syllabi/<DEPT>/silver/05_enrich
rm -rf "$TMPIN"

# Chunk + gold (no LLM)
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit 10 --steps chunk,gold -y
```

**Pilot does NOT include ingest.** Ingest only after the full corpus is processed — partial CSCE data in shared Atlas collections is undesirable.

## Step 1 — Generate the Report (no LLM cost)

The report is an aggregator over existing logs + per-file JSONs. Rebuild anytime without spending API calls:

```bash
python scripts/build_csce_pilot_report.py --dept <DEPT>
# Output: data/syllabi/<DEPT>/silver/docling_pipeline_v4_report.xlsx
```

The script handles transition (reads both `data/syllabi/<DEPT>/logs/` and shared `data/syllabi/logs/`, dedupes by file keeping the latest timestamp). It matches the ISEN report layout exactly:

- **Summary** (58 cols): File, Course Type, Term, Course, Section, Tokens In/Out, Sections Stripped, v1-v8 validation counts, image_recovery + FP + BP + hierarchy details
- **Content Pres. / Strip Compl. / Structural / Metadata** (17 cols each): per-file v1-v8 Count + Findings
- **Stripped Headers** (6 cols): per-file BP detail with `[TYPE] header text (chars)` bullets
- Dark-maroon header fill (`#500000`) on Summary + findings; blue (`#4472C4`) on Stripped Headers; freeze panes A2 everywhere

## Step 2 — Analyze the Report

Open the Summary sheet. Look at v1 Total per file. Color hints: green = 0, yellow = 3+, red = 8+. Then drill into the per-category sheets for the failing files.

Group findings by root cause:
- **Registry gaps**: boilerplate header not in `BOILERPLATE_REGISTRY` → add entry.
- **Tab-separated headers**: PDF used tab chars in header text (CSCE 625_601 case). Either patch the file or run the filter chain so headers get normalized downstream. The image_recovery (Gemini) output already normalizes whitespace — re-running filters from `silver/01_image_recovery` is usually enough.
- **Docling layout artifacts**: textbook info placed under "Grading Policy" instead of "Textbook" (textbook-cover image disrupted layout). Not pipeline-fixable; flag as known.
- **Image-marker hallucinations**: Gemini may invent a textbook-cover *description* (e.g. "An image of the textbook cover for…"). Caught by `content_preservation` findings. Add a post-processing strip if persistent.
- **Tab-header sections**: ones the boilerplate registry would normally catch but doesn't because of whitespace. Fixed by re-running from silver/01_image_recovery (Gemini-cleaned), not from bronze.

## Step 3 — Apply Fixes

Common fix patterns:

**Boilerplate registry** (`boilerplate_stripper.py`):
```python
"TAMU_POLICY": [..., "new header text"]
```
or `BODY_BOILERPLATE_HEADERS` for non-`#` headers.

**False positive** (`filters/false_positive.py`):
- `_EXACT_FP` for exact-match strings
- `_PREFIX_FP` for prefix matches
- `_REGEX_FP` for patterns

**Header regex** (`_HEADER_RE`): already uses `^\s*` to allow indented `### Header`.

## Step 4 — Re-run Only Affected Files

The filter scope now honors `config["file_pattern"]` and `config["limit"]`. Use them. **Never let a re-run touch every file in bronze** (the pre-fix behavior would reprocess all 89 ISEN files when you only wanted CSCE).

```bash
# Re-run just the affected stages, scoped via --department and --limit
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit N \
    --steps boilerplate,hierarchy -y
# Then re-validate (LLM cost):
python -m tamubot.ingestion.pipeline_v4 --department <DEPT> --term "<Term>" --limit N --step validate -y
# Then rebuild the report (no cost):
python scripts/build_csce_pilot_report.py --dept <DEPT>
```

Validation entries are dedupe-by-timestamp in the report builder — old runs are automatically superseded by new ones.

## Step 5 — Ingest to MongoDB Atlas

After the full corpus is chunked AND the report looks clean:

```bash
make ingest-dept DEPT=<DEPT>
# == python -m tamubot.ingestion.ingest --department <DEPT> --v4
```

`ingest.py --v4 --department X` checks `data/syllabi/<DEPT>/silver/06_chunk/` first, falls back to shared `data/syllabi/silver/06_chunk/` for legacy ISEN. Verify in Atlas:
```js
db.chunks_v4.countDocuments({course_id: /^<DEPT>/})  // > 0
db.courses_v4.countDocuments({course_id: /^<DEPT>/}) // == expected file count
```

## Known Bugs (Fixed) — Don't Re-introduce

These were live as of CSCE pilot and are now patched. If you see regressions, check these:

1. **`_find_raw_pdf` version mismatch** — `filters/image_recovery.py:63`. Bronze md files are `_v012.md` after convert bumps the version, but raw PDFs from `step_copy` are `_v011.pdf`. The function now falls back to `glob(f"{base_stem}_v???.pdf")` and picks the highest. Without this, image_recovery silently passes through every file (no Gemini calls, no error printed except `WARN: No raw PDF`).

2. **Filter input_dir fallback** — `pipeline_v4._input_dir_for_step`. When a filter or validate step is run alone (not in a multi-step invocation), the function used to hard-fall-back to `BRONZE_ROOT`, ignoring any silver outputs from prior runs. It now walks back through `silver/04_hierarchy` → `03_boilerplate` → `02_false_positive` → `01_image_recovery` and picks the latest one with files matching `file_pattern`. **Symptom of regression**: re-running validate shows it flagging boilerplate that's clearly already stripped (because it's reading bronze).

3. **Filters globbing every `.md` in bronze** — `filters/*/apply()`. Each filter's `apply()` now respects `config["file_pattern"]` (`{term_code}_{DEPT}_*.md`) and `config["limit"]`. `pipeline_v4.run_pipeline` builds and passes the pattern. **Symptom of regression**: ISEN files get touched/overwritten when you only meant to process CSCE.

4. **Report writes silently drop new files** — `report_writer._find_row` returns None for unknown file stems and every `update_*` function exits early. New departments don't get rows added to the legacy report. Use `scripts/build_csce_pilot_report.py` instead — it builds a fresh per-dept report from logs + JSONs.

## Gotchas

- **Validate variance**: re-validating the same unchanged file produces +/-1-2 in findings counts. Don't chase it.
- **enrich is not a pipeline_v4 step**: it has its own CLI under `tamubot.ingestion.filters.metadata_enrichment`. Stage CSCE-only files into a temp dir before invoking, or it processes everything in input_dir. Output goes to `data/syllabi/<DEPT>/silver/05_enrich/`.
- **`--limit` only applies via `find_pdfs` for copy/convert** historically; now also threaded through filter config and validate. If you run a step alone with `--limit N`, it works correctly.
- **Source data vs pipeline issues**: image-based course schedules, empty Meeting Days for asynchronous-online sections, registrar 691 "Research" templates — these are source data limitations and won't be fixed by pipeline tweaks.
- **Always invoke task-budget skill** before any step that calls `image_recovery`, `validate`, or `metadata_enrichment` on more than ~10 files. Estimate: ~5 Gemini calls + ~1-2 LLM calls/file for validate + ~2 LLM calls/file for enrich. A 100-file pilot can easily hit 400-600 model calls.

## Page mapping — Docling sidecar (NOT PyMuPDF)

The canonical header→page mapping comes from the Docling sidecar
`<stem>.headers.json` emitted alongside the bronze markdown. Format:
```json
[{"text": "Course Schedule", "level": 1, "page": 6}, ...]
```

`docling_converter.convert()` writes this automatically at convert time.
`metadata_enrichment.enrich_file()` reads it via `_load_headers_sidecar()` and
falls back to PyMuPDF text-matching only if no sidecar exists.

**Why this matters**: PyMuPDF text-matching breaks on line-wrapped or
table-cell headers (e.g. "Week 1 - Course Overview and Shape as Structure"
when the PDF shows it as "Week 1" then "Course Overview..."). The original
CSCE pilot saw 16 false `page: null` findings on a single file from this.
Docling's `item.prov[0].page_no` is authoritative — every header gets the
right page.

**Repairing files missing a sidecar** (e.g. converted before this fix):
```bash
python scripts/extract_docling_headers.py --dept <DEPT>     # write sidecars
python scripts/refresh_with_sidecar.py --dept <DEPT>        # refresh enrich + re-chunk
```

`extract_docling_headers.py` defaults to Gemini-skipped files (those whose
markdown matches Docling's items 1:1). For Gemini-recovered files, keep
PyMuPDF fallback — the Gemini-inserted lines aren't in Docling's items.

## LLM robustness — retry on empty results

The TAMU LLM proxy (`protected.gemini-2.5-flash`) occasionally returns `[]`
or malformed JSON non-deterministically, even at `temperature=0.0`.
`generate_summary_statements` in `metadata_enrichment.py` retries once
on empty result or parse error. Apply the same pattern (one retry, log
the WARN, accept second-attempt result if non-empty) for any function
returning a JSON list/object from this proxy. Observed: 18/30 files
returned `[]` on first call during one batch; 0/30 after retry was added.

## Fail-loud on critical-path imports

**Never** wrap a pipeline-critical import in `try/except ImportError: X = None`
to "fix" a missing module. The pattern silently produces worse output (e.g.
flat-H1 markdown when `hierarchical.postprocessor` is unavailable) while the
pipeline still completes and spends API budget. Let ImportError propagate so
the operator notices before spending. See `feedback_fail_loud_critical_imports.md`.

Optional-import is fine only for genuinely optional features (telemetry,
dev-only helpers) that don't shape downstream output.

## Container deps must live in requirements.txt

`pip install <pkg>` inside a running container is wiped on the next rebuild.
Docling was lost twice during CSCE work because it wasn't pinned. **If a
module is needed by the pipeline, it goes in `requirements.txt`** (currently
the only authoritative dep list — `pyproject.toml` is mostly empty). Verify
after edit: `grep <pkg> requirements.txt && pip show <pkg>`.

## Manual refinement via parallel subagents

When validation surfaces ≤10 files with non-systemic issues (broken tables,
wrong heading levels, merged-instructor-field, idiosyncratic boilerplate),
dispatch one subagent per file in parallel rather than fixing them
sequentially or expanding the boilerplate registry for one-off patterns.
Each subagent:

1. Reads `05_validate/<stem>_validation.json` to know what to fix
2. Reads `04_hierarchy/<stem>.md` (the LLM validator's input)
3. Applies surgical `Edit` calls
4. Calls `validate_directory(..., file_pattern=f"{stem}.md")` (1 LLM call)
5. Reports findings before/after

**Each subagent budget: 1 LLM call.** Dispatch 4-5 in parallel; serialize
batches if rate-limited. Per-file independence means no shared-state
conflicts. After subagents finish: refresh enrich headers via sidecar
(`refresh_with_sidecar.py`), then re-chunk and final-validate.

## After manual refinement → refresh downstream artifacts

Editing `04_hierarchy/<stem>.md` invalidates the downstream `05_enrich`
(stale `headers` list with removed/renamed entries) and `06_chunk` (stale
chunks/summary_statements). Don't trust either until refreshed.

**Cheap refresh path** (no LLM calls for header mapping):
1. Sidecar already exists → `_build_headers_from_sidecar(markdown, sidecar)`
   regenerates the `headers` field in `05_enrich/<stem>.json`
2. Re-run chunk step: `chunk_semantic(markdown, flag_threshold=600, min_chunk_tokens=50)`
3. Re-run `generate_summary_statements` (1 LLM call per file)
4. Re-validate (1 LLM call per file) to confirm `page: null` complaints clear

Use `scripts/refresh_with_sidecar.py` — it does steps 1-3 in one pass.
Use `scripts/final_validate_sidecar.py` for step 4.

**Chunk config**: always `flag_threshold=600, min_chunk_tokens=50` (pipeline
defaults). Using higher min (e.g. 120) collapses sections under the threshold
into one chunk — a 1-chunk-per-file file is the symptom.

## Validation findings: real vs noise

Re-validation after refinement often surfaces "metadata_accuracy" findings
that aren't real markdown issues. Common categories:

1. **Stale enrich JSON** (headers list still references removed sections).
   *Fix*: rerun header refresh via sidecar.
2. **Field-not-in-document-body** (validator complains section/CRN/term
   aren't in the markdown text, but they live in `course_metadata`).
   *Fix*: ignore — these are filename-derived fields, not body content.
3. **Truncated `course_summary`** (LLM cut off mid-sentence during enrich).
   *Fix*: re-run enrich for that file (~2 LLM calls).
4. **`syllabus_url` mismatch** (scraper-derived URL doesn't match a URL the
   document itself mentions). *Fix*: real issue if the document's URL is
   the canonical course site; usually safe to leave.

The validator was designed for content quality, not metadata-schema audit.
Treat metadata findings critically — many are noise.
