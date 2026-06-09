# Pipeline failures — ratchet log

Append-only history of **real observed** v6b preprocessing failures and the fixes they
drove. Every algorithm change in the improvement loop must trace to an entry here (see the
`iterate-preprocessing` skill). The error taxonomy (`preprocessing_error_taxonomy.md`) is
the closed *vocabulary*; this file is the open *history*.

Newest first. One entry per distinct failure mode (not per syllabus instance).

## Template

```
### YYYY-MM-DD · <SHORT_TITLE> · <ERROR_TYPE>
- run: iter_<NN>_<sha>            stems: <example stems, 2-3>
- observed: <what the judge/human saw, with evidence — PDF page / chunk index>
- confirmed: <PDF-confirmed? recall@5 impact? anchor agreement?>
- owner: <code file from taxonomy>
- fix: <change made, or "open / not yet fixed">
- result: <post-fix paired-comparison delta; outside noise band? regressions?>
- status: open | fixed | wontfix (reason)
```

## Entries

### 2026-06-09 · Docling drops PDF link-annotation URL lines · FID_CONTENT_DROPPED
- run: iter_05_3e43f07 (30-stem judge)    stems: ECEN_738_601, CSCE_629_600, ECEN_719_601
- observed: instructor `Webpage:`/`Biography:` URLs and a `check daily: https://canvas.tamu.edu/`
  line are visible in the source PDF but absent from PROCESSED *and* ORIGINAL (bronze md) —
  top error class in iter_05 (FID_CONTENT_DROPPED blocker, n=5/4 stems; fidelity 0.80).
- confirmed: PDF-confirmed by sub-agent judges AND a raw-Docling diagnostic — the URLs are
  link **annotations** (PyMuPDF `page.get_links()` has them) but Docling never emits a text
  item for the line, so the loss precedes all our postprocessing. Our existing
  `_recover_urls_by_page` recovered the URL but `_inject_urls_into_blocks` only *wrapped*
  link-text already in a block, discarding any with no host block.
- owner: `converters/docling_block_adapter.py`.
- fix: fixed — added tier-2 `_append_orphan_urls`: a recovered URL with no host block is
  inserted as a new text block, placed by **vertical geometry** (`_orphan_url_insert_index`:
  PyMuPDF link-rect centre vs Docling bottom-origin block tops, `top_from_top = page_height - t`)
  so it lands where it sits on the page instead of being dumped at the page end. Prefers the
  clean visible URL when the annotation's uri target is malformed (CSCE_629's PDF had a typo'd
  href). Unit-tested (`tests/test_docling_block_adapter.py::TestUrlRecovery`, 9 cases incl. 2
  geometry-placement).
- result: re-judged (iter_06_urlfix) — **ECEN_738_601 + CSCE_629_600 fidelity FAIL→PASS**.
  ECEN_719_601 still fails but only on the open ISBN-value class below; its first-pass fix put
  the URL under `ISBN:` (judge flagged corruption), which the geometry placement then moved to
  the instructor area (verified in bronze: URL now between Email/image and Catalog Description).
- status: fixed (URL/hyperlink sub-class only)

### 2026-06-09 · Docling drops right-column label values + table cells/rows · FID_CONTENT_DROPPED / FID_TABLE_LOST
- run: iter_05_3e43f07 (30-stem judge)    stems: ECEN_719_601 (ISBN value), ISEN_625_601 / ECEN_721_700 (table)
- observed: label survives but trailing value dropped (`ISBN:` present, `978-0-387-69957-8`
  gone; `Authors:`/`Publisher:` values gone); two-column Course-Info block split
  (ECEN_671_600); Course-Schedule table trailing cells / a whole `1st MIDTERM | Mar. 3` row
  dropped. All identical-missing in ORIGINAL+PROCESSED → Docling extraction.
- confirmed: raw-Docling diagnostic — value cells absent from `result.document`; the data IS
  in PyMuPDF `page.get_text()`. Table cells come through with empty leading cells (TableFormer
  under our `do_cell_matching=False` config). Web research: label:value/two-column is a known
  Docling reading-order class (no config fix); table under-capture maps to docling #2064.
- owner: `converters/docling_block_adapter.py` (label-value recovery — DONE), `converters/docling_converter.py`
  (`do_cell_matching` A/B for tables — open).
- fix: **label-value class fixed** — `_recover_dropped_label_values` / `_extend_labels_with_values`
  re-attaches a dropped value to its surviving `Label:` block from the PyMuPDF visual line, only
  when the tail carries a content token missing from bronze (incl. table grids) so a kept value is
  never duplicated. Unit-tested (`TestLabelValueRecovery`, 5). Verified: ECEN_719 `ISBN:` →
  `ISBN: 978-0-387-69957-8`, coverage 0.9971→0.9985. **Table cell/row class still open** — needs a
  TableFormer `do_cell_matching` A/B (riskier; the new `text_coverage` check now surfaces these
  drops in `sample_missing`).
- result: ECEN_719 ISBN re-attached; re-judge pending. ISEN_625 / ECEN_721 table drops untouched.
- status: fixed (label-value class); table class open

### 2026-06-08 · Small leading metadata block dropped by tiny-chunk floor · FID_CONTENT_DROPPED
- run: iter_02_f29d74a (10-stem judge)    stems: STAT_620
- observed: after the metadata-keep fix, STAT_620's larger "Instructor Details" section is now
  retained (real improvement vs baseline, which began at "Textbook"), but a *separate* terse
  "Course Information" block ("Time: Tue/Th 12:45–2:00", "Location: BLOCKER 411", "Cancelled
  classes: no class 01/13/2026") is still absent from PROCESSED. Judge flagged FID_CONTENT_DROPPED.
- confirmed: NOT the metadata bug — `chunk_semantic(...drop_metadata_sections=False)` reports
  `dropped_metadata=0`. The block is only ~34 tokens, below `MIN_CHUNK_TOKENS=50`, so it is swept by
  the tiny-chunk rule; because it sits at the top with the (also-tiny) title sections, there is no
  prior chunk to merge into, so it is dropped rather than merged. Owner: `chunker_v4.py`.
- owner: `chunker_v4.py` (`_process_node` tiny-merge / root-leading handling).
- fix: FIXED 2026-06-08 — `chunk_semantic` now merges a sub-floor leading section FORWARD into the
  next real chunk (buffered in `pending_forward`, flushed on the next `_emit`) instead of dropping
  when there is no previous chunk; a doc that is entirely sub-floor still emits one chunk. The
  `dropped_tiny` path is now unreachable (stays 0); new `merged_forward` counter. Mirrors
  Unstructured's `combine_text_under_n_chars` + Docling issue #1174 (research:
  `docs/research/tiny-leading-chunk-qc.md`). Tests in `test_v6b_parsing_qc_fixes.py`.
- result: re-ran CSCE_685_626 + CSCE_685_326 (same failure mode, 202611/202621) → leading
  "Course Information / Instructor Details / Catalog Description / Credit Hours / Walker" now present
  in the retained RAG chunk (`merged_forward`=12–13, `dropped_tiny`=0); judge re-run confirms.
- status: fixed

### 2026-06-08 · Long schedule table truncated by VLM, shadowing fuller Docling grid · FID_TABLE_LOST
- run: iter_03_e126d36 (20-stem judge)    stems: CSCE_611, CSCE_682
- observed: 15-week "Course Schedule" tables show only the first ~3–5 weeks in PROCESSED. The table
  is a single intact chunk (chunker is NOT fragmenting it) but most rows are gone — loss is at
  modal extraction, not chunking.
- confirmed: the VLM `table_markdown` is a CLEAN but truncated 1–4 rows (served `from_cache=True`,
  i.e. a pre-`presence_penalty` decode), while Docling's `table_body` grid in bronze holds the full
  table (CSCE_682 block#53: grid 14 rows vs VLM 0; CSCE_611 block#88: grid 7 vs VLM 1). The render
  ladder's tier-1 kept any non-*degenerate* VLM table, so a clean-but-short transcription shadowed
  the fuller grid. Truncation ≠ repetition, so `is_degenerate_table` did not catch it.
- owner: `validation/table_quality.py` (`render_table_markdown`); surfaced by `assets/silver_modal.py`.
- fix: FIXED 2026-06-08 — row-count conservation in `render_table_markdown`: accept the VLM table
  only if its data-row count is ≥ 0.8× the grid's, else fall back to the (fuller) Docling grid.
  New `vlm_truncated_tables` counter in `summarize_table_outcomes`. Uses data already in bronze →
  no GPU re-extraction needed. Research: `docs/research/long-table-preservation-qc.md`. Tests in
  `test_v6b_parsing_qc_fixes.py`.
- result: re-ran CSCE_611 (3+4 → 8 table rows) + CSCE_682 (4+1 → 15 rows, full schedule); the
  `v6b_silver_modal_no_table_lost` check passes; judge re-run confirms.
- followups (deferred): version the content-keyed table cache (`modalprocessors._cache_key`) with a
  decode-version salt so future decode changes invalidate stale entries; tiling / `max_tokens` for
  tables whose grid is *also* truncated (none observed here). See research doc §3/§6.
- status: fixed

### 2026-06-07 · Chunker dropped front-matter metadata sections in v6b · CHUNK_* (metadata loss)
- run: judge iter (10-stem _review)    stems: CSCE_676, ECEN_749, STAT_620, CSCE_633
- observed: instructor contact, meeting times/location, credit hours, prerequisites missing from
  PROCESSED chunks — the 03_tag chunks started at "Course Description"; the sections were in the
  bronze md but in NO chunk (not in why.json → not boilerplate/dedup; dropped by the chunker).
- confirmed: source-traced — `chunker_v4._METADATA_HEADERS_LOWER` + `_is_metadata_section` +
  `_process_node` deliberately drop these (a v4 assumption: they live in `course_metadata`). In
  v6b nothing re-injects them (`silver_chunk_semantic` sets `instructor_name=None`; the
  `silver_structured` branch is parallel, never merged into retrieval).
- owner: `chunker_v4.py`, `assets/silver_chunk_semantic.py`.
- fix: fixed — `chunk_semantic` gained `drop_metadata_sections` (default True → v4/v5 unchanged);
  v6b passes `False` so metadata sections are emitted as ordinary chunks. ~6 lines.
- result: re-chunked the 4 stems — instructor/email/meeting/prereq now present in chunks
  (header_paths "Course Information"/"Instructor Details"/"Course Prerequisites" retained).
  Chunk count +1..+2/doc; one-time WARN baseline-delta on flagged_rate (expected re-settle).
- status: fixed

### 2026-06-07 · VLM table repetition-loop → silent total table loss · FID_TABLE_LOST
- run: judge iter (10-stem _review)    stems: ECEN_683 (table_000), CSCE_642
- observed: schedule table emitted as `<!-- table: [table processing failed: no JSON object… ] -->`,
  ALL rows lost. The VLM degenerated into an unbounded `| | |` run, hit the 1024 token cap, produced
  unterminated JSON → ValueError; the 0.5-confidence partial text was then discarded by the merge and
  Docling's grid had only 2 rows. The same doc's other table parsed at confidence 1.0.
- confirmed: live repro on the saved crop — `presence_penalty=0.0` → `finish_reason=length`, JSON
  parse-fail, 0 rows; `presence_penalty=1.5` → `finish_reason=stop`, valid JSON, 3 rows. The dense
  sibling table is byte-identical at pp 0.0 and 1.5 (no corruption — table-safe caveat holds).
- owner: `clients/nuextract_http_client.py`, `assets/silver_modal.py`.
- fix: fixed — (P1) `presence_penalty=1.5` on the http table decode path only (temperature stays 0;
  repetition_penalty deliberately NOT used — corrupts repeated cells). (P4) `_merge_to_markdown` now
  resolves tables through `validation/table_quality.render_table_markdown` ladder
  (VLM→grid→kept-partial→lost), keeping a degraded partial under a `<!-- table (unverified) -->`
  marker instead of dropping it. (P2) new `v6b_silver_modal_no_table_lost` /
  `_no_degenerate_tables` WARN checks + `tables_lost`/`degenerate_tables` metadata.
- result: see re-run — table_000 recovered to a real 3-row GFM table; survival gate observes 0 lost.
  Gates ship as WARN pending corpus calibration (then promote G3 to BLOCK). guided_json (P3) deferred.
- status: fixed (P1/P2/P4); P3/P5 deferred

### 2026-06-07 · Heading level-skip aborted whole document at bronze ERROR gate · FID_HEADER_BROKEN
- run: prior eval-corpus run    stems: ISEN_667, ISEN_665, CSCE_704 (3 docs dropped)
- observed: a single H2→H4 skip failed the blocking-ERROR `header_hierarchy_valid` check → the
  partition aborted → silver never ran → the entire syllabus was dropped (not just its breadcrumb).
- confirmed: source-traced — `bronze_blocks_checks.py` wired the check `blocking=True`/ERROR and
  `check_header_hierarchy_valid` is all-or-nothing (`skip_count==0`). The skip is manufactured upstream
  by 3 independent level paths (numbering depth, font-cluster rank, safety-net demote-to-H3).
- owner: `converters/docling_block_adapter.py`, `checks/bronze_blocks_checks.py`.
- fix: fixed — added pure stack-based `normalize_heading_levels` (`validation/header_hierarchy.py`),
  called unconditionally in `docling_to_blocks` after `_recover_heading_levels` (records `raw_level`,
  re-levels to skip-free tree depth, never drops/reorders). Downgraded `header_hierarchy_valid` to
  WARN/non-blocking (now a post-condition assertion); added `header_levels_normalized` WARN
  (records `repaired_skip_count`) + wired `check_min_headers` (catches the flat-doc case no-skip can't).
- result: re-materialized bronze for all 3 — each completed (no longer dropped); raw skip=1 preserved
  in `raw_level`, normalized skip=0. Mis-repair cost is at most a slightly-off breadcrumb (traced),
  never a lost boundary/content — strictly dominates dropping the doc.
- status: fixed

### 2026-06-02 · Tables dropped before chunking in default modal mode · FID_TABLE_LOST
- run: code-review (static)    stems: any with a grading/schedule table
- observed: `silver_modal._merge_to_markdown` renders only `modal_result.table_markdown`;
  with `V6B_MODAL_ENABLED=false` (default) `modal_result` is absent, so the persisted
  `table_body` (Docling cell grid) never reaches the chunker — and the adapter already
  dropped the table's cell TextItems. Net: table content silently absent from chunks.
- confirmed: source-read; `docling_block_adapter.py:418-441`, `silver_modal.py:106-116`.
  `silver_modal_checks` cannot catch it (only validate blocks that have a `modal_result`).
- owner: `assets/silver_modal.py` (FID).
- fix: open — render `table_body` → GFM in `_merge_to_markdown`; re-materialize
  `silver_modal → atlas_upsert` (no re-Docling). See `docs/v6b-preprocessing-review.md` H1.
- result: —
- status: open

### 2026-06-02 · silver_tag served stale/empty index from path-only cache · BP_MISSED / DUP_MISSED
- run: code-review (static)    stems: any tagged in a long-lived process after an index rebuild
- observed: module-level reference/signature caches keyed by file *path* only; a same-path
  parquet rebuild (the two-phase workflow) was not picked up → 0 boilerplate / 0 dedup.
- confirmed: source-read; `assets/silver_tag.py:36-71` (computed `hash_file` but never compared).
- owner: `assets/silver_tag.py`.
- fix: fixed — cache key now `(path, sha256)`. See `docs/v6b-preprocessing-review.md` H2.
- result: behavior-only; re-run tag to repopulate.
- status: fixed

### 2026-06-02 · Atlas upsert key collides across same-CRN sources · (ingest integrity)
- run: code-review (static)    stems: a `_HP` and non-`HP` copy of the same CRN
- observed: upsert filter `{crn, chunk_index, chunk_tag}` ignores `source_file`; same CRN
  from Howdy vs Simple Syllabus overwrites (last-writer-wins). The blocking
  `vector_count_matches_chunks` check counts by `source_file`, so it can flap on collision.
- confirmed: source-read; `assets/silver_atlas_upsert.py:33`, `checks/silver_atlas_upsert_checks.py`.
  Low live impact today (ingest dry-run by default).
- owner: `assets/silver_atlas_upsert.py`.
- fix: open — add `source_file` to the upsert key + align the count check; land before
  `V6B_INGEST_ENABLED=true`. See `docs/v6b-preprocessing-review.md` H3.
- result: —
- status: open

### 2026-06-02 · Boilerplate reference under-populated (only CSCE+STAT materialized) · BP_MISSED
- run: _smoke    stems: 202611_CSCE_608_600_46648, 202611_CSCE_611_600_50668, 202611_CSCE_625_600_19180_HP
- observed: University-policy boilerplate is **not** flagged `is_boilerplate` (rate 0%);
  it either stays fully visible to RAG or surfaces as within-syllabus `is_duplicate`.
- confirmed: **two independent Claude Code sub-agent judges** flagged `BP_MISSED` (major)
  on CSCE_608 and CSCE_611, each citing the verbatim TAMU policy block (Academic Integrity,
  Title IX, ADA, FERPA, …) left in the processed view. Structural root cause:
  `meta_boilerplate_reference` gates on `distinct_depts >= 3`, but only CSCE+STAT are
  materialized, so no cluster qualifies. Not a tagging-logic bug.
- owner: data coverage (materialize ≥3 departments) before trusting `boilerplate_rate`;
  secondary: `util/boilerplate_clustering.py` gate constants.
- fix: open — materialize more departments, then rebuild `meta_boilerplate_reference`.
- result: —
- status: open
