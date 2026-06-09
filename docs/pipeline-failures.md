# Pipeline failures — ratchet log

Append-only history of **real observed** v6b preprocessing failures + their fixes. Every
algorithm change in the improvement loop must trace to an entry here (`iterate-preprocessing`
skill). Taxonomy (`preprocessing_error_taxonomy.md`) is the closed vocabulary; this is the
open history. Newest first; one entry per distinct failure mode.

## Format

```
### YYYY-MM-DD · <TITLE> · <TYPE>
  impact: <what breaks downstream / for the user>
  fix:    <one-line change + owning file>   (or: open)
  check:  <deterministic check that now catches it>   (omit if none)
  status: ✅ fixed | 🔲 open | ⏸ wontfix  ·  <stems · verify/result>
```

`impact` = why it matters (the future cost). `check` = how it surfaces programmatically now
(no LLM judge). Keep each line to one line.

## Entries

### 2026-06-09 · Docling drops PDF link-annotation URL lines · FID_CONTENT_DROPPED
  impact: "link to the instructor page / Canvas" returns nothing; no source to cite
  fix:    `_append_orphan_urls` inserts a dropped-URL line as a geometry-placed text block (`docling_block_adapter.py`)
  check:  `text_coverage` — dropped URL token shows in `sample_missing`
  status: ✅ fixed  ·  ECEN_738_601 + CSCE_629_600 FAIL→PASS (iter_06) · tests TestUrlRecovery×9

### 2026-06-09 · Docling drops right-column label values (ISBN) · FID_CONTENT_DROPPED
  impact: "what textbook / which ISBN" returns the label without the value — looks answered, isn't
  fix:    `_recover_dropped_label_values` re-attaches the dropped tail from the PyMuPDF line, gated on a token missing from bronze (`docling_block_adapter.py`)
  check:  `text_coverage` — dropped value token shows in `sample_missing`
  status: ✅ fixed  ·  ECEN_719 `ISBN: 978-0-387-69957-8`; cov 0.997→0.999 · tests TestLabelValueRecovery×5

### 2026-06-09 · TableFormer under-captures table rows/cells · FID_TABLE_LOST
  impact: "when is the midterm / what average is an A" misses a dropped row — confidently wrong on the most-asked questions
  fix:    `_recover_dropped_table_cells` merges PyMuPDF `find_tables` into the Docling grid, add-only (insert dropped rows, upgrade truncated cells); never touches good cells (`docling_block_adapter.py`)
  check:  `table_cell_capture` (WARN) — tokens find_tables has but the grid dropped; text_coverage misses these (tokens recur in body)
  status: ✅ fixed  ·  ECEN_721 `1st MIDTERM|Mar.3` row; ISEN_625 ×3 cell tails; cov 0.995→0.998 · tests TestTableCellRecovery×8 · iter_08 re-judge: 5 table stems clean (STAT_652, ECEN_749×2 too)

### 2026-06-09 · Two-column course-info read column-first, values orphaned · FID_HEADER_BROKEN
  impact: room/time/section/credit retrieved as a bare `Location:`; bare-label embeddings near-useless — degrades retrieval corpus-wide
  fix:    `_recover_two_column_values` re-pairs label↔value by PyMuPDF y-band geometry, merges value into the label, deletes the non-adjacent orphan (`docling_block_adapter.py`)
  check:  `no_orphaned_labels` (WARN) — short `Label:` blocks with no value beside them
  status: ✅ fixed  ·  ECEN_671 all 6 Course-Info values re-paired · tests TestTwoColumnRecovery×5 · iter_08 re-judge: ECEN_671/749×2 labels verified paired

### 2026-06-09 · iter_08 re-judge (30 stems) confirms the 4 extraction-loss fixes
  impact: re-baseline after URL/ISBN/table/two-column recovery — proves no regressions corpus-wide
  result: boilerplate 1.00 · dedup 1.00 · chunking 1.00 · fidelity 0.80→**1.00** (6 stems FAIL→PASS, 0 regressed, p=0.031)
  residual: FID_IMAGE_LOST minor ×8 (decorative covers/logos — modal disabled, expected); ECEN_749_612 minor lab-time sub-label drop (value survives)
  status: ✅ confirmed  ·  iter_08_recovery vs iter_05_3e43f07 · 30/30 stems pass all dimensions

### 2026-06-08 · Small leading metadata block dropped by tiny-chunk floor · FID_CONTENT_DROPPED
  impact: a terse top "Course Information" block (time/location/cancelled dates) absent from RAG
  fix:    `chunk_semantic` merges a sub-floor leading section FORWARD instead of dropping it (`chunker_v4.py`); `dropped_tiny` now unreachable
  status: ✅ fixed  ·  STAT_620, CSCE_685_626/_326 — leading block now in the retained chunk · tests in test_v6b_parsing_qc_fixes.py

### 2026-06-08 · Long schedule table truncated by VLM, shadows fuller grid · FID_TABLE_LOST
  impact: 15-week schedule shows only first 3–5 weeks in PROCESSED — most rows silently gone
  fix:    row-count conservation in `render_table_markdown` — accept VLM table only if rows ≥ 0.8× grid, else use the grid (`validation/table_quality.py`)
  check:  `vlm_truncated_tables` counter; `v6b_silver_modal_no_table_lost`
  status: ✅ fixed  ·  CSCE_611 →8 rows, CSCE_682 →15 rows · cache-version-salt + tiling deferred

### 2026-06-07 · Chunker dropped front-matter metadata sections · CHUNK_* (metadata loss)
  impact: instructor contact, meeting times, credit hours, prerequisites missing from chunks
  fix:    `chunk_semantic` gained `drop_metadata_sections` (default True; v6b passes False) (`chunker_v4.py`, `assets/silver_chunk_semantic.py`)
  status: ✅ fixed  ·  CSCE_676/ECEN_749/STAT_620/CSCE_633 — sections now chunked

### 2026-06-07 · VLM table repetition-loop → silent total table loss · FID_TABLE_LOST
  impact: a schedule table emitted as `[table processing failed]`, ALL rows lost
  fix:    `presence_penalty=1.5` on http table decode + `render_table_markdown` degradation ladder keeps a partial under an unverified marker (`clients/nuextract_http_client.py`, `assets/silver_modal.py`)
  check:  `v6b_silver_modal_no_table_lost` / `_no_degenerate_tables` (WARN)
  status: ✅ fixed (P1/P2/P4)  ·  ECEN_683, CSCE_642 recovered · guided_json (P3) deferred

### 2026-06-07 · Heading level-skip aborted whole document at bronze gate · FID_HEADER_BROKEN
  impact: a single H2→H4 skip dropped the ENTIRE syllabus (silver never ran), not just its breadcrumb
  fix:    unconditional stack-based `normalize_heading_levels` (records `raw_level`, re-levels skip-free); downgraded `header_hierarchy_valid` to WARN (`docling_block_adapter.py`, `validation/header_hierarchy.py`)
  check:  `header_levels_normalized` (WARN, records `repaired_skip_count`); `min_headers`
  status: ✅ fixed  ·  ISEN_667/665, CSCE_704 no longer dropped

### 2026-06-02 · Tables dropped before chunking in default modal mode · FID_TABLE_LOST
  impact: with `V6B_MODAL_ENABLED=false` (default) the Docling `table_body` never reaches the chunker — table content silently absent
  fix:    open — render `table_body`→GFM in `_merge_to_markdown` (`assets/silver_modal.py`). See v6b-preprocessing-review.md H1
  status: 🔲 open  ·  any grading/schedule table

### 2026-06-02 · silver_tag served stale/empty index from path-only cache · BP_MISSED / DUP_MISSED
  impact: a same-path index rebuild (two-phase workflow) wasn't picked up → 0 boilerplate / 0 dedup
  fix:    cache key now `(path, sha256)` (`assets/silver_tag.py`). See v6b-preprocessing-review.md H2
  status: ✅ fixed  ·  behavior-only; re-run tag to repopulate

### 2026-06-02 · Atlas upsert key collides across same-CRN sources · (ingest integrity)
  impact: `_HP` vs non-HP copy of one CRN overwrites (last-writer-wins); count check can flap
  fix:    open — add `source_file` to the upsert key + align the count check, before `V6B_INGEST_ENABLED=true` (`assets/silver_atlas_upsert.py`). See review H3
  status: 🔲 open  ·  low live impact (ingest dry-run by default)

### 2026-06-02 · Boilerplate reference under-populated (only CSCE+STAT) · BP_MISSED
  impact: TAMU university-policy boilerplate not flagged (rate 0%) → stays visible to RAG or shows as within-doc duplicate
  fix:    open — materialize ≥3 departments, then rebuild `meta_boilerplate_reference` (data coverage; `util/boilerplate_clustering.py` gate)
  status: 🔲 open  ·  CSCE_608/611 flagged by 2 sub-agent judges
