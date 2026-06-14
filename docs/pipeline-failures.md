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

### 2026-06-09 · iter_09 full-100 baseline — 70-stem locked holdout judged · BASELINE
  impact: first fidelity number on the unseen 70; the dev-30 extraction-loss fixes held (iter_08 stays 1.00), the holdout exposes 9 new fails across 3 dimensions
  result: full-100 → boilerplate 0.99 · dedup 1.00 · chunking 0.99 · fidelity 0.93; holdout-70 alone → fidelity 0.90
  status: ✅ recorded  ·  iter_10_full100 (30 iter_08 + 70 iter_09) · code 9b77e0b · 9 open fails logged below

### 2026-06-09 · Orphan mailto recovery fabricates a `[label](mailto:)` block · FID_HALLUCINATION
  impact: RAG sees a synthesized contact line absent from the PDF — here a Zoom `?pwd=` fragment spliced into a fake email link; a hallucinated, citable-looking source
  fix:    APPLIED — `_append_orphan_urls` now skips a `mailto:` orphan whose bare address already appears as text (`docling_block_adapter.py`); regression introduced by ac3edc1. Unit test `test_skips_mailto_orphan_when_email_already_plain_text` (38 passed, 0 regr)
  status: 🟢 CONFIRMED end-to-end (rebuilt 2026-06-10)  ·  ISEN_633_600 — fabricated `[…](mailto:…)` link GONE from RAG-visible output; real `niosh@tamu.edu` retained; the genuine Zoom link `…?pwd=ooFUavFzL…` correctly survives in the visible TA section

### 2026-06-09 · Disabled-modal image fetch leaks errno string into RAG text · FID_TABLE_LOST / FID_IMAGE_LOST
  impact: `![[image processing failed: [Errno 111] Connection refused]](#)` literal shown in processed text; when the image was a content table the table is silently lost
  fix:    APPLIED — `silver_modal._merge_to_markdown` image branch guards `is_failure_marker(caption/description)` and degrades to a clean `<!-- image -->` (mirrors the table branch); never serializes the errno string (26 tests, 0 regr)
  status: 🟢 CONFIRMED end-to-end (rebuilt 2026-06-10)  ·  ISEN_665_600 — `processing failed`/`Errno` strings GONE from RAG-visible output, clean `<!-- image -->` present. NB the lost schedule TABLE itself (it was an image) still needs a GPU modal pass — only the errno-string LEAK is closed here; FID_IMAGE_LOST ×23 corpus-wide remains a modal-disabled artifact

### 2026-06-09 · New bronze line/row drops on the holdout · FID_CONTENT_DROPPED
  impact: lab time, "check daily: canvas.tamu.edu", seminar/final-project grading, a midterm row absent from ORIGINAL+PROCESSED — confidently-wrong answers to high-frequency questions
  fix:    per-stem triage done (4 distinct quirks, 2 families). ECEN_688 = VLM table transcription dropped a spanned row, slipped under the 80% row-conservation guard → owner is `validation/table_quality.py` NOT the adapter. CSCE_629/ECEN_749 = adapter URL/two-column recovery misses (need PyMuPDF visual-line threading). CSCE_704 = wholesale ~25-line region drop, NO existing detector fits.
  status: 🟢 2/4 CONFIRMED end-to-end · 🟡 1 partial · 🔲 1 deferred  (rebuilt 6 stems 2026-06-10, RAG-visible asserted)
          ✔ ECEN_688_602 CONFIRMED — `table_quality.render_table_markdown` rejects a VLM table missing any grid content token → grid fallback (30+15 tests). Visible chunk 7 now carries `| Mid-term exam: Thursday, October 15 th during class |`
          ✔ ECEN_749_602 CONFIRMED — new `_recover_orphaned_line_prefixes` (strict-suffix + novel-token + len floor; 47 tests). `505/605` + `12:20` now present
          🟡 CSCE_629_600 PARTIAL/INEFFECTIVE — code applied (`_recover_urls_by_page` 4-tuple full_line + `_append_orphan_urls` gated substitution, 41 tests) BUT end-to-end the canvas line is STILL lost: (a) the "Also, check daily:" prose did NOT recover because its tokens (check/daily) appear elsewhere in bronze → novel-token gate didn't fire (unit test used synthetic novel prose = FALSE POSITIVE); (b) the bare `https://canvas.tamu.edu/` IS recovered but tagged is_duplicate (cross-section dup of CSCE_629_600_54978#3) → hidden. Real blocker is cross-syllabus dedup, ties to the #4 over-hide work. Code is add-only/harmless (helps genuinely-novel URL-line prose elsewhere) but does NOT close this stem.
          🔲 DEFERRED CSCE_704_600 — wholesale ~25-line grading region drop. CONFIRMED unsafe for CPU/PyMuPDF region-recovery (2nd subagent attempt 2026-06-10 STOPPED, no code applied): page_idx 3 has NO text block in bronze — only an image block (60); the PDF page render is a blank/scrambled OCR-failure placeholder → the grading region is trapped in an un-OCR'd image. The region's anchors survived elsewhere as table cells (BlackBox/WhiteBox Defense on pp.4-5) + summary prose, and a 0.62-Jaccard near-twin of kept block 61 creates a duplication trap, so the contiguous-missing-run gate fragments into 7 disjoint runs with no same-page anchor. → recover ONLY via a GPU Docling/modal VLM re-pass on this stem (gpu-ops); not a host-adapter fix. Stays deferred.

### 2026-06-09 · CSCE_689 grading row dropped — MIS-DIAGNOSED as over-hide, actually FID_TABLE_LOST (render) · FID_TABLE_LOST
  impact: a unique grading-weight row present in ORIGINAL removed from PROCESSED — answer looks complete, isn't (table summed to 95%)
  triage: NOT a tagging over-hide — the "Async Progress Check-in Posts | 5%" row is is_boilerplate=false/is_duplicate=false; tagging never touched it. It's a separate 1-data-row CONTINUATION table that `render_table_markdown` treated as header-only (0 data rows) → OUTCOME_LOST → empty markdown, dropped before chunking. iter_09 verdict agrees: FID_TABLE_LOST.
  fix:    APPLIED — `table_quality.is_continuation_table` + `_merge_to_markdown` continuation-rescue: a same-width table scoring OUTCOME_LOST splices its rows into the preceding table, gated on rows' tokens missing from the rendered output (9+30+47 tests, 0 regr). Adapter untouched.
  status: 🟢 CONFIRMED end-to-end (rebuilt 2026-06-10)  ·  CSCE_689_700 — grading table now carries all 8 rows (sums 100%), "Async Progress Check-in Posts … 5%" appears once in RAG-visible output. NB a SEPARATE Phase-2 schedule header-row loss in the same doc remains (orphaned header row before a heading) — out of scope, not yet fixed.

### 2026-06-09 · Stranded campus-policy headers at chunk tail · CHUNK_ORPHAN_HEADER
  impact: body-less "Campus-Specific Policies / Texas A&M at Galveston" headers end a chunk — retrieval noise, weak embeddings
  fix:    APPLIED — `chunker_v4._process_node` drops a header whose entire subtree has no body (`_has_body_in_subtree`) before the sub-floor backward-merge can glue it onto the prev chunk's tail. Tests `test_orphan_header_not_stranded_at_chunk_tail` + `test_header_with_body_is_never_dropped_as_orphan` (28 passed, 0 regr)
  check:  orphan-header / `no_oversized` check should coincide — verify it fires on this stem
  status: 🟢 CONFIRMED end-to-end (rebuilt 2026-06-10)  ·  STAT_620_600 — no body-less orphan header in any RAG-visible chunk; no stranded `Campus-Specific Policies`/`Texas A&M at Galveston` trailing header

### 2026-06-09 · University-policy boilerplate left untagged · BP_MISSED
  impact: standard TAMU policy text surfaced as course content — dilutes retrieval, repeats across the corpus
  triage: CODE not data-coverage — the boilerplate reference already spans 4 depts (the old "CSCE+STAT-only" note is stale). STAT's policy chunks match their CORRECT cluster but at body-Jaccard far below `BP_THRESHOLD=0.80` (Makeup 0.688, Acad Integrity 0.414, ADA 0.312, Mental Health 0.094, Title IX 0.086) — genuine policy variant + OCR ligature loss ("Ofce", "confdentiality") shreds the 5-gram MinHash. A threshold drop is IMPOSSIBLE (real course content sits at 0.648 → false-positive cliff).
  fix:    APPLIED — header-anchored fallback in `flag_boilerplate`: a chunk whose normalized leading header matches a reference cluster present in ≥`HEADER_ALLOWLIST_MIN_DEPTS`=4 depts AND whose body-Jaccard clears `HEADER_BODY_JACCARD_FLOOR`=0.12 is flagged boilerplate (`boilerplate_match_source="header_anchored"`); `BP_THRESHOLD` + dedup thresholds untouched. TWO over-hide guards added after verification caught regressions: (1) floor raised 0.05→0.12 (ECEN_749 Late Work jac 0.055); (2) the header-anchored path now REJECTS any chunk whose body carries a course-specific signal (% or concrete date) via `find_course_specific_signal` — a fixed-wording univ policy never holds an instructor's own "20% penalty". (silver_tag 17 tests + boilerplate_quality tests, 0 regr.)
  status: 🟢 CONFIRMED end-to-end (rebuilt 2026-06-10)  ·  STAT_651_600 — Makeup/Academic-Integrity/ADA now tagged boilerplate (jac 0.69/0.41/0.31). KNOWN residual: Mental Health (0.094) + Title IX (0.086) stay UNtagged — too OCR-damaged / differently-titled for the 0.12 floor; left as a BP_MISSED miss (visible) rather than risk over-hiding course content. FULL-CORPUS sweep (100 stems re-tagged 2026-06-10): 0 header-anchored chunks below 0.12, and 0 over-hides (boilerplate chunks carrying a course-specific signal) — down from 4 (all "Late Work Policy") before the signal gate. BP-rate corpus median 59%.

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
