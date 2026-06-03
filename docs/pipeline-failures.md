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
