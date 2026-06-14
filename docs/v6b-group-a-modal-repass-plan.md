# Group A — selective GPU modal/VLM re-pass plan (ISEN_665, CSCE_704)

Status: **design only — no GPU run yet.** Closes the two iter_11 fidelity residuals that
no host-side text fix can reach because the content lives in *pixels*, not the PDF text
layer. Both need the vision model; this plan scopes a *targeted* pass so we don't pay the
NuExtract GPU ceiling (~11–14 tok/s, see [[project-nuextract-gpu-ceiling]]) for the whole
corpus.

## The two residuals (one root cause, two shapes)

| stem | judged class | shape | deterministic detector (now shipping) |
|---|---|---|---|
| ISEN_665 | FID_TABLE_LOST | multi-page schedule **table rendered as an image** on pp.6–8; modal disabled → bare `<!-- image -->` | `v6b_bronze_blocks_no_content_image_lost` → flags pp.6,7,8 |
| CSCE_704 | FID_CONTENT_DROPPED | a ~25-line grading region trapped on a **textless, un-OCR'd page** (page 3, bbox-less image) | `v6b_bronze_blocks_no_ocr_failure_page` → flags page 3 |

Both are now **algorithmically targetable** — the two new bronze checks emit the exact
stems + page indices that need vision, so the re-pass is a worklist, not a full sweep.

## Why host-side recovery is impossible here

- ISEN_665: the schedule table has **no text layer at all** on those pages — PyMuPDF
  `get_text()` and `find_tables()` both return empty; there is nothing to thread.
- CSCE_704: page 3 is a scanned/scrambled image placeholder; `pdf_integrity` already shows
  ~0 chars on that page. The grading region's anchors survive *elsewhere* as table cells,
  which is why the contiguous-missing-run heuristic fragmented (see `pipeline-failures.md`
  2026-06-09 CSCE_704 entry). Recovery requires reading the pixels.

## Proposed approach

### 1. Targeting (free, already built)
Materialize the two new checks across the corpus and read their metadata to build the
re-pass worklist:
```
dagster asset materialize --select v6b_bronze_blocks -f .../pipeline_v6b/definitions.py
# then collect stems where no_content_image_lost / no_ocr_failure_page failed,
# with their offending_pages from check metadata.
```
Expected worklist ≈ the FID_IMAGE_LOST ×23 cohort + these 2, but **only the pages that
flag** — not every image (logos are filtered by the area/caption gates).

### 2. Scoped modal pass (GPU, `gpu-ops`)
For each `(stem, page)` on the worklist, run the existing modal/VLM path **on that page's
image block(s) only**:
- ISEN_665 → the image is a table → route to the table-transcription prompt; the existing
  `render_table_markdown` row-conservation guard (`validation/table_quality.py`) already
  protects against VLM truncation, so a recovered grid is gated on rows ≥ 0.8× any
  fallback grid.
- CSCE_704 → the image is a full page → render the page to PNG and run a **page-level VLM
  transcription** (markdown), then splice the recovered text back as a synthetic
  text/table block at the page's position. This is the one genuinely new code path: today
  `silver_modal` operates on Docling image blocks with a bbox; a bbox-less full-page scan
  needs a page-render fallback. Gate the recovered text on the same `is_failure_marker`
  guard that already prevents errno-string leaks (shipped in this branch).

### 3. Pre-flight: bust the content-keyed modal cache
The VLM cache is keyed on image+grid hash, not decode params (see
[[project-v6b-modal-cache-content-keyed]]). For a clean re-test, bust the cache entries
for the worklist stems before the pass; otherwise a prior disabled-modal result can shadow
the new transcription.

### 4. Verify
Re-judge ISEN_665 + CSCE_704 (the cheap 2-stem `judge-preprocessing` path), confirm
`no_content_image_lost` / `no_ocr_failure_page` flip to pass on those pages, and that the
grid checks (`v6b_silver_modal_no_table_lost`) stay green. Log both in
`pipeline-failures.md`.

## Effort + sequencing

- New code: a **page-render VLM fallback** in `assets/silver_modal.py` for bbox-less
  full-page image blocks (CSCE_704). ISEN_665 needs no new code — just enabling the table
  route on its flagged pages.
- GPU cost: ~25 pages across the worklist at the NuExtract ceiling ≈ minutes, not hours —
  because targeting keeps it to flagged pages only.
- Risk: low. Both new bronze checks are add-only WARN gates; the modal path already has
  truncation + failure-marker guards. The page-render fallback is the only net-new surface
  and is itself gated.

## Decision needed

This is the only remaining work that requires a GPU session. Until then, ISEN_665 and
CSCE_704 stay as **known, now-detected** residuals — the two new checks mean they will
never again be silent. Recommend running it as one focused `gpu-ops` session after the
full 100-stem re-judge confirms no host-side regressions.
