# v6b Preprocessing — Code Review (2026-06-02)

Tracking doc for the v6b preprocessing review. Findings are verified against
source and red-teamed (zero false positives). Each links to its
[error taxonomy](preprocessing_error_taxonomy.md) type. Severity = impact on
**retrieved-content correctness**. Status checkboxes track the fix.

> **Sequencing rule.** `code_version_of` (`pipeline_v5/util.py:69`) hashes only an
> asset's *own* compute-function source — **not** its imported util modules. So a
> fix inside `util/tagging.py` or `util/text_normalize.py` does **not** mark
> `v6b_silver_tag_semantic` stale in Dagster; the asset looks current while its
> behavior changed. Any util-only fix must be followed by a **forced**
> re-materialize of the affected stage + downstream.

---

## HIGH

### H1 — Tables guaranteed-dropped in the default pipeline · `FID_TABLE_LOST`
- [ ] **Status: open** (Phase C — content-invalidating)
- **Where:** `assets/silver_modal.py:106-116` (`_merge_to_markdown`); data in
  `converters/docling_block_adapter.py:418-433` (`table_body`).
- **Problem:** The adapter extracts every table into `block["table_body"]` (2-D
  cell list) **and deliberately drops the table's cell `TextItem`s**
  (`docling_block_adapter.py:438-441`). `_merge_to_markdown` only renders
  `modal_result.table_markdown`/captions — it never reads `table_body`. With
  `V6B_MODAL_ENABLED` defaulting to **false**, `modal_result` is absent, so the
  chunker markdown (`silver_chunk_semantic.py:50`) has *no table content and no
  fallback*. Total silent table loss for any syllabus whose grading/schedule/
  weights live in a table. Existing `silver_modal_checks` don't catch it (they
  only validate blocks that *have* a `modal_result`).
- **Fix:** render `table_body` → GFM in `_merge_to_markdown` for the
  no-`modal_result` case (already persisted in bronze blocks). Fixing it here,
  not in the adapter, means re-materialize from `silver_modal` forward only — no
  re-Docling.

### H2 — Stale path-keyed parquet cache serves 0-dedup silently
- [x] **Status: fixed** (Phase 0 — inert)
- **Where:** `assets/silver_tag.py:36-71`.
- **Problem:** `_REFERENCE_INDEX_CACHE` / `_CROSS_SYL_CACHE` were keyed only by
  file *path*; `hash_file(parquet)` was computed but never used for invalidation.
  In a long-lived process (Dagster daemon, or `scripts/v6b_staged_run.py` which
  builds the meta indexes then re-runs tag in the *same* process), a same-path
  index rebuild was not picked up → tag kept the old/empty index → 0 boilerplate,
  0 dedup, silently. Defeated the two-phase dedup workflow.
- **Fix:** cache key now includes `hash_file(parquet)` → content change
  invalidates the cache. Pure in-process; changes nothing on disk.

### H3 — Atlas upsert key collides across same-CRN sources; gate keys differ
- [ ] **Status: open** (Phase D — land before `V6B_INGEST_ENABLED=true`)
- **Where:** `assets/silver_atlas_upsert.py:33` +
  `checks/silver_atlas_upsert_checks.py` (`vector_count_matches_chunks`).
- **Problem:** upsert filter `{crn, chunk_index, chunk_tag}`; `chunk_tag` is the
  constant `"v6b_semantic"` and `crn = parts[4]` is unchanged by the `_HP` suffix
  (`parts[5+]`). A Howdy (`_HP`) and a Simple-Syllabus copy of the same CRN share
  a key → `$set` overwrite, last-writer-wins (silent cross-source loss). The
  **blocking** count check counts by `source_file`, not `crn`, so a collision
  makes upsert and gate disagree → the gate can flap. Low live impact today only
  because ingest defaults to dry-run.
- **Fix:** add `source_file` (stem) to the upsert filter **and** align the count
  check to the same identity, one PR.

---

## MEDIUM

### M1 — Cross-syllabus dedup: same-stem twins + divergent/non-deterministic canonical · `DUP_WRONG_CANON` / `DUP_FALSE`
- [ ] **Status: open** (Phase C — content-invalidating, util-only → forced re-materialize)
- **Where:** `util/tagging.py:49-124`; index from `util/signature_index.py:38`.
- **Problem:** (a) the signature index includes the current stem; the cross pass
  excludes only exact `my_id` (`:102`), not same-stem twins. (b) within-canonical
  = longest content (`:72-77`) vs cross-canonical = lex-min `(stem,chunk_index)`
  (`:111-121`) → the two passes can pick different survivors for one intra-doc
  group, so **both copies can end flagged**. (c) `flag_within_syllabus_dups`
  `visited` keys on one query's exact neighbor `frozenset` (`:68-71`) →
  order-dependent canonical on non-transitive chains.
- **Fix:** skip same-stem candidates in the cross pass; make within-cluster
  assignment deterministic (union-find, single canonical).

### M2 — Empty-MinHash collision (live on the boilerplate-reference path) · `BP_OVERREACH`
- [ ] **Status: open** (Phase C; updates `tests/test_v6b_text_normalize.py:64-68`,
  which currently asserts the buggy behavior)
- **Where:** `util/text_normalize.py:45-53` (`minhash_of`), `:62-113` (`ReferenceIndex`).
- **Problem:** text with <5 word-tokens → `ngrams==[]` → all-default `MinHash`;
  two such have `jaccard==1.0` and LSH returns empty-inserted keys for an empty
  query. The dedup path is shielded by `min_chunk_tokens=50`, but the reference
  path is not: a short boilerplate rep ("Office Hours") becomes an empty signature
  that LSH-collides with any short query at confidence 1.0.
- **Fix:** treat "no n-grams" as "no signature" — skip from sigs; return no-match
  in `match()`.

### M3 — Vision-fallback merge drops multi-page list content · `FID_CONTENT_DROPPED`
- [x] **Status: fixed** (Phase B — vision side-branch only)
- **Where:** `assets/silver_structured.py` (`merge_extracts`).
- **Problem:** "first populated value per field" applied to list fields too, so
  only page 1's `learning_outcomes`/`assessment_weights`/`meeting_schedule`
  survived on multi-page scans.
- **Fix:** list fields now union/extend across pages (order-preserving dedupe);
  scalars keep first-populated.

### M4 — Degenerate (all-null) structured extracts accepted silently
- [x] **Status: fixed** (Phase B — additive check)
- **Where:** `assets/silver_structured.py` (returns degenerate after 3 tries); no
  check existed.
- **Problem:** an all-null extract was written and the partition went green; no
  `silver_structured_checks.py` existed (every other silver asset has one).
- **Fix:** added non-blocking `v6b_silver_structured_not_degenerate` (requires
  `course_code` or `instructor_name`), registered in `definitions.py`.

### M5 — Root-body chunk omits `NO_HEADER`, biasing a BLOCKING check · `CHUNK_*`
- [ ] **Status: open** (Phase C — fold into the H1 re-materialize)
- **Where:** `chunker_v4.py:506-520` vs `_emit:432-433`; gate at
  `checks/silver_chunk_checks.py:56-66` (ERROR) via `token_distribution.py:29`.
- **Problem:** the pre-first-header chunk has `header_path=""` but never appends
  the `NO_HEADER` flag, so the blocking `low_no_header_rate` gate undercounts.
- **Fix:** emit `NO_HEADER` consistently for headerless chunks.

---

## LOW / robustness / cleanup

| ID | Where | Issue | Status |
|---|---|---|---|
| L1 | `silver_embed.py:24-41` | returns 0/1 flag, not real voyage call count (gate meaningless); embeds boilerplate+dup chunks | [x] fixed (real count) |
| L2 | `silver_atlas_upsert.py:28` | `MongoClient` per partition, never closed | [x] fixed (context-managed) |
| L3 | `bronze_blocks.py:40-67` | `<stem>.error.json` dead-letter not removed on later success | [x] fixed |
| L4 | `chunker_v4.py:419,464,480` | tiny-merge ignores `header_path` (legacy `_merge_small:270` guarded it); stale `OVERSIZED` after growth | [ ] open (Phase C) |
| L5 | `checks/silver_tag_checks.py:23-25` | corpus-level BP/dup rate band applied per-partition → noisy WARNs | [ ] open (Phase E) |
| L6 | `baseline_diff.py:83`, `_runner_common.py:79`, `diff_runs.py:56`, `pipeline_v5/util.py:70` | Py2 `except A, B:` — valid on 3.14, breaks on 3.9–3.13 | [ ] open (Phase E) |
| L7 | `signature_index.py:64-68` | `minhash_from_row` doesn't validate `num_perm`/length | [x] fixed (assert) |
| L8 | `chunker_v4` `chunk_markdown`/`chunk_with_log`/`_merge_small` | test-only legacy (used by `tests/test_chunker_v4.py`) — not dead, don't delete | [ ] doc-only |
| L9 | DAG | two-phase ordering not encoded: `silver_tag` deps only on `02_chunk`; meta indexes manual → first run tags 0 | [ ] documented (CLAUDE.md) |

---

## Non-Issues (verified — do not re-flag)

- **`except TypeError, ValueError:` is not a SyntaxError on this stack.** Verified
  on Python **3.14.0**: it compiles, catches *both* types, does **not** rebind
  `ValueError`, lets `KeyError` propagate. Style/portability only → tracked as L6.

---

## NuExtract throughput / GPU ceiling (not a bug — hardware-bound)

Observed ~11–14 tok/s from the NuExtract vLLM sidecar is the **expected ceiling**
on this box, not a misconfiguration. Confirmed from sidecar startup + runtime logs
and `nvidia-smi`:

- GPU is an **RTX A4000 Laptop, 8 GB**. vLLM pre-reserves `--gpu-memory-utilization
  0.78` (~6.4 GB): NuExtract3 4-bit bitsandbytes weights ~3.45 GB, KV pool only
  **0.9 GiB ≈ 23,713 tokens** (`Maximum concurrency for 8,192 tokens/request:
  2.89x`).
- Runtime shows `Running: 1, Waiting: 0`, KV usage ~10%, GPU util ~21%, 32 W/90 W
  → single-stream, **memory-bandwidth-bound** decode. The card isn't saturated;
  each token is slow because of bitsandbytes 4-bit + `--enforce-eager` (CUDA
  graphs off).
- The conservative tuning is **deliberate** for 8 GB and reinforced at two layers:
  `--max-num-seqs 2` at the serving layer and `in_process_executor`
  (`definitions.py:129-131`) at the orchestration layer ("GPU-bound stages must
  not run as parallel subprocesses on a single 8 GB card — they'd OOM").
- Single-stream levers that would help on a bigger card (re-enable CUDA graphs,
  MTP speculative decoding, AWQ/GPTQ instead of bitsandbytes) are **not feasible
  here**: no spare VRAM for graph capture / draft model, and the AWQ/GPTQ path is
  blocked by the `transformers>=5` requirement of NuExtract3 (`docker/vllm-
  nuextract/docker-compose.yml` header). Fanning requests doesn't help: the real
  load is KV-heavy vision (`image:6`), which hits the 2.89x ceiling and stalls.
- **Conclusion:** ~13 tok/s serial is the floor for a *safe* config on 8 GB; let
  serial passes finish. Revisit the levers only on a ≥16 GB GPU.
