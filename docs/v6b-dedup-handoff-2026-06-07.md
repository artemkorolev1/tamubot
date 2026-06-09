# v6b Boilerplate / Dedup — Handoff (2026-06-07)

## Context

Investigating whether the v6b boilerplate reference + cross-syllabus dedup
indexes work correctly across runs. Branch: `feat/v6b-phase2-bp-dedup`.

**Verdict:** the persistence *mechanism* is sound and the old silent-zero-dedup
cache bug (H2) is fixed. But the **current on-disk indexes are stale** (runtime
problem, not in the review doc), and two **open logic bugs** (M1, M2) affect
dedup/boilerplate correctness regardless of staleness.

### How the indexes work (one-paragraph recap)

`data/syllabi/_meta/boilerplate_reference.parquet` and
`chunk_signature_index.parquet` are plain files (not Dagster-I/O-managed). They
**persist** across runs. Each `v6b_meta_*` materialization is a **full overwrite**
that rescans the whole corpus — not an accumulating dictionary. Boilerplate is a
cross-document frequency signal, so the indexes must be rebuilt whenever the
**chunk corpus changes**. `silver_tag` reads both parquets; if either is missing,
that pass is a no-op.

---

## BUG 1 — Stale meta indexes (NEW finding; runtime state, not in review doc)

**Severity: high (data currently wrong corpus-wide) — but it's a re-run, not a code fix.**

- Both parquets built **Jun 2 ~22:29**.
- **16 chunk files were produced *after* that** (ECEN, up to 22:45) → indexes
  don't know about them.
- Corpus state: **85 chunked / 69 tagged** — the 16-file gap is those newer
  files; they have **never been tagged** (0 boilerplate/dedup applied).
- Knock-on: the 69 tagged files were tagged against an index missing the 16, so
  **cross-syllabus dedup is incomplete corpus-wide**, and boilerplate frequency
  thresholds were computed over an incomplete corpus.

**Fix (order matters — `code_version` hashes only the asset's own fn, so Dagster
will NOT auto-flag `silver_tag` stale):**

1. Re-materialize `v6b_meta_boilerplate_reference` + `v6b_meta_chunk_signature_index`
   (full rescan of all 85 chunks).
2. **Force** re-materialize `v6b_silver_tag_semantic` across **all** stems
   (`--select v6b_silver_tag_semantic …`), not just the 16.

This is the recurring two-phase-ordering issue (review-doc **L9**): not encoded
in the DAG, must be done manually whenever the corpus grows.

---

## BUG 2 — M1: cross-syllabus dedup can flag *both* copies / non-deterministic canonical

**Severity: medium · Status: open · `DUP_WRONG_CANON` / `DUP_FALSE`**

- **Where:** `util/tagging.py:49-124`; index from `util/signature_index.py:38`.
- Within-pass canonical = longest content (`:72-77`); cross-pass canonical =
  lex-min `(stem,chunk_index)` (`:111-121`) → the two passes can pick different
  survivors for one intra-doc group, so **both copies can end up flagged**.
- Same-stem twins aren't excluded from the cross pass (only exact `my_id` is,
  `:102`); `visited` keys on a per-query neighbor `frozenset` (`:68-71`) →
  order-dependent canonical on non-transitive chains.
- **Fix:** skip same-stem candidates in the cross pass; make within-cluster
  assignment deterministic (union-find, single canonical).
- **Note:** util-only fix → forced re-materialize of `silver_tag` required after.

---

## BUG 3 — M2: empty-MinHash collision on the boilerplate-reference path

**Severity: medium · Status: open · `BP_OVERREACH`**

- **Where:** `util/text_normalize.py:45-53` (`minhash_of`), `:62-113`
  (`ReferenceIndex`).
- Text with <5 word-tokens → empty n-grams → all-default `MinHash`; two such have
  `jaccard==1.0` and LSH returns empty-inserted keys for an empty query. The
  **dedup** path is shielded by `min_chunk_tokens=50`; the **reference** path is
  **not** — a short boilerplate rep ("Office Hours") becomes an empty signature
  that collides with any short query at confidence 1.0 → over-tagging.
- **Fix:** treat "no n-grams" as "no signature" — skip from sigs, return no-match
  in `match()`. Updates `tests/test_v6b_text_normalize.py:64-68` (currently
  asserts the buggy behavior).

---

## BUG 4 — L5: corpus-level BP/dup rate band applied per-partition (cosmetic)

**Severity: low · Status: open**

- **Where:** `checks/silver_tag_checks.py:23-25`.
- A corpus-level boilerplate/duplicate rate band is checked per-partition → noisy
  WARNs. This is the single ⚠ in the current ledger (`with_failures: 0`, so
  nothing hard-broken). Cosmetic gate-tuning, not a data bug.

---

## Already fixed (do NOT re-investigate)

- **H2** — stale path-keyed parquet cache served 0-dedup silently. Fixed: cache
  key now includes `hash_file(parquet)` (`silver_tag.py:36-71`).

---

## Verification commands

```bash
# staleness re-check (want 0 newer than the parquet):
find data/syllabi -path "*/02_chunk/*.json" -newer data/syllabi/_meta/boilerplate_reference.parquet | wc -l
find data/syllabi -path "*/02_chunk/*.json" | wc -l   # chunked
find data/syllabi -path "*/03_tag/*.json"   | wc -l   # tagged — should match chunked after retag
```

- Full prior review + phasing: `docs/v6b-preprocessing-review.md`.
- Host Python (3.14) has **no pandas/pyarrow** — parquet internals must be
  inspected inside the `tamubot-dev-1` container.
- `DATA_ROOT = data/syllabi` (`pipeline_v5/util.py:11`).
