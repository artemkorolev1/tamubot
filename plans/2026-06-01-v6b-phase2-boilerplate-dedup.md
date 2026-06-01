# v6b Phase 2 — Boilerplate Detection + Deduplication

**Date:** 2026-06-01
**Branch (start):** `feat/v6b-phase1-robustness` → cut `feat/v6b-phase2-bp-dedup` at execution.
**Prior:** Phase 1 robustness (source-integrity check, alert sensor, corpus report, ACCURATE TableFormer) — committed, unpushed.

## 0. One-line goal

Implement the dormant `v6b_silver_tag_semantic` asset as a real tagger that (a) flags boilerplate chunks by matching against a shared reference parquet and (b) flags within-syllabus near-duplicates. Flag-only — never drop.

---

## 1. Constraints (load-bearing — don't relax without a new ADR)

- **No API calls during tagging.** Tagging runs *before* embed in the asset chain (`chunk → tag → embed → atlas`). All matching is text-based: n-gram normalization for boilerplate, MinHash-LSH for dedup. The `# TODO cosine sim` comment in `silver_tag.py:31` is wrong for this stage — cosine would require an API call and break this rule.
- **Never drop chunks at tag stage.** Already enforced by the blocking check `v6b_silver_tag_chunk_count_preserved` (asserts `n_in == n_out`). Phase 2 sets flags, period.
- **Reference parquet shared with v6.** `paths.boilerplate_reference_path()` resolves to `DATA_ROOT/_meta/boilerplate_reference.parquet`. Both pipelines read the same artifact.
- **Pure tagging functions.** Keep `_tag_chunks` IO-free for unit testability. IO lives in the asset wrapper.
- **Dark by default for retrieval.** New `INCLUDE_DUPLICATE` env var (default `false`), mirroring existing `INCLUDE_BOILERPLATE`.

---

## 2. Architecture

### 2a. Boilerplate reference set (new asset)

A corpus-scanning asset `v6b_meta_boilerplate_reference` (group: `v6b_meta`, no partition) builds the parquet by:

1. Globbing every `silver_chunk_semantic_path(stem)` artifact under `DATA_ROOT/*/v6b/silver/02_chunk/`.
2. Normalizing each chunk content: lowercase, collapse whitespace, strip punctuation → `_normalize_text`.
3. Bucketing by exact normalized-text hash for the high-confidence path.
4. Running MinHash-LSH (Jaccard ≥ 0.85) over normalized text to catch near-verbatim variants.
5. Keeping clusters where `doc_frequency ≥ 5` AND `distinct_depts ≥ 3`. Cross-department repetition is the strongest signal: a chunk repeated 20× within one dept is curriculum reuse, not boilerplate.
6. Writing the parquet.

**Parquet schema:**

| column | type | purpose |
|---|---|---|
| `cluster_id` | str | stable SHA-256 of canonical normalized text |
| `representative_text` | str | one example chunk's raw content (for human inspection) |
| `normalized_text` | str | what is actually compared |
| `doc_frequency` | int | distinct syllabi this cluster appears in |
| `distinct_depts` | int | distinct departments this cluster appears in |
| `ngram_signature` | list[str] | top-32 5-grams used for fast Jaccard scoring |

**Not in the per-partition graph.** Materialized manually:
```bash
dagster asset materialize --select v6b_meta_boilerplate_reference -f src/tamubot/ingestion/pipeline_v6b/definitions.py
```

### 2b. Boilerplate tagger (inside `_tag_chunks`)

```python
def _tag_chunks(
    chunks: list[dict],
    reference_index: ReferenceIndex | None,
    bp_jaccard_threshold: float = 0.80,
    dedup_jaccard_threshold: float = 0.92,
) -> tuple[list[dict], dict]:
    # Pass 1: boilerplate via reference set
    for c in chunks:
        if reference_index is None:
            c["is_boilerplate"] = False
            continue
        norm = _normalize_text(c["content"])
        cluster_id, confidence = reference_index.match(norm, threshold=bp_jaccard_threshold)
        if cluster_id is not None:
            c["is_boilerplate"] = True
            c["boilerplate_cluster"] = cluster_id
            c["cluster_confidence"] = confidence
    # Pass 2: within-syllabus dedup over the non-boilerplate set
    _flag_within_syllabus_duplicates(chunks, threshold=dedup_jaccard_threshold)
    return chunks, _stats(chunks)
```

`ReferenceIndex` is a lightweight wrapper around the parquet: at construction it builds (i) an exact normalized-text → cluster_id dict and (ii) a `datasketch.MinHashLSH` for fuzzy matching. **Memoized at module level** so the parquet isn't reread per-partition within a Dagster run.

### 2c. Dedup — both within-syllabus AND cross-syllabus

Phase 2 ships dedup at two scales, each with a different threshold and a different upstream artifact.

**Within-syllabus dedup** (per-partition, no upstream artifact needed):
1. Build a MinHash (128 perms) per non-boilerplate chunk in the syllabus.
2. LSH at Jaccard ≥ 0.92.
3. For each near-duplicate cluster within the syllabus, pick a canonical chunk and tag the others with `is_duplicate=True` + `duplicate_of_chunk_id=<canonical>`.

**Cross-syllabus dedup** (reads from a new corpus-level signature index):
1. A new meta asset `v6b_meta_chunk_signature_index` scans the entire corpus, builds a MinHash per non-boilerplate chunk across all syllabi, and writes a parquet at `DATA_ROOT/_meta/chunk_signature_index.parquet` with columns: `chunk_id`, `stem`, `chunk_index`, `minhash_bytes`, `token_count`.
2. The per-syllabus tagger loads this index once (module-level memo) into a `datasketch.MinHashLSH` keyed by `chunk_id`.
3. For each non-boilerplate chunk in the current syllabus, query LSH at Jaccard ≥ **0.95** (stricter than within-syllabus to avoid false positives on common course-skeleton phrasing).
4. If duplicates exist anywhere in the corpus, deterministically pick canonical: lexically smallest `(stem, chunk_index)` tuple across the cluster. Chunks that aren't canonical get `is_duplicate=True` and `duplicate_of_chunk_id=<canonical chunk_id>`.

**Self-match handling:** the index includes the current syllabus's own chunks. Each chunk's MinHash will match itself with Jaccard 1.0 — the matcher excludes self-matches by `chunk_id`.

**Boilerplate-vs-dup boundary:** boilerplate is "≥5 syllabi AND ≥3 depts" (universally repeated). Cross-syllabus dups that don't clear that bar (e.g., same prof reusing a chunk across 3 sections, all in one dept) still get flagged here. The two flags are independent — a chunk can be either or both but the order of passes ensures boilerplate wins (it's checked first).

**Order in `_tag_chunks`:**
1. Boilerplate pass (sets `is_boilerplate=True` on matches).
2. Within-syllabus dedup over the remainder.
3. Cross-syllabus dedup over the remainder.

A boilerplate chunk is never also tagged as a duplicate (it's already excluded from retrieval).

The chunk id used as `duplicate_of_chunk_id` is the chunk's `_id` from `silver_chunk_semantic` (verify schema during step 5).

### 2d. Schema additions (`rag/models_v4.py`)

```python
# v6b Phase 2 dedup (additive — same back-compat pattern as is_boilerplate)
is_duplicate: bool = False
duplicate_of_chunk_id: Optional[str] = None
```

Atlas docs missing these fields are treated as non-duplicate via `$ne: True`. No index migration needed.

### 2e. Retrieval-side filter (`rag/tools/mongo.py`)

Extend `_atlas_filter` and `_build_text_stage` in lockstep with the existing `is_boilerplate` handling:

```python
if not config.INCLUDE_DUPLICATE:
    f["is_duplicate"] = {"$ne": True}                                 # vector path
    compound["mustNot"].append({"equals": {"path": "is_duplicate", "value": True}})  # bm25 path
```

### 2f. Asset checks

`checks/silver_tag_checks.py` gains two **non-blocking** checks:

| check | severity | range / rule |
|---|---|---|
| `v6b_silver_tag_chunk_count_preserved` (existing) | **ERROR (blocking)** | `n_in == n_out` |
| `v6b_silver_tag_boilerplate_rate_in_band` (new) | **WARN** | `0.05 ≤ rate ≤ 0.45` |
| `v6b_silver_tag_duplicate_rate_in_band` (new) | **WARN** | `rate ≤ 0.20` |

Outlier rates surface in the Dagster UI without halting the run — Phase 2 calibrates thresholds; Phase 3+ may tighten.

---

## 3. File-by-file change list

| Path | Status | Change |
|---|---|---|
| `src/tamubot/ingestion/pipeline_v6b/util/text_normalize.py` | **new** | `_normalize_text`, `_ngrams`, `_minhash_of`, `ReferenceIndex`. Pure. |
| `src/tamubot/ingestion/pipeline_v6b/assets/meta_boilerplate_reference.py` | **new** | Corpus-scanning asset that builds the boilerplate parquet. |
| `src/tamubot/ingestion/pipeline_v6b/assets/meta_chunk_signature_index.py` | **new** | Corpus-scanning asset that builds the cross-syllabus dedup signature index parquet. Depends on `silver_chunk_semantic` artifacts. |
| `src/tamubot/ingestion/pipeline_v6b/paths.py` | modify | Add `chunk_signature_index_path()` → `DATA_ROOT/_meta/chunk_signature_index.parquet`. |
| `src/tamubot/ingestion/pipeline_v6b/assets/silver_tag.py` | modify | Replace no-op `_tag_chunks` with real logic; load reference parquet via memoized helper; add within-syllabus dedup pass. Update docstring. |
| `src/tamubot/ingestion/pipeline_v6b/checks/silver_tag_checks.py` | modify | Add two non-blocking range checks. |
| `src/tamubot/ingestion/pipeline_v6b/definitions.py` | modify | Register new asset + new checks. |
| `src/tamubot/rag/models_v4.py` | modify | Add `is_duplicate`, `duplicate_of_chunk_id`. |
| `src/tamubot/rag/tools/mongo.py` | modify | Extend filter for `is_duplicate` (both vector + BM25). |
| `src/tamubot/core/config.py` | modify | Add `INCLUDE_DUPLICATE` env var (default `false`). |
| `src/tamubot/ingestion/pipeline_v6b/assets/silver_chunk_semantic.py:83-85` | modify | Pre-seed `is_duplicate: False`, `duplicate_of_chunk_id: None` alongside existing boilerplate defaults — keeps the per-chunk dict shape consistent before tag stage. |
| `requirements.txt` | modify | Add `datasketch` (MinHash-LSH). Verify `pyarrow` is already present for parquet. |
| `tests/test_v6b_text_normalize.py` | **new** | Unit tests: normalization idempotence, n-gram set determinism, MinHash Jaccard accuracy. |
| `tests/test_v6b_silver_tag.py` | **new** | Unit tests for `_tag_chunks`: no-reference path, exact-match path, fuzzy-match path, within-syllabus dedup, never-drop invariant. |
| `tests/test_v6b_meta_boilerplate_reference.py` | **new** | End-to-end builder test on a synthetic 10-stem fixture corpus. |
| `tests/test_v6b_meta_chunk_signature_index.py` | **new** | End-to-end builder test on a synthetic 10-stem fixture corpus; assert round-trip Jaccard accuracy. |

---

## 4. Validation gates (must pass before commit)

1. `pytest tests/test_v6b_*.py -v` — green.
2. `make test lint typecheck` — green.
3. `dagster asset list -f src/tamubot/ingestion/pipeline_v6b/definitions.py` — expect 10 assets (was 8 — added boilerplate-ref + signature-index), 24 checks (was 22).
4. Build the reference parquet against the current corpus and verify it has ≥ 1 cluster:
   ```bash
   dagster asset materialize --select v6b_meta_boilerplate_reference -f src/tamubot/ingestion/pipeline_v6b/definitions.py
   python -c "import pandas as pd; print(pd.read_parquet('tamu_data/_meta/boilerplate_reference.parquet').head())"
   ```
5. Smoke re-materialize `v6b_silver_tag_semantic` for one stem; inspect output JSON for non-zero `is_boilerplate=True` count and confidence values.
6. `make probe` (or eval golden recall) with `INCLUDE_DUPLICATE=false` — recall must not regress vs. Phase 1.

---

## 5. Rollout / shipping-dark plan

- `INCLUDE_BOILERPLATE=false` (existing default) → boilerplate hidden from retrieval. **Unchanged.**
- `INCLUDE_DUPLICATE=false` (new default) → duplicates hidden from retrieval. **Tightening from Phase 1**, but back-compat-safe via `$ne: True`.
- Atlas index requires no migration — sparse-tolerant filter pattern handles pre-Phase-2 docs.
- New non-blocking checks surface in the Dagster UI but don't gate runs.

---

## 6. Out of scope (explicit)

- Semantic clustering of boilerplate (LLM-named clusters, etc.) → separate effort; Phase 2 ships content-hash cluster IDs only.
- Backfilling pre-v6b chunks in Atlas → not needed; filters are back-compat-safe.
- v6 pipeline parallel work → Phase 2 is v6b-only; the shared parquet is the only cross-pipeline artifact.

---

## 7. Risks & mitigations

| Risk | Mitigation in Phase 2 |
|---|---|
| Reference parquet drifts stale → mis-tagging | Log `parquet_mtime` + `parquet_sha256` in `MaterializeResult.metadata`. Freshness alert is a Phase 3 candidate. |
| Jaccard thresholds (0.80 / 0.92) miscalibrated | Non-blocking rate-in-band checks will surface outliers. Thresholds are tunable constants at the top of `silver_tag.py`. |
| Module-level `ReferenceIndex` memo goes stale mid-run | Acceptable: parquet builds are manual/offline; per-run staleness is the trade for not paying parquet-read cost per partition. |
| `datasketch` adds a runtime dep | Pin in `requirements.txt`; verify it's pure-Python (it is — no native deps). |

---

## 8. Execution sequence (subagent-driven)

Steps are mostly sequential because each builds on the prior; subagents are appropriate for **steps 2 and 6** (independent helper modules + retrieval-side wiring) and for the test files which can be drafted in parallel with implementation once interfaces are fixed.

1. Cut `feat/v6b-phase2-bp-dedup` from current branch.
2. **[subagent-able]** `text_normalize.py` + `test_v6b_text_normalize.py` (TDD). Pin `datasketch` in `requirements.txt`.
3. `meta_boilerplate_reference.py` asset + `test_v6b_meta_boilerplate_reference.py`. Register in `definitions.py`.
3b. `meta_chunk_signature_index.py` asset + `test_v6b_meta_chunk_signature_index.py`. Register in `definitions.py`.
4. Materialize **both** meta parquets against the current corpus (manual command); verify each is sensible.
5. Rewrite `_tag_chunks` in `silver_tag.py` (boilerplate + within-syllabus dedup + cross-syllabus dedup, in that order) + `test_v6b_silver_tag.py`. Update `silver_chunk_semantic.py` per-chunk defaults.
6. **[subagent-able]** Schema additions to `models_v4.py` + filter extension in `rag/tools/mongo.py` + `INCLUDE_DUPLICATE` in `config.py`.
7. Add two non-blocking checks; register in `definitions.py`.
8. Smoke re-materialize `silver_tag` for one stem and inspect.
9. `make test lint typecheck` clean.
10. Commit in 4–6 logical chunks; do not push (user pushes).

---

## 9. Definition of done

- All validation gates in §4 pass.
- `feat/v6b-phase2-bp-dedup` branch has 4–6 clean commits, unpushed.
- Reference parquet exists at `DATA_ROOT/_meta/boilerplate_reference.parquet` and contains ≥ 1 cluster.
- A spot-checked stem shows non-zero `is_boilerplate` chunks and (where present) `is_duplicate` chunks, all with `chunks_in == chunks_out`.
- The two new non-blocking checks materialize and report metadata in the Dagster UI.
- No regression in `make probe` golden recall at default flag settings.
