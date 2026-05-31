# v6 Ingestion Pipeline — Additive + Vector-Tagged

Status: planning (2026-05-25). Supersedes v5 stripping-based design once piloted.

## Why v6

The v5 pipeline is **subtractive** — it strips boilerplate by header-name registry, then runs LLM enrich + validate on what's left. Pain points surfaced on the STAT pilot:

- Registry needs manual updates per department (~10 entries added during STAT pilot).
- Wrapper-cascade bug: `## College and Department Policies` stripped `### Homework Policy` (course content) — patched today but the *class* of bug recurs.
- Docling hierarchy errors push course content under boilerplate headers, then it gets swept (4 of 5 real losses in the v3 pilot were this).
- Validator-vs-registry disagreement: judge flags 8+ TAMU-policy strips per pilot as "missing content"; philosophical, unresolvable via tweaks.
- TAMU gateway returns degenerate JSON ~20% on combined enrich call (mitigated via 3 retries + max_tokens=8192, not fixed at root).
- Image-recovery is brittle: Gemini path costs money, hallucinates; Claude-via-skill is manual and not Dagster-tracked.

Root cause: **decisions are made too early and destructively**. We delete content before we know whether it matters, then can't recover.

## v6 design — additive

Nothing is destructively removed. Every section is preserved and chunked; each chunk is tagged with `is_boilerplate` via corpus-wide vector similarity. Retrieval filters at query time.

### Stage flow

```
raw_pdf
   │  (file copy, unchanged from v5)
   ▼
bronze              ONE multimodal call per file:
                    PDF → clean markdown + hierarchy JSON sidecar
                    (replaces Docling + silver/01_image_recovery + 02 + 03 + 03b)
   │
   ▼
chunk               Semantic chunker (existing v5 chunker), ~600 tok
                    Every section preserved — no stripping.
                    Output: chunks[] with stable IDs + parent headers.
   │
   ▼
embed               Voyage AI embeddings per chunk
                    (same model as retrieval — chunk-tagging and retrieval
                    share the same embedding space)
   │
   ▼
tag                 For each chunk, cosine sim against the corpus
                    "boilerplate reference set":
                      • is_boilerplate = max_sim ≥ 0.92
                      • boilerplate_cluster = nearest cluster label
                      • confidence = max_sim
                    Borderline chunks (0.85–0.92) → cheap LLM tiebreak (TAMU).
   │
   ▼
enrich              ONE LLM call per file: extract course_metadata +
                    course_summary as JSON (existing v5 combined call).
                    Runs only on NON-boilerplate chunks (cheaper input,
                    cleaner signal).
   │
   ▼
validate            Optional. ONE LLM call per file comparing bronze
                    markdown vs tagged chunks: catches VLM extraction
                    errors AND bad cluster assignments.
   │
   ▼
ingest              Chunks → Atlas with tags. Retrieval filters
                    `is_boilerplate=false` by default; opt-in for
                    TAMU-policy queries.
```

## Model assignments

**Hard constraint**: TAMU API does NOT work with multimodal inputs (confirmed from prior incidents). Use TAMU only for text-only stages.

| Stage | Client | Model | Why | Cost |
|---|---|---|---|---|
| **bronze** (PDF→markdown, multimodal) | `get_genai_client()` (direct Google) | `gemini-3.1-flash-lite` | Native PDF input, 1M context, GA May 2026. TAMU can't do this. | ~$0.40/dept |
| **chunk** | n/a | n/a (rule-based) | Existing v5 chunker | $0 |
| **embed** | Voyage AI (existing) | `voyage-3` (current) | Same model as retrieval | ~$0.025/dept |
| **tag tiebreak** (~10% of chunks) | `get_tamu_client()` | TAMU Gemini 2.5 Flash | Free, text-only classifier prompt | $0 |
| **enrich** | `get_tamu_client()` | TAMU Gemini 2.5 Flash | Free, combined metadata+summary JSON call (already retry-hardened) | $0 |
| **validate** | `get_tamu_client()` | TAMU Gemini 2.5 Flash | Free, long-context judge (bronze vs chunks vs enrich JSON) | $0 |
| **cluster labeling** (one-time) | `get_tamu_client()` | TAMU Gemini 2.5 Flash | Free, one prompt per cluster (~20 calls total across all depts) | $0 |

**Total per dept**: ~$0.43 (down from v5 ~$0.30 + manual recovery labor). Almost all cost is in the one paid stage (bronze).

## Reference-set construction (one-time)

Run once across all existing chunks (ISEN + CSCE + STAT ≈ 240 files × ~30 chunks each):

1. Embed every chunk via Voyage.
2. Cluster chunks by cosine similarity (threshold 0.92, e.g. via HDBSCAN or simple agglomerative).
3. A cluster is **boilerplate** if it contains chunks from ≥30% of distinct files (TAMU policies recur across most syllabi; course content does not).
4. Label each boilerplate cluster via one TAMU LLM call: "What policy/section does this represent?" → human-readable label (e.g. `ada_policy`, `ai_statement`, `dept_banner_stat`).
5. Persist as `data/syllabi/_meta/boilerplate_reference.parquet`:
   ```
   {chunk_id, cluster_id, centroid_vec, label, n_files_seen, confidence}
   ```

After this, the `tag` stage just computes cosine sim against the ~20-30 cluster centroids — fast, no LLM in the hot path.

Re-run reference-set construction quarterly or when a new department is added.

## What gets deprecated from v5

| v5 artifact | v6 replacement | Why removed |
|---|---|---|
| `silver/01_image_recovery` | Built into bronze | VLM sees images directly |
| `silver/02_false_positive` | Not needed | VLM produces clean headers |
| `silver/03_boilerplate` | `tag` stage (similarity) | No registry, no cascade |
| `silver/03b_relocate_textbook` | Not needed | VLM places blocks correctly |
| `boilerplate_stripper.py:BOILERPLATE_REGISTRY` | Reference-set clustering | Self-adapting |
| `boilerplate_stripper.py:BODY_BOILERPLATE_HEADERS` | Same | Same |
| `boilerplate_stripper.py:WRAPPER` category (added today) | Same | Same |
| `recover-images` skill | Not needed | Bronze handles it |
| `scripts/render_marker_pages.py` | Not needed | No more `<!-- image -->` markers |
| Existing `silver_false_positive`, `silver_boilerplate`, `silver_relocate_textbook` Dagster assets | New `bronze_vlm`, `tag` assets | Simpler graph |

Existing v5 outputs stay on disk for comparison until v6 is validated.

## Implementation order

Each step is independently rollbackable. Bronze can land first as a one-stage swap before touching the tag idea.

1. **Wire `gemini-3.1-flash-lite` for bronze** via the existing `get_genai_client()` (~10 min — model string + PDF input).
2. **10-file STAT bronze bake-off** vs current Docling bronze: compare extracted markdown for fidelity, hierarchy, table reconstruction. Cost ~$0.05.
3. **Build the boilerplate reference set** from existing ISEN + CSCE + STAT chunks (~7200 chunks). Free except for cluster labeling (~20 TAMU calls).
4. **Implement `tag` stage** as a Dagster asset reading the reference set; tag the same 10 STAT files.
5. **Run v6 end-to-end on the 10 STAT files**: bronze → chunk → embed → tag → enrich → validate.
6. **Compare v6 vs v5 on the 10 files**:
   - chunk count and content fidelity (v6 should have more chunks since nothing stripped)
   - boilerplate tag accuracy (manual spot-check 20 random chunks)
   - enrich JSON quality
   - validate findings count
   - retrieval quality via probe-rag (apples-to-apples on a fixed query set)
7. **If v6 wins**: scale to remaining 71 STAT files, then deprecate v5 stages 01/02/03/03b.
8. **Migrate ISEN + CSCE** to v6 (one dept at a time, keeping v5 outputs as fallback).

## Open questions to answer during implementation

- Does the existing Voyage embedding model handle TAMU policy text well enough that boilerplate chunks reliably cluster ≥0.92? (Should — they're near-identical strings.)
- Cold-start: how confident can we be in clusters from a single-dept run? (Mitigated: we have 3 depts already; cold-start risk only re-emerges for the next net-new dept.)
- Borderline tagging: what's the right LLM tiebreak prompt? Probably "Is this paragraph university-wide institutional policy, or course-specific content?" with the chunk as context.
- Chunk boundary stability: if VLM bronze produces slightly different markdown each call, do chunks still align across re-runs of the same file? (Temperature=0 should help; need to verify with two runs of one file.)
- Does `gemini-3.1-flash-lite` handle full PDFs in one call, or do we need to render per-page? (Spec says PDF input native; verify on a 20-page file.)

## Fallback paths

- If `gemini-3.1-flash-lite` underperforms Docling on hierarchy: keep Docling as a sidecar source of truth, use VLM only for image regions.
- If clustering doesn't separate borderline boilerplate cleanly: revert tag stage to registry-based classification on the chunks (still better than v5 because no cascade).
- If TAMU enrich keeps degenerating on combined call: split back to 2 calls (cost still $0 since TAMU is free).

## Out of scope for v6

- Replacing Voyage embeddings.
- Multimodal validate (sending PDF to judge). Stays text-only via TAMU.
- Local Ollama path. Worth exploring after v6 is stable.
- Manual hierarchy subagent refinement (existing CSCE workflow). v6 reduces need but doesn't eliminate.
