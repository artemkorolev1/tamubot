# VLM Table QC — Strengthening Quality-Control Gates for `silver_modal`

Research proposal · 2026-06-07 · status: PROPOSAL (no code changed)

Scope: the v6b `silver_modal` stage, which parses syllabus tables/images with the
local NuExtract3 VLM (Qwen3.5-VL base) served by the vLLM sidecar. This document
proposes QC gates + fixes. It does **not** implement them and does **not** run the
pipeline or any GPU/VLM materialization.

Owning files referenced throughout:
- `src/tamubot/vendor/raganything/modalprocessors.py` — `TableModalProcessor`, `_robust_json_parse`
- `src/tamubot/ingestion/pipeline_v6b/assets/silver_modal.py` — `_merge_to_markdown`, `_table_body_to_gfm`
- `src/tamubot/ingestion/clients/nuextract_client.py` / `nuextract_http_client.py` — generation params
- `docker/vllm-nuextract/docker-compose.yml` — sidecar launch config
- `src/tamubot/ingestion/pipeline_v6b/checks/silver_modal_checks.py` — existing checks
- `src/tamubot/ingestion/validation/` — reusable pure validators (`CheckOutcome`)

---

## 1. Problem restatement + root-cause confirmation (from the code)

**Symptom (motivating case).** A Week/Topic schedule table degenerated into a
repetition loop: the VLM emitted `{"table_markdown": "| Week | Topic | | | | | | …`
then an unbounded run of empty `| |` cells, never closed the JSON, was truncated at
the token cap, and the parse raised `ValueError("no JSON object…")`. The same
document's other table parsed at confidence 1.0. Docling's own `table_body` for that
table captured only 2 rows, so the merge fallback was empty too. Net result: the
table became `<!-- table: [table processing failed: …] -->` and **all rows were lost
from what RAG sees** — taxonomy code `FID_TABLE_LOST` (blocker).

**Root cause, confirmed in code — four independent contributors:**

1. **Greedy decoding with NO anti-repetition lever.** Both backends decode greedily
   and pass *no* repetition/presence/frequency penalty and *no* `no_repeat_ngram`:
   - in-process: `nuextract_client.py:170` → `self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)`.
   - http: `nuextract_http_client.py:42-48` (`build_chat_payload`) sends only
     `temperature: 0.0` + `max_tokens` — no penalties, no `stop`.
   Greedy/temperature-0 decoding is the classic trigger for neural text
   *degeneration* (tail-loops). On a sparse/merged-cell table the highest-probability
   continuation after `| |` is another `| |`, so the model locks into the loop and
   only the token cap stops it. This is exactly the "tail-loop / single-token
   domination" degeneration mode in the literature (see §2).

2. **No structured-output enforcement.** The "JSON" is produced by free generation +
   a NuExtract chat template, not by constrained/guided decoding. Nothing forces the
   `}` to ever be emitted. vLLM's `guided_json` / `guided_grammar` is **not** wired in
   (`build_chat_payload` has no `guided_*` / `response_format` / `extra_body`).

3. **`max_new_tokens=1024` caps the runaway but doesn't recover it.** The comment at
   `nuextract_client.py:148-150` is explicit: 1024 "caps a degenerate no-EOS run."
   That bounds *cost*, not *correctness* — a capped run is still unterminated JSON,
   so `_robust_json_parse` fails and the block is lost. The http path uses the same
   1024 default (`_post_raw(..., max_new_tokens=1024)` at `nuextract_http_client.py:80`).

4. **The fallback ladder has a hole, and the failure is silent at the gate.**
   - `TableModalProcessor._parse` (`modalprocessors.py:350-382`): on JSON failure it
     drops to `confidence=0.5` with `table_markdown=""` and stuffs the raw text into
     `description`. But `_merge_to_markdown` (table branch, `silver_modal.py:131-146`)
     **only reads `modal_result.table_markdown`**, never the 0.5-confidence
     `description`, so the degenerate-but-partial text is discarded.
   - The merge then falls back to `_table_body_to_gfm(table_body)` — but for this
     table Docling's `table_body` had only 2 rows (`docling_block_adapter.py:257-283`
     `_extract_table_body`), so the fallback is near-empty too.
   - The two existing checks (`silver_modal_checks.py`) **cannot catch this**:
     `result_schema_valid` only checks that `confidence` *key* exists;
     `budget_not_exceeded` only counts blocks with **no** `modal_result` at all. A
     `confidence=0.0`/`0.5` table with an empty markdown passes both. The
     v6b-review H1 finding already flags the merge hole; this proposal adds the
     **detection** layer the review didn't cover.

**Conclusion.** The loss is the product of (a) a decode config that *invites*
degeneration, (b) no structural guarantee of valid JSON, and (c) a fallback ladder
that silently bottoms out with no gate to flag it. All three are addressable without
new hardware; none of the existing asset checks observe table-content survival.

---

## 2. Findings per research question

### Q1 — Why small VLMs repetition-loop on sparse tables, and how it's prevented

- **Mechanism.** Greedy / temperature-0 sampling "often leads to repetitive
  (degeneration)" output; degeneration shows up as **tail-loops** (near-identical
  tokens repeated at the end) and **single-token domination** — precisely the `| |`
  run we saw. Small VLMs specifically "exhibit decoding instability including
  repetition/looping behaviors and incomplete coverage of the page." Qwen-VL family
  (NuExtract3's base) has a tracked, reproducible "infinite repetition" bug class.
- **The right lever for tables is *presence* penalty, not *repetition* penalty.**
  This is the single most important external finding for us:
  > "Increasing the repetition penalty is not an acceptable solution because it
  > breaks the transcription of naturally repetitive text, for example in tables."
  `repetition_penalty` (and to a lesser degree `frequency_penalty`) scale/subtract
  logits for *any* token seen before — in a table, "0", "A", "MWF", blank cells, and
  `|` recur legitimately, so a penalty corrupts good rows. **Qwen's own recommendation
  to stop endless repetition is `presence_penalty` in `[0, 2]`, typically `1.5`**
  (Qwen3 non-thinking default; Qwen3-VL-4B generation config). Presence penalty
  applies a *flat one-time* nudge the first time a token recurs, which is enough to
  knock the model out of a `| |` rut without compounding per-occurrence the way
  `frequency_penalty` does.
- **Prompt-level mitigation also works** and is cheap: terse, imperative "do not
  explain / do not think / output only the table" instructions drove repetition/empty
  failures "to nearly zero" in one VLM-OCR study. Our `TABLE_PROMPT` already says "No
  prose, no truncation, no '...'" — but NuExtract is template-driven, so the *prompt*
  text is largely unused (`silver_modal.py:11-13` notes the prompt arg is accepted but
  ignored); the lever that actually applies is the sampling config + a `stop` string.
- **`no_repeat_ngram` exists but is risky here** for the same reason as
  `repetition_penalty`: a 2-cell `| |` is a legitimate 2-gram in a sparse table;
  hard-banning it can corrupt valid empty-cell rows. Prefer `presence_penalty` +
  `min_tokens`/`stop`.

Recommended decode config for the **table** path (Qwen-aligned, table-safe):
`temperature=0.0` (keep determinism), `presence_penalty≈1.3-1.5`,
`repetition_penalty=1.0` (neutral — do NOT raise), optional `stop=["\n\n\n"]` /
EOS, `max_tokens` bounded. Keep `frequency_penalty=0`.

### Q2 — Structured-output enforcement (guided/constrained decoding on vLLM)

- vLLM supports **Structured Outputs / Guided Decoding** out of the box on the
  OpenAI-compatible server: `guided_json` (JSON-schema-constrained), `guided_regex`,
  `guided_grammar` (CFG), passed via `extra_body` / `response_format`. Backends:
  **xgrammar** (default, recommended), outlines, lm-format-enforcer. xgrammar gives
  "low time per output token… ideal for longer generations" and caches compiled
  grammars across reused schemas — our schema is reused for *every* table, so the
  compile cost amortizes to ~0.
- **What it buys us:** the model is *forced* to emit a JSON object that **closes**.
  The pathological `| |…` run can't continue forever because the grammar's string
  value must terminate and the closing `}` must appear, so we never hit the
  "unterminated JSON → ValueError → table lost" path. It directly removes failure
  contributor (2) from §1.
- **Trade-offs on the 8GB box.** Guided decoding adds CPU↔GPU sync overhead per
  step; vLLM's perf hit is worst at **batch ≥ 8** with sequential mask gen. But our
  sidecar runs `--max-num-seqs 2` single-stream (review §"GPU ceiling"), which is the
  *best* case for guided overhead — xgrammar's per-token mask is cheap at batch 1-2,
  and vLLM V1 "introduces minimal overhead." Latency cost is small relative to the
  ~11-14 tok/s decode ceiling we already accept. The grammar is reused → compile
  cached. Net: **low risk, high payoff**; this is the strongest single fix.
- **Schema vs raw-markdown-grammar — recommendation.** Constrain to a **JSON schema**
  with two string fields (`table_markdown`, `caption`), NOT a full GFM grammar. A
  strict GFM-table CFG is brittle (variable column counts, escaped pipes, multi-table
  concatenation) and easy to get subtly wrong, which would *itself* cause losses. The
  JSON-schema route guarantees the envelope closes (the actual bug) while leaving the
  markdown body free-form; combined with the `presence_penalty` from Q1 to stop the
  in-string loop, that covers both the structural and the degeneration failure mode.
  Note xgrammar may require disabling `any_whitespace` / using minified schema for
  some models; validate on a sample first (§4).
- **Caveat.** Guided decoding is a sidecar-side change (`docker-compose.yml` +
  `build_chat_payload`). The **in-process** transformers path can't use vLLM guided
  decoding; for it, the cheaper wins are the sampling penalties (Q1) + recovery
  parsing (Q4). Since `.env` routes us to `http` in practice (gpu-ops skill), focus
  guided decoding on the sidecar.

### Q3 — Robust table-extraction best practices (multi-engine + graceful degradation)

- **Two complementary engines, neither sufficient alone:** Docling's **structural
  grid** (`table_body`, deterministic, free, CPU) vs **VLM transcription** (handles
  merged/visually-complex tables but can degenerate). Best practice in the
  PDF-to-markdown VLM literature is exactly this hybrid: structural extraction for
  bulk + VLM for the blocks structural parsing mangles. Our pipeline *has* both
  signals already (`table_body` in bronze, VLM in modal) but the merge picks VLM-only
  and silently drops to a near-empty grid on VLM failure.
- **Page-spanning / merged cells** are the hard cases — these are precisely where
  Docling under-captures (2 rows) and where the VLM is most likely to loop. So the
  two engines fail on *correlated* inputs, which is why "VLM with grid fallback"
  isn't enough; we need a **third tier** and a **quality decision between tiers**, not
  just first-non-empty.
- **Graceful-degradation principle: content must NEVER be silently dropped.** The
  ranked degradation ladder should be (best→worst), each tier *gated by a quality
  check* before acceptance:
  1. VLM `table_markdown`, confidence 1.0, passes degeneracy gate → use it.
  2. VLM degenerate/failed → **try Docling `_table_body_to_gfm`** (already exists).
  3. Grid also degenerate/empty → **keep the VLM's partial raw text** (the 0.5
     `description` currently discarded) as a `<!-- table (unverified) -->` block so
     rows survive for RAG even if imperfect — better than total loss.
  4. Nothing usable → emit the `[table processing failed]` marker **and** raise a
     non-suppressible signal (metric + WARN/ERROR check) so a human can recover it
     (e.g. via the `recover-images`/manual refine path). Losing the row silently is
     the only true failure.
- A retry tier (re-run the VLM once with `presence_penalty` bumped / a different
  template) is viable but **budget-aware**: each retry is a full ~slow VLM call on an
  8GB card. Gate retries behind the degeneracy detector so only *flagged* tables pay
  the cost (typically a small minority).

### Q4 — QC GATE design (the core deliverable)

Design principles, consistent with `ingestion/CLAUDE.md`:
- Pure logic in a reusable validator under `src/tamubot/ingestion/validation/`
  (returns `CheckOutcome`); thin Dagster adapter in
  `checks/silver_modal_checks.py`. Mirror `text_quality.py` / `baseline_diff.py`.
- L1 = structural/invariant (blocking), L2 = run-over-run drift (warn), L3 =
  semantic/opt-in. Severity: **block** only what is unambiguously broken at this
  stage; **warn** for quality signals that need human judgment.

Proposed new validator module: `src/tamubot/ingestion/validation/table_quality.py`
(pure functions; unit-testable without GPU/Dagster — consistent with the
host-can't-load-defs constraint in MEMORY).

| # | Gate | Measures | Metric + threshold | Severity | Owning file | Surfaces as |
|---|---|---|---|---|---|---|
| G1 | **Degenerate-repetition detector** | repeated `\| \|`/empty-cell runs & single-token domination in `raw_response`/`table_markdown` | max contiguous empty-cell run ≥ 8, OR longest repeated n-gram (n=3, on cell tokens) coverage > 0.5, OR empty-cell ratio > 0.6 of all cells | **WARN** (per-stem) | validator `table_quality.is_degenerate_table()`; check in `silver_modal_checks.py` | L1 (per-partition) `v6b_silver_modal_no_degenerate_tables` |
| G2 | **Truncation / unterminated-output detector** | VLM hit the token cap without closing JSON | `len(raw_response_tokens) ≥ 0.98 * max_tokens` AND JSON did not parse (confidence==0.5/0.0) | **WARN** | same validator | L1 `…_no_truncated_modal_output` |
| G3 | **Table-content-survival gate** (the FID_TABLE_LOST gate) | every `table` block contributes ≥1 data row to the merged markdown | for each block with `type==table`: produced GFM has ≥ 2 lines beyond header+separator OR a non-empty grid fallback existed; count `tables_lost` | **ERROR/BLOCK** when `tables_lost>0` and no fallback rows | `_merge_to_markdown` instrumented in `silver_modal.py`; check in `silver_modal_checks.py` | L1 `v6b_silver_modal_no_table_lost` |
| G4 | **Confidence-floor gate** | fraction of low-confidence modal results | `low_confidence_blocks / total_modal_blocks` (already partly computed at `silver_modal.py:211`) > 0.25 | **WARN** | `silver_modal_checks.py` | L1 `…_confidence_floor` |
| G5 | **Failure-marker gate** | `[table processing failed`/`[image processing failed` markers leaking into the merged markdown | count of `processing failed]` substrings in `silver_modal_markdown` | **WARN** (>0), **ERROR** (table marker with no grid fallback) | reuse pattern of `text_quality.check_no_replacement_chars` | L1 `…_no_failure_markers` |
| G6 | **Modal-quality drift (L2)** | corpus regression run-over-run | `% tables transcribed` and `mean table confidence` vs median of last N (reuse `compute_baseline_delta`, `max_drift_pct≈0.15`) | **WARN** | `baseline_diff.py` (existing) + `silver_modal_checks.py` | L2 baseline-delta |

Notes:
- G1/G2/G3 are the **new** detection layer — none of today's two checks observe
  them. G3 is the direct counter to the motivating bug (and pairs with the v6b-review
  H1 merge fix).
- All thresholds above are **starting points** to calibrate on the eval corpus
  (§4) — emit the raw metric in check metadata first, set thresholds from the
  observed distribution before flipping any to BLOCK.

### Q5 — Observability / metrics (per-stem and per-corpus)

Per-stem metadata to add to the `MaterializeResult` in `_compute_silver_modal`
(`silver_modal.py:213-222`), beyond the existing `low_confidence_blocks`:
- `table_blocks_total`, `tables_transcribed` (VLM markdown accepted), `tables_grid_fallback`,
  `tables_partial_kept`, `tables_lost` (the headline number).
- `mean_table_confidence`, `degenerate_tables`, `truncated_tables`.
- `image_blocks_total`, `images_with_text`, `images_lost`.
- `modal_calls_made` (exists), and per-block latency if cheap to capture.

Per-corpus rollup (an unpartitioned `v6b_meta_modal_quality` asset, mirroring the
existing `v6b_meta_*` indexes, or a column in the Inspect judge report): `% tables
transcribed`, `% degraded-to-grid`, `% partial-kept`, `% lost`, `mean confidence`,
`degenerate-rate`. These tie directly to the `FID_TABLE_LOST` / `FID_IMAGE_LOST`
taxonomy codes so the deterministic check rates and the model-graded judge agree
(the judge "defers to deterministic checks for rates" — taxonomy doc).

---

## 3. Prioritized, concrete proposals (each with file:line, metric, severity, sketch)

Ranked by (content-correctness impact × cheapness). **Nothing below is implemented.**

### P1 — Wire `presence_penalty` into the table decode path  ⟵ highest ROI, lowest risk
- **Why:** directly attacks the degeneration root cause with the Qwen-recommended,
  table-safe lever; one-line config; no new failure modes.
- **Where:** `nuextract_http_client.py:42-48` (`build_chat_payload`) — add
  `presence_penalty` (≈1.3-1.5) and an optional `stop`. Mirror for the in-process
  path at `nuextract_client.py:170` via `generate(..., repetition_penalty=1.0,
  ...)`-style kwargs (transformers uses `repetition_penalty`/`no_repeat_ngram_size`;
  prefer a *small* `repetition_penalty≈1.05` there since transformers has no
  presence_penalty — calibrate carefully given the table caveat). Keep
  `temperature=0.0` for determinism.
- **Metric/threshold:** none (config); validated by G1 degenerate-rate dropping.
- **Severity:** N/A (fix). **Risk:** over-penalizing legitimately repeated cells —
  keep the value moderate and validate (§4).

### P2 — Add the degenerate-output recovery + survival gate (G1+G2+G3)
- **Why:** detection so a loss can never again be silent; pairs with the v6b-review
  H1 merge fix.
- **Where:** new pure validator `validation/table_quality.py`
  (`is_degenerate_table`, `empty_cell_ratio`, `max_repeated_cell_run`,
  `table_row_count`); thin checks in `checks/silver_modal_checks.py`; instrument
  `_merge_to_markdown` (`silver_modal.py:131-146`) to record per-table outcome
  (transcribed / grid-fallback / partial / lost) into block metadata.
- **Metric/threshold:** G1 (empty-cell run ≥8 OR empty ratio >0.6), G3 (`tables_lost>0`).
- **Severity:** G1/G2 WARN; G3 ERROR/BLOCK once calibrated.

### P3 — Enforce structured output via vLLM `guided_json` on the table call
- **Why:** *guarantees* the JSON envelope closes; removes the unterminated-JSON loss
  class entirely; cheap at our single-stream batch size.
- **Where:** `docker-compose.yml` (xgrammar is vLLM default — no flag needed; confirm
  version) + `build_chat_payload` (`nuextract_http_client.py:34-48`): add
  `extra_body={"guided_json": <2-field schema>}` / `response_format`. Schema =
  `{table_markdown: string, caption: string}`.
- **Metric/threshold:** none; validated by G2 truncation-rate → 0.
- **Severity:** N/A. **Risk:** small per-token overhead; possible xgrammar/whitespace
  quirks → validate on samples before corpus run.

### P4 — Close the fallback-ladder hole (keep partial VLM text instead of dropping it)
- **Why:** today a 0.5-confidence VLM result with usable rows is discarded by the
  merge; preserving it (tier 3 in §Q3) turns a total loss into a degraded-but-present
  table.
- **Where:** `_merge_to_markdown` table branch (`silver_modal.py:131-146`): when
  `table_markdown` empty AND grid fallback empty AND `modal_result.description`
  contains table-like rows, emit it under a `<!-- table (unverified) -->` marker.
- **Severity:** N/A (fix); feeds G5.

### P5 — Budget-gated single retry for flagged tables
- **Why:** recovers the minority of degenerate tables that P1+P3 don't fully fix.
- **Where:** `TableModalProcessor.process` (`modalprocessors.py:312-348`) or the
  `process_blocks` loop (`modalprocessors.py:422-427`): on G1-degenerate result, retry
  once with bumped `presence_penalty`; count against `V6B_MODAL_CALL_BUDGET`.
- **Severity:** N/A. **Risk:** doubles cost on flagged tables only — acceptable given
  they're rare; respects the 8GB serial ceiling.

### P6 — Observability rollup (metrics from Q5)
- **Why:** track table-parse quality over time; make `FID_TABLE_LOST` rate visible.
- **Where:** `_compute_silver_modal` metadata (`silver_modal.py:213-222`) + optional
  `v6b_meta_modal_quality` asset; G6 L2 drift check via `compute_baseline_delta`.
- **Severity:** WARN (drift).

---

## 4. Cheap experiments to validate (no full corpus run)

All are single-image or single-stem, well within the "≤10 LLM calls" budget rule.

1. **Repro + penalty sweep on the one bad table.** Re-send the saved degenerate
   table PNG to the sidecar (`POST /v1/chat/completions`) with the current params,
   then with `presence_penalty` ∈ {0.5, 1.0, 1.5} (and separately a small
   `repetition_penalty=1.05`). Confirm the `| |` loop stops and JSON closes. ~4 calls.
   Watch for the table-safe caveat: verify a *dense* table (the document's other,
   1.0-confidence table) still transcribes correctly under the chosen penalty (1
   call) — guards against the "penalty corrupts repeated cells" failure.
2. **Guided-json smoke test.** Same image with `extra_body={"guided_json": schema}`;
   confirm output parses first-try and check the latency delta vs unconstrained on
   this box. ~2 calls. (Cross-check xgrammar version/whitespace behavior.)
3. **Degeneracy-detector calibration (offline, 0 GPU).** Run the proposed
   `is_degenerate_table` over the saved `silver_modal/*.json` `raw_response`s in the
   eval corpus to set G1 thresholds from the real distribution before flipping G3 to
   BLOCK. Pure Python on the host (no model load — satisfies the host-can't-load-defs
   constraint).
4. **Grid-fallback coverage.** For the eval corpus, count how many `table_body`
   grids would yield ≥2 data rows — quantifies how often tier-2 fallback actually
   rescues a lost table vs how often we need tier-3/retry.

---

## 5. Risks / trade-offs given the 8GB GPU ceiling

- **Penalties are a scalpel, not a hammer.** The strongest external caution: raising
  `repetition_penalty` breaks legitimately repetitive table text. Use a *moderate
  presence_penalty* (Qwen-recommended) and validate on a dense table (Exp. 1). Do not
  enable `no_repeat_ngram` for tables.
- **Guided decoding overhead is real but minimized at batch 1-2.** Our `--max-num-seqs
  2`, single-stream, ~11-14 tok/s config is the *favorable* regime for xgrammar; the
  known perf cliffs are at batch ≥8, which we never hit. Still, measure the per-table
  latency delta (Exp. 2) before committing — the card is memory-bandwidth-bound and
  `--enforce-eager` (no CUDA graphs), so any added per-step work is felt.
- **Retries cost full VLM calls.** P5 must be budget-gated and detector-gated so only
  the rare flagged table pays — never blanket-retry. Respect `V6B_MODAL_CALL_BUDGET`
  and the serial-pass guidance (no fanning concurrency — it stalls the tiny KV pool).
- **In-process vs http divergence.** vLLM `guided_json` only helps the http path; the
  in-process transformers path needs the penalty + recovery route instead. Keep the
  two paths' behavior documented so they don't silently diverge (parity is currently
  assumed via greedy decoding).
- **Threshold churn.** Flipping G3 to BLOCK prematurely could red-gate good partitions
  on edge tables. Ship gates as WARN emitting raw metrics first; promote to BLOCK only
  after corpus calibration (Exp. 3). This matches the project's "emit metric, then
  set threshold" check philosophy.
- **`code_version_of` staleness trap.** Per `ingestion/CLAUDE.md` + review header:
  edits to a new `validation/table_quality.py` util won't mark `silver_modal` stale.
  Any fix touching merge behavior needs a **forced** re-materialize from
  `silver_modal` forward — call this out in the implementing PR.

---

## Sources (external claims)

- [Sampling Parameters — vLLM](https://docs.vllm.ai/en/v0.6.2/dev/sampling_params.html)
- [We Reduced LLM Repetition from 15% to 0% — Tony Seah (Medium)](https://tonyseah.medium.com/we-reduced-llm-repetition-from-15-to-0-and-parameter-tuning-wasnt-the-answer-e1a1cd811c3c)
- [vLLM vs TensorRT-LLM #3: Sampling Methods — SqueezeBits](https://blog.squeezebits.com/vllm-vs-tensorrtllm-3-understanding-sampling-methods-and-their-performance-impact-31921)
- [Structured Outputs — vLLM](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Structured Decoding in vLLM: A Gentle Introduction — BentoML](https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction)
- [Structured outputs in vLLM — Red Hat Developer](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses)
- [Guided Decoding Performance on vLLM and SGLang — SqueezeBits](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
- [Benchmarking VLMs for French PDF-to-Markdown — arXiv 2602.11960](https://arxiv.org/html/2602.11960)
- [Vision Language Models for Spreadsheet Understanding — arXiv 2405.16234](https://arxiv.org/pdf/2405.16234)
- [Repetition In Repetition Out: Neural Text Degeneration — arXiv 2310.10226](https://arxiv.org/pdf/2310.10226)
- [Understanding the Modern LLM Part 5: Text Degeneration — Inkyu Kim (Medium)](https://medium.com/@ikim1994914/understanding-the-modern-llm-part-5-understanding-text-degeneration-during-decoding-and-methods-966a4d33e9c8)
- [Qwen3-VL repo / generation config — QwenLM (GitHub)](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL infinite repetition bug — Issue #1611](https://github.com/QwenLM/Qwen3-VL/issues/1611)
- [Qwen3 presence_penalty discussion #1744](https://github.com/QwenLM/Qwen3/discussions/1744)
- [Vendor-recommended LLM parameter quick reference — Muxup](https://muxup.com/2025q2/recommended-llm-parameter-quick-reference)
- [How to use Qwen 3.5 to OCR documents — Martin Alderson](https://martinalderson.com/posts/how-to-use-qwen-3-5-to-ocr-documents/)
