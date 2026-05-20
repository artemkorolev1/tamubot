# Follow-ups — curated20_v2 ISEN eval (run `curated20_v2_baseline_20260519_20260519_231827`)

Source run: `tamu_data/evals/reports/eval_curated20_v2_baseline_20260519_20260519_231827.xlsx`
Golden set: `tamu_data/evals/golden_sets/ragas_20260519_curated20_v2.xlsx` (+ `_UPDATED.xlsx` with `run:<exp>:context_precision/recall` columns)

Aggregate (n=17 with scores, 3 NULL): mean `context_precision` 0.602, mean `context_recall` 0.653, router accuracy 85%.

This doc lists the issues we did **not** fix in this session, grouped by where the fix lives.

## A. Router / retrieval bugs (in `src/tamubot/rag/...`)

### A1. Router classifies `out_of_scope` on author-only buried-detail queries — `q17`
- **Observed**: "What is the title of the recommended resource material authored by Lee, J.D., Wickens, C.D., Liu, Y., and Ng Boyle, L.?" → `out_of_scope`, 0 chunks, NULL scores.
- **Pattern**: queries that name a *person* or *textbook* without naming a course code or topic keyword fall through. This was a red-team probe in the eval set; current router has no path for it.
- **Fix surface**: router prompt needs an example for author/textbook lookups, or a fall-through to `semantic_general` instead of `out_of_scope` when the query mentions proper nouns but no course code.

### A2. Router classifies `out_of_scope` on "Credit X" framing — `q3`
- **Observed**: "For a course designated as 'Credit 3', what are the fundamental concepts and advanced techniques of engineering economic analysis…" → `out_of_scope`, 0 chunks, NULL scores.
- **Pattern**: "Credit 3" looks like a course identifier to the model but isn't — it's the credit-hours field. Router treats it as a missing-course-code case.
- **Fix surface**: either teach the router to ignore "Credit X" tokens and fall through on the rest of the query, or drop this question from the golden set (the stem is genuinely ambiguous).

### A3. Hybrid retrieval returns 0 chunks for a correctly-routed course — `q5`
- **Observed**: "Is a graduate classification a prerequisite for the Management of Engineering Systems course?" → router classified `hybrid_course` correctly (no hallucination on re-run), but retrieval returned 0 chunks → NULL scores.
- **Pattern**: prerequisite information is in the chunk content, but neither BM25 nor semantic match surfaced anything above threshold for the rewritten query.
- **Fix surface**: investigate whether the syllabus chunk that carries the prerequisite line is in the index; check the threshold; consider adding a guaranteed-anchor "Catalog Description" chunk to hybrid retrieval.

### A4. Router hallucinated a course code on the first run, self-corrected on re-run — `q5`
- **Observed**: first run produced `course_ids=['ISEN 601']` for the Management of Engineering Systems question (no such course in the corpus); re-run produced the correct course IDs.
- **Pattern**: LLM nondeterminism on a borderline classification. Even when it self-corrects, downstream retrieval still failed (see A3), so masking this in a re-run is not a real fix.
- **Fix surface**: post-process router output to validate `course_ids` against a known-course list and drop any that don't match.

### A5. `course_summary` route returned 1 anomalous chunk with empty content — `q2`
- **Observed**: first run took `hybrid_course`, re-run took `course_summary` (router flipped paths between runs — LLM nondet.). On re-run, the retrieval node returned exactly 1 chunk with `score=0`, empty anchor, missing content fields. `cp=1.0` is misleading because there's nothing to judge against `cr=0`.
- **Fix surface**: drop empty/zero-score chunks from the retrieval output before the metric layer ever sees them; investigate the `course_summary` path's fallback when no real chunk matches.

### A6. Retrieval over-concentrates on a single course — `q11`, `q18`, `q20`
- **Observed**:
  - `q11` ("6σ methods"): all 4 chunks from ISEN 645 — recall perfect, precision diluted because not every chunk is 6σ-specific.
  - `q18` ("simulation courses"): 12 of 20 chunks from ISEN 625 alone; reference expected a 4-CRN synthesis.
  - `q20` ("Exam 2 at 30%"): retrieved 20 `Grading Policy` chunks but the right course's % can't be inferred from anchors.
- **Pattern**: `semantic_general` with no diversity penalty over-represents the top-ranked course.
- **Fix surface**: add per-course cap (e.g. max-3-per-course) or MMR-style diversity to `semantic_general` retrieval before reranking.

## B. Golden-set defects (in `tamu_data/evals/golden_sets/ragas_20260519_curated20_v2.xlsx`)

These items will keep scoring low even after pipeline fixes because the reference itself is wrong/narrow.

### B1. `q8` — `reference_answer` conflates two courses into one
- **Observed**: question asks for a course covering BOTH "human information processing" AND "management of engineering projects". Retrieval correctly returned ISEN 635 (HF) + ISEN 608 (PM). The reference_answer says **"ISEN 608 provides instruction on human information processing… and also teaches the management and leadership of engineering projects"** — attributing both topics to 608.
- **Fix**: rewrite the reference_answer to acknowledge the two-course answer, or split q8 into two items.

### B2. `q15` — `reference_answer` is right but RAGAS judge under-detects the match
- **Observed**: top-1 retrieved chunk literally is `ISEN 608 — Project Management: A Strategic Managerial Approach` (the Wiley textbook in the reference). Judge scored `cp=0, cr=0` on both runs.
- **Pattern**: this is either RAGAS judge weakness on conjunctive queries ("Wiley AND PM") or a chunk-text-payload truncation issue in `EvalInputs.contexts`.
- **Fix**: dump the exact `contexts` payload sent to RAGAS for q15 and confirm whether the textbook line was included; if yes, switch to `llm_context_precision_with_reference` or a stricter judge variant.

### B3. `q19` — `reference_answer` excludes the most-obvious course
- **Observed**: question: "I wanna lern about lean and manufacturin." Reference picks ISEN 689 + ISEN 665. Retrieval returned 17 chunks all from **ISEN 645 "Lean Engineering"** — the most direct match for "lean".
- **Pattern**: the synthesizer built a multi-course narrative that omitted ISEN 645.
- **Fix**: rewrite the reference_answer to include ISEN 645, or replace the item entirely.

## C. Adversarial probes — working as designed

### C1. `q20` numeric trap (Exam 2 = 30%)
- Anchors-only judging can't expose the specific percentage. Keep the item; the score itself is the signal that anchor-only retrieval / generation is insufficient for numeric-precision questions.

## D. Observability hygiene

### D1. Langfuse SDK regression — `'TraceClient' object has no attribute 'update'`
- **Observed**: when re-running with `--ids`, the code path that tags prior traces `superseded` raises `AttributeError`. Non-blocking, but the "supersede" semantics are broken until fixed.
- **Fix surface**: `src/tamubot/evals/_runner_common.py` (or wherever `tag_trace_superseded` lives) — adapt to the SDK's current trace-mutation API.

### D2. FUSE-lock fallback workflow
- When the host has the golden-set `.xlsx` open in Excel, the runner falls back to writing `<file>_UPDATED.xlsx`. The lock-fallback bug (each call clobbering the previous) is **fixed** in `golden_set.py` this session — columns now accumulate in the fallback file. Operational note: close the file in Excel before running an eval if you want updates in the original.

## Suggested ordering

1. A1 + A2 — router edge cases (easy prompt-level fix, unlocks 2 NULL items).
2. A5 — drop empty chunks from retrieval output (one-line guard).
3. B1 / B3 — rewrite the two clearly-wrong reference_answers.
4. A6 — retrieval diversity (touches `semantic_general` node, biggest expected delta).
5. D1 — Langfuse SDK adapter (small, cosmetic but useful for re-run hygiene).
6. A3 / A4 — deeper investigations (index coverage + post-route validation).
7. B2 — RAGAS judge calibration probe (need a reproducer harness before changing the judge).
