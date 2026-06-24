# GEPA Prompt Optimization — Research Findings (for later work)

> Status: **research only, not yet implemented.** Captured 2026-06-10.
> Goal: evaluate GEPA for optimizing prompts in the v6b preprocessing pipeline and
> the RAG pipeline. Three threads: (1) what GEPA is + where it fits, (2) the metric
> design, (3) reusable open-source implementations.

---

## 1. What GEPA is

**GEPA (Genetic-Pareto)** is a *reflective prompt optimizer*. Instead of hill-climbing
a single eval number, it reads the **textual feedback** from each failure (errors,
judge rationales, traces), uses an LLM to *reflect* on why the prompt failed, and
mutates it. It keeps a **Pareto frontier across evaluation instances** (each candidate
best on at least one example), which avoids local optima.

Key facts (paper: Agrawal et al., arXiv 2507.19457, ICLR 2026 oral):
- Beats MIPROv2 ~+13%, beats RL (GRPO) ~+20% with **35× fewer rollouts**.
- Works with **as few as 10 examples**; budget ~**100–500 metric calls** (`max_metric_calls`).
- Produces prompts up to **9× shorter** than other optimizers.
- Ships two ways: `pip install gepa` (standalone) **and** `dspy.GEPA` (inside DSPy).

Standalone API: `gepa.optimize(seed_candidate, trainset, valset, task_lm,
reflection_lm, max_metric_calls, adapter=...)`. A custom `GEPAAdapter` implements:
- `evaluate(candidate)` — run the system with the candidate prompt, return **score +
  diagnostic text** ("actionable side information" = the *"text-optimization analogue
  of a gradient"*).
- `make_reflective_dataset()` — shape failures for the reflection LLM.

**Does NOT require DSPy.** Built-in adapters: DefaultAdapter (single prompt),
LangChain, Generic RAG (vector-store agnostic), MCP, plus others.

### The one hard constraint
GEPA only optimizes **text artifacts (prompts)**, and only works where you can return
a **per-instance scalar score + natural-language feedback**. It helps exactly where you
have (a) an LLM prompt and (b) a judge that emits a textual rationale. It does **nothing**
for deterministic code.

---

## 2. Fit in TamuBot's pipelines

### Preprocessing (v6b)

| Component | Prompt-driven? | GEPA fit |
|---|---|---|
| `silver_structured` (Gemini extraction) | ✅ LLM prompt | **Strong** |
| `silver_modal` (VLM image→table) | ✅ VLM prompt | Moderate (VLM prompts evolve less predictably) |
| `silver_tag` / boilerplate / dedup (`util/tagging.py`, `signature_index.py`) | ❌ deterministic | **None** |
| `chunker_v4.py` | ❌ deterministic | **None** |
| Inspect judge (boilerplate/dedup/chunking/fidelity) | ✅ judge prompt | Use it **as the feedback function** |

**Standout fit:** the `iterate-preprocessing` Inspect judge already emits per-dimension
verdicts with rationales (the `FID_*`/`BP_*`/`DUP_*`/`CHUNK_*` taxonomy in
`docs/preprocessing_error_taxonomy.md`). That rationale **is** GEPA's reflection input.
Natural application: wrap the **`silver_structured` extraction prompt** as the seed
candidate; use the **fidelity judge score + rationale** per syllabus as the feedback
function; run over a slice of the 100-syllabus eval corpus. Note: current fidelity fails
are mostly upstream Docling/bronze drops (deterministic, **not** GEPA-addressable) —
GEPA only helps the extraction-prompt-driven misses.

### RAG (higher-leverage fit)

| Surface | GEPA fit |
|---|---|
| Answer-generation prompt (`tools/llm.py` → `call_llm`) | **Strong** |
| Query reformulation (if/when added) | **Strong** |
| Context formatting (`format_context_xml`, primacy-recency) | Partial (instructional parts only) |
| Citation gate (`validate_citations_with_trace`, regex) | **None** (deterministic) |
| Retrieval / embeddings | GEPA tunes the *query/rerank prompt*, not the embedder |

The Generic RAG Adapter optimizes query reformulation + reranking + context synthesis +
answer generation **jointly**. Eval half is already built: **L3 `golden_recall_at_5`** +
golden set. Caveat: gains can be modest on small/local setups; larger with a strong
`reflection_lm` and a real eval set (which we have).

---

## 3. Metric design (DECIDED)

### Structural fact that shapes everything
GEPA's Pareto frontier is **across eval instances, not across objectives.** Per the
maintainers: *"GEPA doesn't maintain trade-off frontiers for those — it just sees
whatever single score your metric returns. So if you want multi-objective optimization,
you have to bake it into the metric yourself."* You get **no recall-vs-cost Pareto curve
for free.** Collapse everything into one scalar per instance.

Second free channel: the **feedback text** (read by the reflection LLM, NOT scored).
- **Scalar** = what gets optimized (encodes the priority ordering).
- **Feedback text** = why it failed — put cost/latency diagnostics here in plain English.

Metric return: `{'score': float, 'feedback': str}`.

### Don't use a naive weighted sum
`0.6*recall − 0.2*latency − 0.2*tokens` lets the optimizer **buy back** a recall loss
with token savings. Recall is paramount and must not be tradeable. ("If the right chunk
never lands in the candidate set, no downstream tuning can recover it.")

### Recommended: gated lexicographic with a small efficiency tie-breaker

```python
# per-instance, normalized to [0,1]
quality = 0.70 * recall_at_k        # primary: did the right chunk get retrieved
        + 0.30 * faithfulness       # precision/quality guard (no hallucination)

efficiency_penalty = min(0.10,      # capped: NEVER outweighs a quality delta
        w1*norm(tokens_generated) + w2*norm(tokens_retrieved) + w3*norm(latency))

score = quality - efficiency_penalty
if latency > HARD_BUDGET or tokens_retrieved > CTX_BUDGET:
    score = score * 0.5             # hard gate: bloated/slow answers clamped
```

- Recall stays the driver (0.70). Faithfulness = the precision/quality guard.
- Efficiency capped at 0.10 → prunes bloated prompts (reinforces GEPA's 9× shorter
  tendency) but can never trade away a real recall/faithfulness difference.
- Hard gate handles "I never want a slow/bloated answer" without polluting the linear region.
- All terms normalized per-instance against a max budget.

Feedback string carries the un-scored detail, e.g.:
> "recall@5=1.0, faithfulness=0.8 (one unsupported claim re office hours),
> tokens_generated=950, latency=2.1s within budget. Tighten the answer; drop the
> restated question."

### User decisions (2026-06-10)
- **Recall = most important metric.** Faithfulness guards precision.
- **Context-token reduction IS in scope** → must expose a **rerank / context-compression
  prompt** as an optimized component (otherwise GEPA has no lever to shrink retrieved
  tokens; the term would be near-constant and dead weight in the scalar).
- **Budgets: "the lower the better, no specific number"** → implement latency/token cost
  as the capped soft penalty + a relative hard gate (e.g. clamp candidates in the worst
  quantile of the current population) rather than an absolute threshold.

### Crucial caveat — GEPA can only move costs it has a *prompt lever* for
| Cost | GEPA can reduce? | Lever |
|---|---|---|
| tokens generated | ✅ | answer-gen prompt brevity |
| latency (generation) | ✅ partial | shorter outputs |
| tokens retrieved / context size | ⚠️ only if a **rerank/compression prompt** is optimized | else fixed by top-k |
| recall, faithfulness | ✅ | query reformulation + answer-gen prompts |

Inputs we already own: `golden_recall_at_5` (recall), an Inspect faithfulness grader
(to add), and **Langfuse traces** for `tokens_generated` / `tokens_retrieved` / `latency`.

---

## 4. Open-source implementations & applicability

| Project | What | Fit |
|---|---|---|
| **`gepa-ai/gepa` → `generic_rag_adapter`** | Official RAG adapter; optimizes query/rerank/synthesis/answer prompts jointly; vector-store agnostic | **★ Best — directly reusable** |
| `gepa-ai/gepa` → `langchain_adapter` | Any LangChain/LangGraph RAG | Only if we adopt LangChain (we don't) |
| DSPy `dspy.GEPA` + RAG tutorial | GEPA inside DSPy | Only if we rewrite RAG as DSPy modules — large cost |
| Kargar multi-agent RAG (article + gist) | Multi-agent ReAct RAG optimized w/ DSPy+GEPA | **Best pattern reference** — copy metric design, not code |
| SuperOptiX / Superagentic | Framework layering GEPA RAG opt | Low — framework dep for little gain |
| `CerebrasResearch/gepa` | Fork/mirror | Ignore — use upstream |

**Only one established reusable artifact: the official `generic_rag_adapter`.**

### `generic_rag_adapter` structure (`src/gepa/adapters/generic_rag_adapter/`)
- `vector_store_interface.py` — **ABC to subclass** for a custom store (`search`/
  `similarity_search` contract).
- `vector_stores/` — built-ins: ChromaDB, LanceDB, Milvus, Qdrant, Weaviate.
- `rag_pipeline.py` — retrieve → synthesize → generate flow.
- `evaluation_metrics.py` — retrieval/generation/combined scoring (**replace with our
  gated-lexicographic metric**).
- `generic_rag_adapter.py` — the GEPAAdapter wiring.

### Applicability to TamuBot
Blocker: our store isn't a built-in. We're on **MongoDB Atlas today, migrating to
pgvector** (see `project_postgres_migration`). Neither ships as a built-in. The adapter
is designed for this — subclass `vector_store_interface.py`. Integration path:

1. **Write one `PgVectorStore`** implementing `VectorStoreInterface` (thin wrapper over
   our retrieval call, ~50–80 lines). Target pgvector, not Atlas, since we're migrating off.
2. **Swap `evaluation_metrics.py`** for the gated-lexicographic metric, fed by
   `golden_recall_at_5` + Langfuse token/latency fields.
3. **Seed the four prompts** from current `tools/llm.py` generation prompt + a new
   rerank/synthesis prompt (the context-token lever).
4. Run `gepa.optimize(...)` with a budgeted `max_metric_calls`.

Less work than the DSPy route (full RAG rewrite) and avoids a LangChain dependency.

### Lessons to steal (from the multi-agent RAG reference)
- **Dual-mode metric**: LLM judge at program level + cheap heuristic checks (valid JSON,
  non-empty query, sane `k`, no premature stop) at predictor level, ~0.5/0.5. Heuristics
  are free and stabilize early iterations.
- **GEPA overfits early, generalizes late** — prompts get verbose then shorten; budget
  enough iterations, don't judge at iteration 2.
- **`add_format_failure_as_feedback=True`** — captures malformed outputs as feedback. On.
- Reference gains were ~4–8%/round on 2–3 full evals, 32 parallel threads — calibrates budget.
- **Strict judging** matters for factual domains (TAMU policy/course facts: plausible-but-
  wrong = failure).

---

## 5. Open decisions / constraints before implementing

- **LLM budget:** GEPA needs **100–500 metric calls**, each running system + judge LLM —
  far beyond the CLAUDE.md "ask before >10 LLM calls" rule. Any real run must be a
  deliberate, confirmed, budgeted run (cheap `task_lm`, strong `reflection_lm`, start
  `max_metric_calls≈100` on ~10–20 syllabi/queries).
- **Durability:** if we proceed, add `gepa` to `requirements.txt` + `pip install` (per
  CLAUDE.md durability rule).
- **Sequencing:** best first target = **RAG answer-gen prompt** (self-contained, eval
  harness exists, user-facing payoff). Second = **`silver_structured` extraction prompt**
  via the Inspect fidelity judge.

## Next concrete step (when picked up)
Scaffold, as a **dry-run (no LLM spend)**:
1. `VectorStoreInterface` subclass for pgvector.
2. The gated-lexicographic metric module wired to `golden_recall_at_5` + Langfuse fields.
3. Seed prompts incl. a rerank/compression prompt for context-token reduction.
Review wiring before authorizing a budgeted `optimize()` run.

---

## Sources
- GEPA paper — https://arxiv.org/abs/2507.19457 (ICLR 2026 oral)
- gepa-ai/gepa — https://github.com/gepa-ai/gepa
  - RAG guide — https://github.com/gepa-ai/gepa/blob/main/src/gepa/examples/rag_adapter/RAG_GUIDE.md
  - generic_rag_adapter — https://github.com/gepa-ai/gepa/tree/main/src/gepa/adapters/generic_rag_adapter
- dspy.GEPA — https://dspy.ai/api/optimizers/GEPA/overview/ ; DSPy RAG tutorial — https://dspy.ai/tutorials/rag/
- Multi-agent RAG w/ DSPy+GEPA (Kargar) — https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2
  - gist — https://gist.github.com/Diwas2055/2fcecb996de1bc8e75dcbd1be7530a72
- "Non-Obvious Things About GEPA" (one-scalar point) — https://www.elicited.blog/posts/non-obvious-things-about-gepa
- GEPA & Multi-Objective (Ax) — https://deepwiki.com/ax-llm/ax/7.4-gepa-and-multi-objective-optimization
- RAG metrics (recall/precision/faithfulness/composite) — https://www.digitalapplied.com/blog/rag-system-metrics-recall-precision-faithfulness-2026 ; https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more
- gepa on PyPI — https://pypi.org/project/gepa/
