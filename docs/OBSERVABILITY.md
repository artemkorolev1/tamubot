# TamuBot Observability Runbook

How to monitor, interpret, and act on telemetry from the TamuBot RAG pipeline using Langfuse and RAGAS.

**Langfuse project:** https://cloud.langfuse.com/project/cmlyjfvy200qbad07ezy65y21

---

## 1. Where to Find Things in Langfuse

### Traces tab
**URL:** `/project/.../traces`

Each user query produces one trace. Trace names follow the convention:
- `tamubot.request` — production (app.py)
- `tamubot.probe` — probe runs (run_probe.py)
- `tamubot.benchmark` — benchmarks and chunking evals

**What to look at in a trace:**
- **Input / Output** at the top — original question and final answer
- **Scores section** — RAGAS scores (0–1 scale), failure scores (-1)
- **Timeline** — visual waterfall of all spans
- **Span tree** — expandable hierarchy of all stages

### Scores tab
**URL:** `/project/.../scores`

Plots all RAGAS scores across traces over time. Filter by score name and date range.

### Generations tab
**URL:** `/project/.../generations`

Every `node.generator` generation appears here with model name, token counts, and latency.

---

## 2. The Trace Hierarchy — Span Naming

Spans are named `node.<node>[.<sub-op>]` to reflect that each is a single graph-node operation (renamed from `pipeline.*` in May 2026).

```
tamubot.request  (Trace)
├── node.router                    (Generation)
├── node.retrieval.embed           (Span) — Voyage voyage-3 embedding
├── node.retrieval.search.hybrid   (Span) — MongoDB Atlas hybrid search
├── node.retrieval.search.semantic (Span) — MongoDB Atlas semantic search
├── node.retrieval.rerank          (Span) — Voyage rerank-2
├── node.generator                 (Generation) — main answer
├── node.generator.comparison      (Generation) — multi-course comparison
├── node.router.recursive          (Generation) — recursive routing
└── node.history.summary           (Generation) — conversation summary
```

**Cache hits:** When a node hits the session cache, the span still appears but has `cache_hit: true` in metadata.

**Absent spans mean:**
- No retrieval spans → intent was `out_of_scope`
- No `node.generator` → retrieval crashed (check for errors)

**Saved Langfuse filters:** any dashboard or saved search matching `pipeline.*` must be updated to `node.*` after this rename.

---

## 3. Observability Config System

All tracing and evaluation is controlled by `rag/observability/`:

```python
from rag.observability import prod_config, probe_config, benchmark_config, chunking_config
from rag.observability import create_trace, finalize_trace, EvalInputs, run_evals
```

### Preset factories

| Factory | Caller | Trace Name | Evals |
|---------|--------|------------|-------|
| `prod_config()` | app.py | `tamubot.request` | none |
| `probe_config(ragas=True)` | run_probe.py | `tamubot.probe` | faithfulness, answer_relevancy (async) |
| `benchmark_config(ragas=True)` | run_benchmark.py | `tamubot.benchmark` | faithfulness, answer_relevancy (sync) |
| `chunking_config(ragas=True)` | eval_chunking.py | `tamubot.benchmark` | context_precision, context_recall (sync) |

### Eval blocks

Declarative blocks with retry and failure scoring:
- **faithfulness** — RAGAS Faithfulness (requires answer)
- **answer_relevancy** — RAGAS AnswerRelevancy (requires answer + embeddings)
- **context_precision** — RAGAS ContextPrecision (requires reference)
- **context_recall** — RAGAS ContextRecall (requires reference)

On failure after retry: score = -1 posted to Langfuse with failure metadata.

---

## 4. RAGAS Scores — How to Interpret Them

### Faithfulness (0 to 1)
Measures whether every factual claim in the answer is grounded in the retrieved context.

| Score | Meaning | Action |
|-------|---------|--------|
| 0.8 - 1.0 | Answer fully grounded | None |
| 0.5 - 0.8 | Some claims lack context support | Review retrieval quality |
| 0.0 - 0.5 | Significant hallucination risk | Investigate |
| -1.0 | Evaluation failed | Check logs for RAGAS errors |

### Answer Relevancy (0 to 1)
Measures whether the answer addresses the user's question (embedding-based).

### Context Precision / Recall (0 to 1)
Retrieval quality metrics — require a reference answer for computation.

---

## 5. Routine Monitoring Checklist

### After each significant code change
1. Open **Traces** -> run 3-5 representative queries
2. Verify spans use dotted-lowercase names and nest correctly
3. Check **Scores** after ~30s -- confirm metrics are populated
4. Compare average scores to baseline

### After a data refresh
1. Run one query per intent type
2. Check retrieval spans for chunk counts > 0
3. Watch for faithfulness drop from near-duplicate chunks

---

## 6. Diagnosing Common Problems

### "Traces not appearing in Langfuse"
- Check `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`
- Look for `Langfuse SDK client initialised.` in logs
- Traces appear after `finalize_trace()` calls `lf.flush()`

### "RAGAS scores not appearing"
- Probe RAGAS runs async — wait 10-30s then refresh
- Benchmark RAGAS runs sync — scores appear immediately
- Check logs for `Eval block '...' failed`
- Score of -1 means evaluation attempted but failed

### "faithfulness = 0.0 on every query"
- Generator producing non-verifiable content (truncated tables, empty context)
- Check retrieval span metadata for chunk counts

### "answer_relevancy = NaN"
- Voyage AI embedding call failed — check API key and quota

---

## 7. Implementation Reference

| File | Role |
|------|------|
| `rag/observability/__init__.py` | Public API re-exports |
| `rag/observability/config.py` | `ObservabilityConfig` dataclass + preset factories |
| `rag/observability/tracing.py` | `get_langfuse()` singleton + `create_trace()` + `finalize_trace()` |
| `rag/observability/evals.py` | `EvalBlock` base, registry, `run_evals()` runner, critic LLM factory |
| `rag/observability/ragas_blocks.py` | Concrete blocks: faithfulness, answer_relevancy, context_precision, context_recall |
| `app.py` | Creates trace via `prod_config()`, calls `finalize_trace()` |
| `evals/run_probe.py` | Uses `probe_config()` + `run_evals()` for async RAGAS |
| `evals/run_benchmark.py` | Uses `benchmark_config()` + `run_evals()` for sync RAGAS |
| `evals/eval_chunking.py` | Uses `chunking_config()` + `run_evals()` for retrieval RAGAS |
| `config.py` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` |
