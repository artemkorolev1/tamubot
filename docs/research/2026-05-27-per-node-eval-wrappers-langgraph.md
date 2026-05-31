# Research: Per-node eval wrappers for RAG/agentic pipelines (LangGraph + Langfuse, OSS)
*Generated 2026-05-27 · scout + 3 search agents + summarizer*

## TL;DR
The OSS landscape converges on one viable stack for your constraints: **Langfuse SDK (`create_score` with `observation_id`) as the emit target, with a thin wrapper that runs per-node scorers and parents an OTel `gen_ai.evaluation.result` event to the LangGraph node's span**. No single eval library is a drop-in for LangGraph nodes — Ragas/TruLens/DeepEval/Pydantic Evals are all scorer libraries whose outputs you must shim back into Langfuse Scores. The load-bearing design choices come from the academic literature, not the libraries: claim-level decomposition (RAGChecker beats RAGAS triad by ~14 Pearson on human correlation), ternary per-step labels for agent nodes, order-randomized judge calls (κ drops 0.807→0.639 without it), and `pass^k` over `pass@1`. Expect to defensively handle Langfuse LangGraph context bugs (#10721, #3729, #7749, #8573, #11221) — observation-id propagation inside nested nodes and async contexts is the recurring footgun. Build your own wrapper; use Ragas/DeepEval as pluggable scorers behind it.

## Your Requirements
- **Framework**: LangGraph
- **Platform**: Single-platform on Langfuse — wrapper must emit standardized scores to Langfuse spans, no second observability backend
- **Run modes**: Same wrapper used online (prod streaming, sampled) and offline (CI / golden-set batch)
- **Hosting**: OSS / self-hostable (rules out LangSmith, Braintrust, Galileo SaaS, Weave)
- **Granularity**: Per-node, not just end-to-end trace

## Candidates at a glance

| Solution | Activity | License | Language | Key strength | Key weakness |
|---|---|---|---|---|---|
| **Langfuse SDK (native scoring)** | Active (pushed 2026-05-27, 28k★) | MIT (+ EE) | TS backend, Py/JS SDK | Only OSS path to Langfuse Score schema with `observation_id`; same API online + offline | LangGraph context-propagation bugs (#10721, #3729, #7749, #8573) |
| **Ragas** | Stale main (~3 mo) (14k★) | Apache-2.0 | Python | Strongest RAG-specific metrics, native Langfuse cookbook | Trace-level scoring only in cookbook, no per-node `observation_id`; metric-name collision footguns |
| **DeepEval** | Active (2026-05-26, 15.7k★) | Apache-2.0 | Python | 50+ metrics, Pytest-style CI gate | Backend = Confident Cloud only, no documented Langfuse Score export |
| **TruLens** | Active (2026-05-25, 3.3k★) | MIT | Python | `TruGraph` auto-instruments LangGraph `@task` nodes; cleanest per-component feedback model | No native Langfuse export; needs custom bridge |
| **Pydantic Evals** | Active (2026-05-27, 17.3k★) | MIT | Python | OTel span-based evaluators (v1.83–1.87); clean Evaluator class API | Requires `logfire` for OTel access; no Langfuse emit; no longitudinal/baseline-diff |
| **agentevals (LangChain)** | Active | OSS | Python | Graph-trajectory evaluators on nodes-visited — closest open primitive to "per-node contract" | LangSmith-biased docs; thin library, you wire emission yourself |
| **OpenInference** | Active (2026-05-26, 990★) | Apache-2.0 | Python | OTel-native LangGraph instrumentation, Langfuse ingests via `/api/public/otel` | Instrumentation only, no scorers; schema-drift with Langfuse for some calls (#11221) |

## Detailed profiles (top 5)

### Langfuse (native scoring SDK)
- **What it is**: The observability backend you've already chosen, with a Score API (`name`, `value`, `trace_id` required, `observation_id` optional, `comment`, `data_type`, `config_id`). Same `create_score()` works online (per-span, in callback) and offline (batch, by trace/observation id). Backfill supported.
- **Best for**: The emit target. Every per-node score in your wrapper ultimately lands here.
- **Avoid if**: You need turnkey LangGraph node-level scoring out of the box — UI-driven evals are LLM-as-judge only; code scorers must be wired via SDK.
- **Benchmarks/metrics**: No public quality benchmarks (it's the substrate, not a scorer).
- **Community sentiment**: Strong OSS edge over LangSmith (HN #44840323); Feb 2026 official guide explicitly frames LLM-as-judge as same-API online + offline. Observation-level evals shipped 2026-02-13 (v3.155.0 self-host).
- **Known gotchas / open issues** (all load-bearing for your wrapper):
  - **#10721** — LangGraph `CallbackHandler` observations don't nest under active parent span; `parent_observation_id=` and `propagate_attributes()` both fail
  - **#3729** — `@observe` + LangChain `CallbackHandler` cannot retrieve current `observation_id` from inside a LangGraph node
  - **#7749** — `@observe` broken in async contexts (sibling instead of parent)
  - **#8573** — `langfuse.flush()` hangs when scoring inside LangGraph callback context (closed "not planned")
  - **#11221** — OpenInference LangChain auto-instrument breaks Langfuse trace rendering for LangGraph direct `model.invoke` (falls back to raw JSON)
  - **#8001** — open feature ask: `LangchainCallbackHandler` should register each observation as "current"
  - **#5925** — `run_id` propagation inconsistent in LangGraph vs LangChain
  - Practical impact: your wrapper needs an explicit `observation_id` resolver (don't rely on `@observe` context inside nested nodes), an async-safe path, and probably a non-blocking flush strategy (background queue, not `langfuse.flush()` in-callback).
- **Links**: `langfuse/langfuse` GitHub; `/api/public/scores`; `/api/public/otel` for OTLP ingestion.

### Ragas
- **What it is**: RAG metric library (faithfulness, answer relevance, context precision/recall, plus newer agent metrics). Official Langfuse cookbook exists.
- **Best for**: Pluggable RAG scorers behind your wrapper — especially retrieval-side metrics with ground truth.
- **Avoid if**: You expect per-node `observation_id` out of the box. The cookbook integration scores at **trace level only** via `langfuse.create_score(trace_id=...)`. You'll wire `observation_id` yourself.
- **Benchmarks/metrics**:
  - **RAGAS self-reported (arXiv 2309.15217)**: WikiEval pairwise agreement 95% faithfulness, 78% answer relevance, 70% context relevance.
  - **RAGChecker (NeurIPS 2024, arXiv 2408.08067)**: Pearson vs human across 280 pairs / 10 domains / 8 systems — **RAGAS 48.31** vs **RAGChecker 61.93** vs TruLens 35.15 vs ARES 17.81 vs BLEU 35.14. Human-human ceiling 63.67. **The two numbers disagree — RAGAS' own paper looks good in pairwise agreement, but RAGChecker's cross-system correlation puts the triad ~14 Pearson points below claim-level entailment.**
- **Community sentiment**: Mixed. Official LangGraph path requires LangChain message → Ragas message conversion (friction). Topuz (Medium 2026-04) flags **metric-name collisions causing silent failures** — `AgentCore Builtin.Faithfulness ≠ Ragas faithfulness` and swapping silently changes meaning. "Monolithic evaluator that just looks at final output misses everything in the middle" was the explicit driver for 3-layer per-node design.
- **Known gotchas**: main not pushed ~3 months as of 2026-05-27; closed issue #898 confirms Ragas LLM-judge calls don't attach to parent trace_id by default.
- **Links**: `explodinggradients/ragas`; Langfuse Ragas cookbook.

### DeepEval
- **What it is**: 50+ metrics, Pytest decorator pattern, LangChain `CallbackHandler` integration; component-level metrics declared in `metadata` of `BaseLanguageModel`s and tool decorators; `next_llm_span()` stages a metric for the next call.
- **Best for**: The CI / offline half of your wrapper — Pytest-style gating is the strongest in this space.
- **Avoid if**: You want a single backend. **DeepEval's native backend is Confident Cloud; no documented Langfuse Score export.** You'd run DeepEval as the scorer and write a small adapter that pushes its output into `langfuse.create_score()`.
- **Benchmarks/metrics**: No third-party human-correlation benchmark on the scale of RAGChecker's RAGAS/TruLens comparison.
- **Community sentiment**: Broadest metric coverage cited as primary strength. "Operational burden" called out when stacked with Ragas + Langfuse — practitioners warn against three-service deployments.
- **Known gotchas**: Confident Cloud lock-in; component-level metric declarations are framework-coupled (LangChain-style).
- **Links**: `confident-ai/deepeval`.

### Pydantic Evals
- **What it is**: Evaluator classes + span-based evaluators (`HasMatchingSpan` and friends). v1.83–1.87 (Apr 2026) added OTel evals and closed some LangGraph-gap features.
- **Best for**: Teams already in the Pydantic AI orbit; hybrid setups where PydanticAI agents are dropped into LangGraph nodes.
- **Avoid if**: You need longitudinal eval (baseline-diff, suite re-execution, dataset grouping) — these are missing. You also need `logfire` for OTel span access, and there's **no Langfuse emit path documented**. You'd build a custom OTel span processor + `create_score()` shim.
- **Benchmarks/metrics**: No public human-correlation numbers.
- **Community sentiment**: Positive on per-node ergonomics; weak on longitudinal eval; clean API.
- **Known gotchas**: Logfire dependency; manual Langfuse bridge.
- **Links**: `pydantic/pydantic-ai`.

### agentevals (LangChain) + OpenInference instrumentation
- **What it is**: `agentevals` provides graph-trajectory evaluators operating on **nodes-visited rather than messages** — the closest open primitive to a per-node contract. OpenInference provides `LangChainInstrumentor` (covers LangGraph) emitting OTLP spans that Langfuse ingests at `/api/public/otel`.
- **Best for**: Trajectory checks ("right answer, wrong path = critical failure" — Arize 2026) layered on top of node-level scorers; OpenInference for getting LangGraph spans into Langfuse via OTel rather than callbacks.
- **Avoid if**: You expect drop-in compatibility — agentevals docs are LangSmith-biased (library itself is OSS and Langfuse-compatible via callbacks), and OpenInference's "drop-in to any backend" marketing is contradicted by Langfuse issue **#11221** where the LangChain auto-instrument breaks Langfuse trace rendering for LangGraph direct `model.invoke` calls.
- **Benchmarks/metrics**: No human-correlation benchmark for trajectory evaluators specifically. The relevant data point is **AgentRewardBench (arXiv 2504.08942, Apr 2025)** — 12 judges over 1,302 trajectories, **no judge >70% precision**: GPT-4o 69.8%, Claude 3.7 Sonnet 68.8%, Llama 3.3 67.7%; rule-based 83.8% precision / 55.9% recall. **Don't trust standalone LLM-judge trajectory scores — pair with rule-based or state-based proxies.**
- **Community sentiment**: LangChain's own deep-agents post says ~half of useful tests are single-step node assertions; trajectory eval framed as "the actual hard part."
- **Known gotchas**: Schema drift with Langfuse (#11221); LangSmith-biased docs.
- **Links**: `langchain-ai/agentevals`; `Arize-ai/openinference`.

## Ruled out (single-platform OSS constraint)
- **LangSmith** — best LangGraph node UX, but non-OSS, per-trace pricing, 14-day default retention. Fails self-host constraint.
- **Braintrust** — SaaS only.
- **Galileo** — SaaS only.
- **Weave / W&B** — closed source despite self-hostable deploy. Rules out.
- **Arize Phoenix** — Elastic v2 license ("Other"), evaluators run inside Phoenix and aren't exportable to Langfuse Scores directly. Effectively a parallel observability stack, violates single-platform.
- **MLflow GenAI** — Apache-2.0 and has `@scorer` + `mlflow.genai.evaluate()`, but the Langfuse integration is **Langfuse → MLflow only (one-way)**, not reverse. Wrong direction for you.
- **TruLens** — MIT and `TruGraph` auto-instruments LangGraph, but feedbacks go to TruLens' own SQLite/Snowflake backend with no native Langfuse export. Would need a custom bridge; mention only.

## Benchmarks & Metrics — what the research shows

**RAG eval-framework human correlation** (RAGChecker, NeurIPS 2024, 280 pairs / 10 domains / 8 RAG systems):

| Framework | Pearson vs human |
|---|---|
| Human–human ceiling | 63.67 |
| **RAGChecker** | **61.93** |
| RAGAS | 48.31 |
| BLEU | 35.14 |
| TruLens | 35.15 |
| ARES | 17.81 |

Disagreeing number: RAGAS' own paper (arXiv 2309.15217, WikiEval pairwise) reports 95% / 78% / 70% agreement for faithfulness / answer relevance / context relevance. **Both numbers are real; they measure different things** (pairwise agreement on a curated set vs cross-system Pearson). Do not average. Context relevance is consistently the hardest dimension.

**Agent trajectory judge precision** (AgentRewardBench, arXiv 2504.08942):
- **No judge >70% precision** across 1,302 trajectories / 12 judges
- Best LLM judges: GPT-4o 69.8%, Claude 3.7 Sonnet 68.8%, Llama 3.3 67.7%
- Rule-based: 83.8% precision but 55.9% recall
- Failure modes: grounding mismatch, sycophancy, missed-instruction details, action-intent confusion

**LLM-as-judge order bias** ("Judging the Judges," IJCNLP 2025, 150k instances):
- Flipping answer order drops κ **0.807 → 0.639**
- Fleiss' κ for general judges sits modest 0.1–0.32; collapses to κ = −1.0 in some high-context domains (Polish legal exams)

**Agent benchmarks** (consistency, not accuracy):
- τ-bench (arXiv 2406.12045) — final-state only; GPT-4o <50% retail, **pass^8 <25%**; Step-3.5-Flash 0.882, GLM-4.7 0.874 (top)
- AppWorld (arXiv 2407.18901) — 750 tasks, state-based; GPT-4o ~49% normal / ~30% challenge; LOOP-trained Qwen2.5-32B ~71% TGC
- Takeaway: **state-based proxies > LLM judges**; capture pass^k not pass@1

**Human-LLM rating divergence**: arXiv 2509.26205 (2025) — "numeric LLM and human ratings lacked agreement" — store rationale, not just score.

## Wrapper design implications (load-bearing)

These six takeaways shape your wrapper's payload regardless of which scorer library plugs in. They come from the academic agent and should drive the schema, not be footnoted.

1. **Per-node payload schema** = `score` + `rationale` (string) + `judge_model_id` + `prompt_hash` + `order_randomization_seed`. Numeric scores alone diverge from humans (arXiv 2509.26205); rationale must be persisted alongside.

2. **Claim-level decomposition for RAG nodes**. RAGChecker beats RAGAS triad by ~14 Pearson points (61.93 vs 48.31) precisely because it decomposes to claim-level entailment. Don't ship a single scalar per RAG node; emit per-claim scores aggregated to a node-level summary.

3. **Retrieval nodes: prefer rule-based / IR metrics when ground truth exists.** AgentRewardBench shows rule-based wins on precision (83.8% vs ≤69.8% for any LLM judge). Recall@k, MRR, nDCG over LLM-judge of retrieval relevance.

4. **Agent step contract = ternary (+1/0/-1).** AgentProcessBench convention. Binary correctness over-prunes recoverable trajectories (AgentPRM, ToolPRMBench finding). Encode progress, not pass/fail.

5. **Capture `pass^k`, not `pass@1`.** τ-bench's pass^8 <25% for GPT-4o exposes the consistency gap. Your offline runs should sample k times per item and store the distribution.

6. **Emit `gen_ai.evaluation.result` events parented to the LangGraph node's span.** Merged in OTel semconv v1.39.0; carries `score.value`, `.score.label`, parents via span_id or `gen_ai.response.id`. Langfuse ingests OTLP at `/api/public/otel`. **Caveat**: `gen_ai.*` still Experimental as of 2026; OpenInference still co-exists. Emit both attribute sets via `OTEL_SEMCONV_STABILITY_OPT_IN` until stable.

Additionally — **order-randomization is mandatory** for any pairwise/preference judge call (κ 0.807→0.639 without). Either randomize per call or call twice with swap and average.

## Tradeoff matrix

| Axis | Langfuse-native SDK | Ragas | TruLens | DeepEval | agentevals | Pydantic Evals |
|---|---|---|---|---|---|---|
| OSS / self-host | Yes (MIT + EE) | Yes (Apache-2.0) | Yes (MIT) | Yes (Apache-2.0) | Yes | Yes (MIT) |
| Native Langfuse Score emit | Yes (the API) | Trace-level only via cookbook | No (custom bridge) | No (Confident Cloud) | Via callbacks | No (custom OTel→Score shim) |
| Per-node `observation_id` | Yes (manual) | No in cookbook | n/a (own backend) | n/a (own backend) | Node-visited primitive | Span-based, needs logfire |
| Same API online + offline | Yes (`create_score`) | Yes (`create_score`) | TruLens own | Pytest offline only | Yes | Yes |
| LangGraph coupling | Callback handler (buggy) | Message conversion needed | `TruGraph` auto | LangChain `CallbackHandler` | Direct (nodes-visited) | Hybrid via PydanticAI |
| Scorer library breadth | None (substrate) | RAG-focused | Composable feedback fns | 50+ metrics | Trajectory-focused | Generic + span |
| Human-correlation evidence | n/a | 48.31 Pearson (RAGChecker comparison) | 35.15 Pearson | None public | None public | None public |
| Critical gotcha | Context bugs #10721/#3729/#7749/#8573/#11221 | Silent metric-name collisions; main stale | No Langfuse path | Confident Cloud lock-in | LangSmith-biased docs | Logfire required |

## Decision guide

Five forks based on your constraints (LangGraph + Langfuse single-platform OSS, online+offline, per-node).

**Fork 1 — Default recommendation**: Build a thin in-house wrapper that (a) resolves the current LangGraph node's `observation_id` explicitly (don't rely on `@observe` context — bugs #3729, #7749), (b) runs pluggable scorers (start: Ragas for RAG nodes, agentevals for trajectory, rule-based IR for retrieval), (c) emits via `langfuse.create_score(observation_id=...)` AND an OTel `gen_ai.evaluation.result` event for forward-compatibility, (d) uses a background queue for flush to avoid #8573, (e) carries the full payload from "Wrapper design implications" §1.

**Fork 2 — If you're heavily async**: Bug #7749 means `@observe` produces sibling-not-parent observations. Avoid `@observe` for async LangGraph nodes; resolve `observation_id` from the callback handler's `run_id` manually and pass it into `create_score` directly. Add an integration test for nested async nodes before trusting any score.

**Fork 3 — If RAG quality is the main concern**: Claim-level scoring (RAGChecker-style) over the RAGAS triad. RAGAS the library is fine as a scorer, but configure it to emit per-claim faithfulness, not just the aggregate. The 14-Pearson-point gap is the single largest quality lever the literature surfaces.

**Fork 4 — If you have a trajectory / multi-step agent**: Layer agentevals on top of node-level scorers (nodes-visited primitive). Don't trust LLM-judge trajectory scores standalone — AgentRewardBench says no judge clears 70% precision. Pair with state-based / rule-based proxies wherever ground truth exists. Store `pass^k` distributions, not `pass@1`.

**Fork 5 — If you want one offline CI gate today, no custom code**: DeepEval Pytest decorators are the strongest off-the-shelf CI gate, but you'll write an adapter that catches DeepEval's results and re-emits via `langfuse.create_score()`. Accept that this is a one-way bridge and the wrapper still needs to exist for the online path.

## Sources

### GitHub (Search A)
- `langfuse/langfuse` — 28,004★, pushed 2026-05-27, MIT (+EE). Score schema, `/api/public/scores`, `/api/public/otel`.
- Langfuse issues #10721 / #3729 / #7749 / #8001 / #8573 / #11221 — LangGraph observation-id propagation, async `@observe`, callback nesting, flush hang, OpenInference rendering.
- Langfuse discussion #5925 — `run_id` propagation inconsistency LangGraph vs LangChain.
- `explodinggradients/ragas` — 14,074★, pushed 2026-02-24 (stale), Apache-2.0. Langfuse cookbook trace-level only.
- `truera/trulens` — 3,346★, 2026-05-25, MIT. `TruGraph` LangGraph auto-instrument.
- `confident-ai/deepeval` — 15,723★, 2026-05-26, Apache-2.0. Confident Cloud backend.
- `pydantic/pydantic-ai` — 17,325★, 2026-05-27, MIT. `HasMatchingSpan`, logfire dependency.
- `Arize-ai/openinference` — 990★, 2026-05-26, Apache-2.0. `LangChainInstrumentor` covers LangGraph.
- `Arize-ai/phoenix` — 9,855★, Elastic v2.
- `mlflow/mlflow` — 26,132★, Apache-2.0. `@scorer`, one-way Langfuse→MLflow.

### Papers (Search B)
- **RAGChecker** — arXiv 2408.08067 (NeurIPS 2024). 280 pairs / 10 domains / 8 systems. Pearson vs human: RAGChecker 61.93, RAGAS 48.31, TruLens 35.15. Claim-level entailment beats triad.
- **RAGAS** — arXiv 2309.15217. WikiEval pairwise: 95/78/70.
- **Human-centered RAG eval** — arXiv 2509.26205 (2025). LLM-human rating divergence; persist rationale.
- **EncouRAGe** — arXiv 2511.04696 (Nov 2025). Node-level + end-to-end split.
- **CoFE-RAG** — arXiv 2410.12248. Full-chain checklist.
- **AgentRewardBench** — arXiv 2504.08942 (Apr 2025). 12 judges, no judge >70% precision; failure-mode taxonomy.
- **τ-bench** — arXiv 2406.12045 (Sierra). pass^k methodology; GPT-4o pass^8 <25% retail.
- **AppWorld** — arXiv 2407.18901. State-based programmatic tests > judges.
- **AgentPRM / ToolPRMBench / AgentProcessBench** — ternary step labels (+1/0/-1).
- **"Judging the Judges"** — IJCNLP 2025. 150k instances; order-flip drops κ 0.807→0.639.
- **Scoring Bias** — arXiv 2506.22316 (2025). Formal bias dimensions.
- **Same-model judge bias** — arXiv 2509.26072.
- OTel GenAI semconv v1.39.0 — `gen_ai.evaluation.result` event; `OTEL_SEMCONV_STABILITY_OPT_IN`.

### Opinions / practitioner (Search C)
- HN #44840323 — Langfuse OSS edge over LangSmith.
- Topuz (Medium, 2026-04) — metric-name collision silent failures (`AgentCore Builtin.Faithfulness ≠ Ragas faithfulness`); 3-layer per-node design rationale; convergent pattern: callback tracing + thin decorator wrapper + agentevals + Ragas/DeepEval as pluggable scorers + same wrapper CI+prod.
- Arize 2026 post — "right answer, wrong path = critical failure"; trajectory eval as the hard part.
- LangChain deep-agents post — ~half of useful tests are single-step node assertions.
- Langfuse Feb 2026 official guide — LLM-as-judge online + offline through same API.
- Langfuse discussion #4845 — dataset-run linking friction.
- Ragas closed issue #898 — LLM-judge calls not attached to parent trace_id by default.
