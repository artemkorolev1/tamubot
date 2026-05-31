# RAG-Anything Vendor Notes

This directory contains code adapted from [RAG-Anything](https://github.com/HKUDS/RAG-Anything) by HKUDS for the v6b ingestion pipeline. See `/home/claude/.claude/plans/toasty-weaving-taco.md` for the design rationale.

## Upstream

- Repo: `github.com/HKUDS/RAG-Anything`
- Commit pinned: `8395634289f68445001219fd7f725965cf46a394` (2026-05-23)
- License: see upstream LICENSE (MIT-equivalent — verify before redistribution).

## Vendor strategy

Upstream `parser.py` is 2660 lines and `modalprocessors.py` is 1607 lines, both heavily coupled to LightRAG (KG storage, vector DB upserts, entity extraction). A literal copy-and-strip would leave fragile dead-code paths.

Instead the vendored files **adapt the contract and the prompts** while **rewriting the bodies fit-for-purpose**:

- **Block format** (`{type, text|img_path, page_idx, ...}`): preserved verbatim so block lists produced here interoperate with anything that reads RAG-Anything's native JSON.
- **Prompt templates** (`VISION_PROMPT`, `TABLE_PROMPT`, `IMAGE_ANALYSIS_SYSTEM`, `TABLE_ANALYSIS_SYSTEM`): lifted verbatim from upstream `raganything/prompt.py`.
- **Robust JSON parse + think-tag strip**: behavioral parity with upstream `BaseModalProcessor._robust_json_parse` and `_strip_thinking_tags`, but reimplemented inline.
- **LightRAG coupling**: removed. No imports of `lightrag`, `neo4j`, `merge_nodes_and_edges`, `knowledge_graph_inst`, `chunks_vdb`, `entities_vdb`, `relationships_vdb`. Modal processors return plain dicts; storage is the caller's responsibility.

## File-by-file

| File | Upstream source | Adaptation |
|---|---|---|
| `parser.py` | `raganything/parser.py` lines 68-693 (Parser base), 694-1454 (MineruParser) | Parser ABC contract kept. MineruParser is a Phase 2 placeholder that raises NotImplementedError. DoclingParser/PaddleOCRParser intentionally not vendored — Docling support lives in `tamubot.ingestion.converters.docling_block_adapter` so it can reuse the existing `DocumentConverter` resource. |
| `modalprocessors.py` | `raganything/modalprocessors.py` lines 832-1262 (Image, Table) + prompts from `raganything/prompt.py` lines 60-225 | Prompts lifted verbatim. Class bodies rewritten: no LightRAG inheritance, no async (sync API matches Dagster asset model), no entity-VDB / KG side effects. Cache decorator simplified to plain JSONL on a configurable path. |
| `batch_parser.py` | `raganything/batch_parser.py` | ThreadPoolExecutor pattern preserved; tqdm + RAGAnything orchestrator coupling removed. |
| `resilience.py` | `raganything/resilience.py` | Reduced to a single `retry_call` wrapper using `tenacity` (matches existing tamubot retry patterns). LightRAG-specific retry hooks dropped. |

## Update protocol

If upstream RAG-Anything ships a relevant change:
1. Read the upstream diff between the pinned SHA and HEAD.
2. Decide per-file whether the change touches contract (block format, prompts) or implementation.
3. Contract changes flow through here; implementation drift does not (we own these bodies now).
4. Bump the pinned SHA above on every sync, even when no behavior changes — proves the review happened.

## What we do NOT vendor and why

- `PaddleOCRParser`: not needed (decision in plan).
- `EquationModalProcessor`, `GenericModalProcessor`: syllabi don't have equations or generic content we want LLM-described separately.
- `BaseModalProcessor`, `ContextExtractor`: tied to LightRAG storage layout; equivalent functionality is inlined.
- Upstream `RAGAnything` orchestrator class: replaced by Dagster assets in `tamubot.ingestion.pipeline_v6b`.
- Upstream config (`config.py`): tamubot config lives at `tamubot.core.config`.
