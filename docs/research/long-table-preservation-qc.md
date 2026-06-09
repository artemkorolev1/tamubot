# Preserving Long Structured Tables Through Preprocessing + Chunking for RAG

Grounded research report for the TAMU syllabus pipeline (`assets/silver_modal.py` VLM extraction → `chunker_v4.py`). Motivated by iter_03 error Class 2: 15-week Course Schedule tables losing most rows in what RAG finally sees (CSCE_611, CSCE_682).

## TL;DR for the two failures

1. **Extraction (silver_modal):** row loss is the well-documented VLM long-table failure mode — sequence-length truncation + repetition looping + row merging — not just a decode-param issue. `presence_penalty=1.5` helps but is not sufficient. Add (a) a row-count validation gate, (b) tall-table tiling before extraction, (c) cache-key versioning so pre-fix extractions aren't served.
2. **Chunking (chunker_v4):** don't let a 15-week schedule split silently. Default to **keep-the-table-intact** (it fits a syllabus chunk budget); if it ever must split, use **row-window chunks that REPEAT the header row** + prepended caption/section path. Docling has a known bug where the header is NOT propagated to sibling chunks — precisely the silent-row-loss shape seen here.

## 1 — Chunking tables (ranked)

1. **Keep the whole table in one chunk when it fits.** Ragie and Docling default to the full table as one markdown chunk, splitting only on token limits. A 15-row × 4-col schedule is a few hundred tokens — it should never need splitting. If RAG sees only the first weeks, the table is fragmented/truncated upstream or split on too-small a budget. [ragie](https://www.ragie.ai/blog/our-approach-to-table-chunking); [docling](https://docling-project.github.io/docling/concepts/chunking/)
2. **If you must split, row-windows that REPEAT the header.** Ragie packs as many rows as fit, headers with each chunk, single-row split last resort. Docling exposes `repeat_table_header=True` (default) + `omit_header_on_overflow`. Without the header a retrieved row is an ambiguous fact. [ragie](https://www.ragie.ai/blog/our-approach-to-table-chunking)
3. **One-chunk-per-row is usually too granular** for narrow schedules (index bloat, near-dup retrieval, lost cross-row context). Row-windows (10–30 rows) are the middle ground; for 15 rows, one window = whole table. [rohan-paul](https://www.rohan-paul.com/p/how-to-handle-tables-during-chunking)
4. **Serialization: Markdown is the right default** (token-efficient, natively understood, good for BM25). [ragie](https://www.ragie.ai/blog/our-approach-to-table-chunking) Caveat from an 11-format benchmark: for pure value-lookup, `Markdown-KV` scored 60.7% vs `Markdown-Table` 51.9% but at 2.7× tokens — so keep markdown-table for storage/retrieval, consider KV expansion at answer-time if the generator mis-reads. [improvingagents](https://www.improvingagents.com/blog/best-input-data-format-for-llms/); [HtmlRAG](https://arxiv.org/html/2411.02959v1)

Whole-table-intact + markdown maximizes recall for "what's due in week 11." Row-splitting *without* repeated headers is the biggest recall killer.

## 2 — Making table chunks retrievable

1. **Prepend section path / caption / heading** before embedding (Docling `contextualize()`): e.g. `"<Course> — Course Schedule (15 weeks)\n\n<table>"`. [docling](https://docling-project.github.io/docling/concepts/chunking/)
2. **Anthropic Contextual Retrieval** — 50–100-token LLM context per chunk before embedding + BM25; −35% / −49% / −67% failures (embeddings / +BM25 / +rerank); ~$1.02 per M doc tokens with prompt caching. Re-attaches "weeks 9–15 of CSCE 670 schedule" to a split fragment. [anthropic](https://www.anthropic.com/news/contextual-retrieval); [cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
3. **Dual representation** — short table summary alongside raw rows (summary matches semantic queries, rows match exact lookups). [nanonets](https://nanonets.com/blog/table-extraction-using-llms-unlocking-structured-data-from-documents/)

## 3 — VLM long-table extraction reliability (silver_modal)

Documented failure modes match the symptom:
- **Truncation / sequence-length overflow** — large tables exceed VLM input limits; HTML targets worst (token redundancy). [TALENT 2510.07098](https://arxiv.org/pdf/2510.07098); [MinerU2.5 2509.22186](https://arxiv.org/pdf/2509.22186)
- **Repetition loops** — decoder loops without concluding; `presence_penalty` targets this but penalties alone don't guarantee completion. [DTS 2511.00640](https://arxiv.org/pdf/2511.00640)
- **Row merging / hallucination on long tables.** [TALENT](https://arxiv.org/pdf/2510.07098)

Mitigations (ranked):
1. **Reduce structural tokens** — emit markdown (compact) not HTML. IBM OTSL cuts structural tokens 28+→5, seq length ~50%. [MinerU2.5/OTSL](https://arxiv.org/pdf/2509.22186)
2. **Tile tall tables** (>~12–15 rows) into vertical strips with header carried into each, extract per-strip, merge by deduping the repeated header + concatenating bodies. [PubTables-v2 2512.10888](https://arxiv.org/pdf/2512.10888)
3. **Guided/constrained decoding** (vLLM `guided_json`/xgrammar) to force array-of-rows — but A/B it, schema constraints can hurt quality. [vLLM](https://docs.vllm.ai/en/stable/features/structured_outputs/); [SqueezeBits](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
4. **Decoding params** — keep penalties, raise `max_tokens` to cover the widest expected table (a too-small cap is itself a row-loss cause). [DTS](https://arxiv.org/pdf/2511.00640)
5. **Narration + OCR-span hybrid (TALENT)** for over-limit tables — a more robust version of the grid/text fallback ladder. [TALENT](https://arxiv.org/pdf/2510.07098)
6. **Retry on degeneracy**, gated on the §6 validation check, not just "looks empty."

## 4 — Should a long table split across chunks?

Only when it genuinely exceeds budget — a 15-week syllabus schedule essentially never does, so **keep it intact**. When unavoidable: repeat header in every fragment, overlap by one row, prepend caption/section path, assert Σ(rows) == detected rows. **Watch the Docling header-propagation bug** (#2975): on split, only the first chunk keeps the header, siblings get bare rows — the canonical silent-row-loss trap. Confirm chunker_v4 doesn't have this. [docling #2975](https://github.com/docling-project/docling/issues/2975)

## 5 — Frameworks out of the box

- **Docling** HybridChunker: hierarchy + token-aware; splits oversized tables, merges undersized peers, `repeat_table_header=True` default; `contextualize()` for embedding; MarkdownTableSerializer (the sibling-header bug). [docling](https://docling-project.github.io/docling/concepts/chunking/)
- **Unstructured**: each Table element is its own chunk; less hierarchy-aware.
- **LlamaIndex/LangChain**: table-as-node + generated summary for retrieval + raw markdown for answer (dual representation).

## Recommendation

### silver_modal
1. Emit **markdown rows** (not HTML).
2. **Tile tall tables** (>~12–15 detected rows) with header per strip; merge by header dedup. (Gate to long tables only — respects the ≤10-call budget and ~11–14 tok/s GPU ceiling.)
3. Keep penalties; **raise `max_tokens`** to cover widest strip.
4. Optional `guided_json` array-of-rows — A/B it.
5. **Version the content-keyed cache** — add a `decode_version` salt to the image+grid-hash key so pre-fix extractions aren't served (see memory `project-v6b-modal-cache-content-keyed`).

### chunker_v4
- **Keep rendered table as ONE chunk**; prepend section path + caption ("Course Schedule") + contextual sentence before embedding.
- Only if over budget: row-window split, header repeated per window, 1-row overlap, caption prepended, assert no row loss.

## Validation checks to assert (highest-leverage single action)

Add an asset check between detect → VLM → chunk:
1. **Row-count conservation (primary):** `markdown_data_rows(extracted) >= detected_grid_rows - tol`. If `<`, VLM truncated/looped → fallback/retry. Directly catches "only first weeks survive."
2. **Sequential/monotonic check:** Week column should be `1..N` contiguous; a gap = dropped rows.
3. **Chunk-level conservation:** `Σ data_rows across chunks == extracted rows`, and every chunk with table rows also has the header. Catches sibling-header bug + fragmentation.
4. **Degeneracy guard:** reject if any row repeats > k times or last N tokens are a repeating cycle.

Metric note: LLM-judge correlates with humans at r=0.93 vs TEDS 0.68 / GriTS 0.70 — combine a hard count check with an optional LLM spot-check. [GriTS 2203.12555](https://arxiv.org/pdf/2203.12555); [nanonets](https://nanonets.com/blog/the-ultimate-guide-to-assessing-table-extraction/)

## Sources
- https://www.ragie.ai/blog/our-approach-to-table-chunking
- https://docling-project.github.io/docling/concepts/chunking/ · https://github.com/docling-project/docling/issues/2975
- https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
- https://www.rohan-paul.com/p/how-to-handle-tables-during-chunking · https://pub.towardsai.net/rag-chunking-techniques-for-tabular-data-10-powerful-strategies-aba887de331e
- https://www.improvingagents.com/blog/best-input-data-format-for-llms/ · https://arxiv.org/html/2411.02959v1
- https://arxiv.org/pdf/2510.07098 · https://arxiv.org/pdf/2509.22186 · https://arxiv.org/pdf/2511.00640 · https://arxiv.org/pdf/2512.10888
- https://docs.vllm.ai/en/stable/features/structured_outputs/ · https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang
- https://www.anthropic.com/news/contextual-retrieval · https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide
- https://arxiv.org/pdf/2203.12555 · https://nanonets.com/blog/the-ultimate-guide-to-assessing-table-extraction/ · https://nanonets.com/blog/table-extraction-using-llms-unlocking-structured-data-from-documents/
