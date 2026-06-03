# tamubot.ingestion

## Contract

Producer for `tamubot.rag.models` — import schema models from there, never define them here.

## Pipeline

Full run: `make scrape-catalog && make scrape-classes` → `process_syllabi_v3.py --department CSCE` → `setup_atlas` → `ingest`. **Always run all steps together** — `--step N` is for debugging only, partial runs leave downstream stale.

Reset catalog crawl: delete `tamu_data/scraper/logs/progress_log.txt`.

## Gotchas

- **U+FFFD chars**: PyMuPDF emits replacement chars for un-decodable bytes. `clean_replacement_chars()` handles post-parse.
- **Boilerplate registry** (`boilerplate_stripper.py`): font-annotated headers → `BOILERPLATE_REGISTRY`; body-size → `BODY_BOILERPLATE_HEADERS`. Only add long, unambiguous phrases to body list.
- **`_BP_KEYWORDS`** in `process_syllabi_v3.py`: flags non-stripped headers as new candidates → `new_bp_candidates` column. Expand when new patterns emerge.

## NuExtract backend

Structured extraction has two backends, selected by `NUEXTRACT_BACKEND` (default `in_process`). Set `NUEXTRACT_BACKEND=http` + `NUEXTRACT_SERVER_URL` in `.env` to route extraction to the vLLM sidecar (`docker/vllm-nuextract/`, launched host-side); the in-process transformers+fla path stays the default and fallback. `get_extractor()` in `clients/nuextract_client.py` is the single entry point.

## Dagster UIs

Two pipelines, two UIs (separate processes — different ports):

```bash
make dagster-v6b   # http://localhost:3000
make dagster-v6c   # http://localhost:3001
```

Each UI shows the asset graph, materialization history, and inline check status badges. Click an asset → "Checks" tab for per-check metadata (including run-over-run delta % for L2 checks).

Asset checks: see `pipeline_v6{b,c}/checks/` for L1/L2/L3 definitions. Reusable validators live in `tamubot/ingestion/validation/`.

L3 `golden_recall_at_5` is opt-in: `RUN_GOLDEN_RECALL_CHECK=true V6B_INGEST_ENABLED=true dagster asset materialize ...`

## v6b preprocessing

Stages (per-stem, partitioned): `bronze_blocks` (Docling) → `silver_modal`
(vision, **default no-op**) → `silver_chunk_semantic` → `silver_tag_semantic`
(boilerplate + within/cross dedup) → `silver_embed` → `silver_atlas_upsert`.
`silver_structured` is a parallel branch off bronze (Gemini extract, NuExtract
vision fallback). Corpus-level meta indexes are **unpartitioned + manual**.
Full review + open findings: `docs/v6b-preprocessing-review.md`.

Gotchas:

- **Two-phase dedup (ordering not in the DAG).** `meta_boilerplate_reference` and
  `meta_chunk_signature_index` are built *from* `02_chunk`, but `silver_tag` deps
  only on `02_chunk` — so a first run tags **0** boilerplate/dedup. Sequence:
  run chunk → materialize both `v6b_meta_*` → **then** (re-)run `silver_tag`.
- **`code_version_of` only hashes the asset's own function** (`pipeline_v5/util.py`).
  Edits to `util/tagging.py` / `util/text_normalize.py` do **not** mark
  `silver_tag` stale — Dagster shows it current while behavior changed. After a
  util-only fix, **force** a re-materialize (`--select v6b_silver_tag_semantic …`).
- **Tag parquet cache is content-keyed.** `silver_tag` caches the reference /
  signature indexes by `(path, sha256)` — a same-path rebuild *is* picked up
  mid-process. (Was a path-only key → silently served the stale/empty index.)

Code → error-taxonomy ownership (`docs/preprocessing_error_taxonomy.md`):

| Code | Owns |
|---|---|
| `util/tagging.py`, `util/boilerplate_clustering.py`, `util/signature_index.py` | `BP_*`, `DUP_*` |
| `chunker_v4.py` | `CHUNK_*` |
| `assets/silver_modal.py`, `assets/bronze_blocks.py`, `converters/docling_block_adapter.py` | `FID_*` |
