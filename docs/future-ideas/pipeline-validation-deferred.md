# Pipeline Validation — Deferred Items

Parked from the 2026-05-27 research round. Revisit when the trigger condition fires.

## 1. Awaiting bake-off completion
- **Wire v6c silver stages** — mirror v6b after letter-drop bake-off picks a winner. Trigger: bake-off decision.
- **Wire pipeline_v6 (pure VLM) into Dagster** — currently orphan assets only. Trigger: bake-off decision.
- **v5 pipeline check expansion** — user excluded v5 from this round. Trigger: explicit user request.

## 2. Cost / API budget gated
- **LLM-as-judge for table extraction** — Claude judge r=0.94 vs TEDS r=0.68. Worth it for golden subset only. Trigger: stable table corpus + budget approval.
- **Expanded `golden_recall_at_K`** beyond opt-in flag (currently L3 warn-only behind `RUN_GOLDEN_RECALL_CHECK=true`). Trigger: Voyage budget headroom in CI.
- **Per-page VLM confidence scoring** (pdfmux 5-heuristic pattern). Trigger: VLM stage stable + budget approval.

## 3. Requires new infrastructure
- **Embedding drift detection** (FED, MMD, persistent homology via drift-lens-monitor). Trigger: model upgrade event.
- **Cosine-distance check on re-embed sample** — requires periodic re-embedding job. Trigger: scheduled re-embed cron exists.
- **IVF centroid balance monitoring** — Atlas Vector Search abstracts this away. Trigger: migrating off Atlas.
- **Schema-on-write contracts** (Avro/Protobuf, Confluent/Glue). Trigger: multi-producer ingestion.
- **External frameworks** — Great Expectations, deepchecks NLP, dlt schema contracts. Trigger: validation logic outgrows in-house helpers.
- **Langfuse extension to ingestion-side observability** — chunk quality scores as traces. Trigger: chunks need cross-run UI beyond Dagster.

## 4. Needs custom detection logic
- **`no_mid_table_splits` check** — chunker_v4 doesn't flag these. Trigger: at least 3 confirmed mid-table-split failures in pilot.
- **Near-duplicate detection at index time** (MinHash/SimHash, chunk-level semantic dedup at 0.90–0.95). Trigger: index dedup rate > 5% on golden queries.
- **HOPE / Boundary Clarity / Chunk Stickiness metrics** — research-paper-fresh, requires implementation. Trigger: chunk-quality regression that current checks miss.

## 5. Excel / reporting layer
- **Run-over-run regression sheet** in existing pilot reports. Trigger: stakeholders want diff visibility outside Dagster UI.
- **Consolidated validation dashboard** across pipelines. Trigger: 3+ pipelines wired (currently 2).
- **Dagster freshness policies** + freshness alerts. Trigger: stale data has caused at least one wrong answer.

## 6. Asset-side changes that unlock more checks
- **v6b `silver_tag` actual cosine-sim implementation** (currently no-op pass-through). Trigger: boilerplate reference parquet built.
- **TEDS-style table-structure ground truth** on golden subset. Trigger: table extraction becomes blocking quality issue.
