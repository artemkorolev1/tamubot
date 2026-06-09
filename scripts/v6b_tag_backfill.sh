#!/bin/sh
# v6b tag PASS 2 as a *grouped backfill* — the cohort-page replacement for the
# per-stem `asset materialize` loop in v6b_tag_phase.sh.
#
# Why: looping `dagster asset materialize --partition <s>` launches one "jobless"
# run per file, so the 20 files scatter across 20 runs with no shared view. A
# backfill of v6b_tag_job launches all of them under ONE backfill, giving a native
# Runs -> Backfills page that groups every file's tag run + per-partition check
# status — the "one screen for the cohort" view. (Tag is CPU-only, so a backfill
# won't contend for the GPU; do NOT backfill the GPU parse stages this way.)
#
# Honors the two-phase dedup ordering (project-v6b-dedup-two-phase): both meta
# indexes scan 02_chunk, so they are (re)built once here BEFORE the tag backfill,
# or dedup tags 0. Assumes PASS 1 (bronze->modal->chunk) already produced 02_chunk.
#
# Usage (inside tamubot-dev-1, cwd /workspace):
#   sh scripts/v6b_tag_backfill.sh [STEMS_FILE]
# STEMS_FILE: one canonical_stem per line (default: the preprocessing-lab set).
#
# After it finishes, see the grouped cohort under Runs -> Backfills, and run the
# ledger scoped to this cohort for the per-check rollup:
#   V6B_LEDGER_LAST_RUNS=<N> dagster asset materialize \
#     --select v6b_pipeline_ledger -f <DEFS>
set -u
cd /workspace
export V6B_INGEST_ENABLED=false
export V6B_MODAL_ENABLED=false
DEFS=src/tamubot/ingestion/pipeline_v6b/definitions.py
STEMS_FILE="${1:-data/syllabi/_preprocessing_lab/stems_chunked.txt}"
LOG=/tmp/v6b-tag-backfill.log
: > "$LOG"

STEMS=$(grep -v '^[[:space:]]*$' "$STEMS_FILE")
TOTAL=$(printf '%s\n' "$STEMS" | wc -l | tr -d ' ')
CSV=$(printf '%s\n' "$STEMS" | paste -sd, -)

stamp() { date -u +"%H:%M:%S"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
run_step() { lbl="$1"; shift; if "$@" >>"$LOG" 2>&1; then log "OK    $lbl"; else log "FAIL  $lbl"; fi; }

log "=== META indexes (scan 02_chunk over corpus) ==="
run_step meta_boilerplate_reference dagster asset materialize --select v6b_meta_boilerplate_reference -f "$DEFS"
run_step meta_chunk_signature_index dagster asset materialize --select v6b_meta_chunk_signature_index -f "$DEFS"

log "=== TAG backfill: $TOTAL stems as one v6b_tag_job backfill ==="
log "NOTE: needs the Dagster daemon running (it executes backfills). Cap concurrency"
log "      via run_coordinator in dagster.yaml if you want strictly serial runs."
run_step "tag_backfill" dagster job backfill -j v6b_tag_job --partitions "$CSV" -f "$DEFS" --noprompt

log "=== LAUNCHED — watch Runs -> Backfills for the grouped cohort. ==="
