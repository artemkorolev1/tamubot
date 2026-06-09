#!/bin/sh
# v6b eval-corpus materialization — drive the 100-doc preprocessing eval set through
# bronze -> modal -> chunk -> (corpus meta indexes) -> tag, honoring the two-phase
# cross-syllabus dedup ordering (project-v6b-dedup-two-phase).
#
# Collapsed 2-pass (vs the pilot's 3-pass): both meta indexes scan 02_chunk, not 03_tag,
# so they are built once after chunking and a single tag pass picks up bp + dup together.
#
# Usage (inside tamubot-dev-1, cwd /workspace):
#   sh scripts/v6b_eval_corpus_run.sh [STEMS_FILE]
# STEMS_FILE: one canonical_stem per line (default: the eval-corpus 100).
set -u
cd /workspace
export V6B_INGEST_ENABLED=false   # no live ingest side effects
export V6B_MODAL_ENABLED=false    # local vLLM NuExtract sidecar, not Modal cloud

DEFS=src/tamubot/ingestion/pipeline_v6b/definitions.py
STEMS_FILE="${1:-data/syllabi/_preprocessing_lab/stems_all.txt}"
LOG=/tmp/v6b-eval-corpus-run.log
: > "$LOG"

STEMS=$(grep -v '^[[:space:]]*$' "$STEMS_FILE")
TOTAL=$(printf '%s\n' "$STEMS" | wc -l | tr -d ' ')

stamp() { date -u +"%H:%M:%S"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

run_step() {
    label="$1"; shift
    if "$@" >>"$LOG" 2>&1; then
        log "OK    $label"
    else
        log "FAIL  $label (continuing)"
    fi
}

dept_of() { echo "$1" | sed -E 's/^[0-9]+_([A-Z]+)_.*/\1/'; }
chunk_path() { echo "data/syllabi/$(dept_of "$1")/v6b/silver/02_chunk/$1.json"; }
tag_path()   { echo "data/syllabi/$(dept_of "$1")/v6b/silver/03_tag/$1.json"; }

log "=== START eval-corpus run: $TOTAL stems from $STEMS_FILE ==="

# ----------------------------------------------------------------- PASS 1
log "=== PASS 1: bronze -> modal -> chunk (skip stems with existing 02_chunk) ==="
i=0
for s in $STEMS; do
    i=$((i+1))
    if [ -f "$(chunk_path "$s")" ]; then
        log "skip  [$i/$TOTAL] $s (02_chunk exists)"
        continue
    fi
    log "BEGIN [$i/$TOTAL] PASS1 $s"
    run_step "PASS1 $s" dagster asset materialize \
        --select v6b_bronze_blocks,v6b_silver_modal,v6b_silver_chunk_semantic \
        --partition "$s" -f "$DEFS"
done

# ----------------------------------------------------- corpus meta indexes (global)
log "=== META: boilerplate_reference + chunk_signature_index (scan 02_chunk) ==="
run_step "meta_boilerplate_reference" dagster asset materialize \
    --select v6b_meta_boilerplate_reference -f "$DEFS"
run_step "meta_chunk_signature_index" dagster asset materialize \
    --select v6b_meta_chunk_signature_index -f "$DEFS"

# ----------------------------------------------------------------- PASS 2 (tag all)
log "=== PASS 2: silver_tag (bp + cross-syllabus dup; re-tags all $TOTAL) ==="
i=0
for s in $STEMS; do
    i=$((i+1))
    log "BEGIN [$i/$TOTAL] TAG $s"
    run_step "TAG $s" dagster asset materialize \
        --select v6b_silver_tag_semantic --partition "$s" -f "$DEFS"
done

NTAG=$(printf '%s\n' "$STEMS" | while read s; do [ -f "$(tag_path "$s")" ] && echo x; done | wc -l | tr -d ' ')
log "=== DONE: $NTAG/$TOTAL stems have 03_tag ==="
