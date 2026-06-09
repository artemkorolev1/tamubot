#!/bin/sh
# v6b tag-only phase — build the two corpus meta indexes (scan 02_chunk) then run
# silver_tag over already-chunked stems. No GPU. Use after PASS1 chunking exists.
#   sh scripts/v6b_tag_phase.sh [STEMS_FILE]
set -u
cd /workspace
export V6B_INGEST_ENABLED=false
export V6B_MODAL_ENABLED=false
DEFS=src/tamubot/ingestion/pipeline_v6b/definitions.py
STEMS_FILE="${1:-data/syllabi/_preprocessing_lab/stems_chunked.txt}"
LOG=/tmp/v6b-tag-phase.log
: > "$LOG"
STEMS=$(grep -v '^[[:space:]]*$' "$STEMS_FILE")
TOTAL=$(printf '%s\n' "$STEMS" | wc -l | tr -d ' ')
stamp() { date -u +"%H:%M:%S"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
run() { lbl="$1"; shift; if "$@" >>"$LOG" 2>&1; then log "OK   $lbl"; else log "FAIL $lbl"; fi; }

log "=== META indexes (scan 02_chunk over corpus) ==="
run meta_boilerplate_reference dagster asset materialize --select v6b_meta_boilerplate_reference -f "$DEFS"
run meta_chunk_signature_index dagster asset materialize --select v6b_meta_chunk_signature_index -f "$DEFS"

log "=== TAG $TOTAL stems (bp + cross-syllabus dedup) ==="
i=0
for s in $STEMS; do
    i=$((i+1))
    log "BEGIN [$i/$TOTAL] $s"
    run "TAG $s" dagster asset materialize --select v6b_silver_tag_semantic --partition "$s" -f "$DEFS"
done
NT=$(printf '%s\n' "$STEMS" | while read s; do d=$(echo "$s" | sed -E 's/^[0-9]+_([A-Z]+)_.*/\1/'); [ -f "data/syllabi/$d/v6b/silver/03_tag/$s.json" ] && echo x; done | wc -l | tr -d ' ')
log "=== DONE: $NT/$TOTAL tagged ==="
