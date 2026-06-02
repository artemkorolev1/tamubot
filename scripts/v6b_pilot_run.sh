#!/bin/sh
# v6b phase2 pilot — 10 CSCE stems, dry-run Atlas. Run inside tamubot-dev-1.
set -u
cd /workspace
export V6B_INGEST_ENABLED=false
export V6B_MODAL_ENABLED=false

DEFS=src/tamubot/ingestion/pipeline_v6b/definitions.py
LOG=/tmp/v6b-pilot-run.log
: > "$LOG"

STEMS="202611_CSCE_608_600_46648 202611_CSCE_611_600_50668 202611_CSCE_612_600_42640 202611_CSCE_614_600_42743 202611_CSCE_625_600_19180_HP 202611_CSCE_629_600_54978 202611_CSCE_633_600_12367_HP 202611_CSCE_635_600_46646 202611_CSCE_641_600_58437 202611_CSCE_648_600_60319"

stamp() { date -u +"%H:%M:%S"; }

run_step() {
    label="$1"; shift
    echo "[$(stamp)] BEGIN $label" | tee -a "$LOG"
    if "$@" >>"$LOG" 2>&1; then
        echo "[$(stamp)] OK    $label" | tee -a "$LOG"
    else
        echo "[$(stamp)] FAIL  $label (continuing)" | tee -a "$LOG"
    fi
}

echo "=== PASS 1: bronze -> modal -> chunk ===" | tee -a "$LOG"
for s in $STEMS; do
    run_step "bronze+modal+chunk $s" dagster asset materialize \
        --select v6b_bronze_blocks,v6b_silver_modal,v6b_silver_chunk_semantic \
        --partition "$s" -f "$DEFS"
done

echo "=== meta_boilerplate_reference (global) ===" | tee -a "$LOG"
run_step "meta_boilerplate_reference" dagster asset materialize \
    --select v6b_meta_boilerplate_reference -f "$DEFS"

echo "=== PASS 2a: silver_tag (bp ref only) ===" | tee -a "$LOG"
for s in $STEMS; do
    run_step "silver_tag-pass1 $s" dagster asset materialize \
        --select v6b_silver_tag_semantic --partition "$s" -f "$DEFS"
done

echo "=== meta_chunk_signature_index (global) ===" | tee -a "$LOG"
run_step "meta_chunk_signature_index" dagster asset materialize \
    --select v6b_meta_chunk_signature_index -f "$DEFS"

echo "=== PASS 2b: silver_tag (full three-pass) ===" | tee -a "$LOG"
for s in $STEMS; do
    run_step "silver_tag-pass2 $s" dagster asset materialize \
        --select v6b_silver_tag_semantic --partition "$s" -f "$DEFS"
done

echo "=== PASS 3: embed + atlas (dry-run) + structured ===" | tee -a "$LOG"
for s in $STEMS; do
    run_step "embed+atlas+structured $s" dagster asset materialize \
        --select v6b_silver_embed,v6b_silver_atlas_upsert,v6b_silver_structured \
        --partition "$s" -f "$DEFS"
done

echo "=== corpus_report (global) ===" | tee -a "$LOG"
run_step "corpus_report" dagster asset materialize \
    --select v6b_corpus_report -f "$DEFS"

echo "=== DONE $(stamp) ===" | tee -a "$LOG"
