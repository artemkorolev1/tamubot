"""v6b_pipeline_ledger: non-partitioned, report-only asset that renders ONE
table answering "which files ran through the pipeline, and how did each fare at
each step?"

Rows = files (syllabus stems, the dynamic partition keys). Columns = the seven
pipeline stages. Each cell rolls up that file's status at that stage:

    ✓  materialized, all asset checks passed (or the stage has no checks)
    ⚠  materialized, a WARN-severity check failed (non-blocking)
    ✗  materialized, an ERROR-severity check failed (blocking)
    ·  not materialized yet

REPORT ONLY — reads the Dagster instance + event log, never asset outputs, so it
takes NO `deps` and never mutates anything (same shape as v6b_corpus_report).

The status-derivation logic lives in the pure `build_ledger()` helper (no Dagster
imports) so it is unit-testable; `_compute` only does the instance I/O.
"""

import json
from pathlib import Path

from dagster import (
    AssetCheckKey,
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)
from dagster._core.event_api import PartitionKeyFilter

from tamubot.ingestion.pipeline_v5.util import DATA_ROOT
from tamubot.ingestion.pipeline_v6b.partitions import stem_partitions

# (asset key, short column label) in pipeline order. struct is the parallel
# branch off bronze (chunk/tag/embed/atlas is the main chain).
STAGES: list[tuple[str, str]] = [
    ("v6b_bronze_blocks", "bronze"),
    ("v6b_silver_modal", "modal"),
    ("v6b_silver_chunk_semantic", "chunk"),
    ("v6b_silver_tag_semantic", "tag"),
    ("v6b_silver_embed", "embed"),
    ("v6b_silver_atlas_upsert", "atlas"),
    ("v6b_silver_structured", "struct"),
]

# Mirrors definitions.py `asset_checks=[...]`. The check name equals the
# decorated function name (no explicit name= on @asset_check). Keep in sync when
# checks are added/removed.
STAGE_CHECKS: dict[str, list[str]] = {
    "v6b_bronze_blocks": [
        "v6b_bronze_blocks_nonempty",
        "v6b_bronze_blocks_has_text",
        "v6b_bronze_blocks_no_replacement_chars",
        "v6b_bronze_blocks_header_hierarchy_valid",
        "v6b_bronze_blocks_source_integrity",
        "v6b_bronze_blocks_block_count_vs_baseline",
    ],
    "v6b_silver_modal": [
        "v6b_silver_modal_budget_not_exceeded",
        "v6b_silver_modal_result_schema_valid",
    ],
    "v6b_silver_chunk_semantic": [
        "v6b_silver_chunk_count_nonzero",
        "v6b_silver_chunk_low_no_header_rate",
        "v6b_silver_chunk_no_oversized",
        "v6b_silver_chunk_schema_valid",
        "v6b_silver_chunk_total_vs_baseline",
        "v6b_silver_chunk_flagged_rate_vs_baseline",
    ],
    "v6b_silver_tag_semantic": [
        "v6b_silver_tag_chunk_count_preserved",
        "v6b_silver_tag_boilerplate_rate_in_band",
        "v6b_silver_tag_duplicate_rate_in_band",
    ],
    "v6b_silver_embed": [
        "v6b_silver_embed_count_matches_chunks",
        "v6b_silver_embed_model_field_present",
        "v6b_silver_embed_voyage_calls_vs_baseline",
    ],
    "v6b_silver_atlas_upsert": [
        "v6b_silver_atlas_vector_count_matches_chunks",
        "v6b_silver_atlas_index_status_ready",
        "v6b_silver_atlas_index_size_vs_baseline",
        "v6b_silver_atlas_golden_recall_at_5",
    ],
    "v6b_silver_structured": [],
}

GLYPH_MISSING = "·"
GLYPH_PASS = "✓"
GLYPH_WARN = "⚠"
GLYPH_FAIL = "✗"

# A check outcome is a plain dict so build_ledger stays Dagster-free:
#   {"name": str, "status": "SUCCEEDED"|"FAILED"|"PLANNED", "severity": "ERROR"|"WARN"}


def _cell(materialized: bool, outcomes: list[dict]) -> str:
    """Roll up one (stem, stage) into a single glyph. Only failures matter for
    severity; a non-materialized stage is always missing regardless of checks."""
    if not materialized:
        return GLYPH_MISSING
    failed = [o for o in outcomes if o.get("status") == "FAILED"]
    if any(o.get("severity") == "ERROR" for o in failed):
        return GLYPH_FAIL
    if failed:  # remaining failures are WARN-severity
        return GLYPH_WARN
    return GLYPH_PASS


def build_ledger(
    roster: list[str],
    materialized_by_stage: dict[str, set[str]],
    checks_by_stage: dict[str, dict[str, list[dict]]],
) -> dict:
    """Pure builder: turn pre-fetched instance state into the ledger.

    - roster: every registered stem (dynamic partition keys).
    - materialized_by_stage: {stage_asset: {stem materialized, ...}}.
    - checks_by_stage: {stage_asset: {stem: [check-outcome dicts]}}.

    Returns {"rows", "summary", "markdown"}; absent stems/stages are treated as
    missing (never an error) so a registered-but-never-run stem renders cleanly.
    """
    rows: list[dict] = []
    n_passed = n_warn = n_fail = n_not_started = 0

    for stem in sorted(roster):
        cells: dict[str, str] = {}
        failed_checks: list[dict] = []
        any_materialized = False
        for asset_name, _label in STAGES:
            mat = stem in materialized_by_stage.get(asset_name, set())
            any_materialized = any_materialized or mat
            outcomes = checks_by_stage.get(asset_name, {}).get(stem, [])
            cells[asset_name] = _cell(mat, outcomes)
            for o in outcomes:
                if o.get("status") == "FAILED":
                    failed_checks.append({"stage": asset_name, **o})

        glyphs = set(cells.values())
        if not any_materialized:
            n_not_started += 1
            row_status = "not_started"
        elif GLYPH_FAIL in glyphs:
            n_fail += 1
            row_status = "fail"
        elif GLYPH_WARN in glyphs:
            n_warn += 1
            row_status = "warn"
        else:
            n_passed += 1
            row_status = "pass"

        rows.append(
            {
                "stem": stem,
                "status": row_status,
                "cells": cells,
                "failed_checks": failed_checks,
            }
        )

    summary = {
        "files": len(rows),
        "fully_passed": n_passed,
        "with_warnings": n_warn,
        "with_failures": n_fail,
        "not_started": n_not_started,
    }
    return {"rows": rows, "summary": summary, "markdown": _render_markdown(rows, summary)}


def _render_markdown(rows: list[dict], summary: dict) -> str:
    labels = [label for _key, label in STAGES]
    header = "| File | " + " | ".join(labels) + " |"
    divider = "|------|" + "|".join([":--:"] * len(STAGES)) + "|"
    body = ["| " + r["stem"] + " | " + " | ".join(r["cells"][key] for key, _label in STAGES) + " |" for r in rows]
    legend = (
        f"**{summary['files']} files** — "
        f"✓ {summary['fully_passed']} passed · "
        f"⚠ {summary['with_warnings']} warnings · "
        f"✗ {summary['with_failures']} failures · "
        f"· {summary['not_started']} not started\n\n"
        "Legend: ✓ passed · ⚠ WARN check failed · ✗ ERROR check failed · · not materialized"
    )
    return legend + "\n\n" + "\n".join([header, divider, *body])


def _fetch_checks_by_stage(context: AssetExecutionContext, roster: list[str]) -> dict[str, dict[str, list[dict]]]:
    """One event-log call per stem (all check keys batched) -> per (stage, stem)
    list of plain check-outcome dicts."""
    els = context.instance.event_log_storage
    # AssetCheckKey -> stage asset name, so we can bucket results by stage.
    stage_of: dict[AssetCheckKey, str] = {}
    all_keys: list[AssetCheckKey] = []
    for asset_name, check_names in STAGE_CHECKS.items():
        for name in check_names:
            key = AssetCheckKey(asset_key=AssetKey(asset_name), name=name)
            stage_of[key] = asset_name
            all_keys.append(key)

    result: dict[str, dict[str, list[dict]]] = {asset_name: {} for asset_name, _ in STAGES}
    for stem in roster:
        records = els.get_latest_asset_check_execution_by_key(all_keys, partition_filter=PartitionKeyFilter(key=stem))
        for key, record in records.items():
            evaluation = record.evaluation
            severity = evaluation.severity.value if evaluation is not None else "ERROR"
            result[stage_of[key]].setdefault(stem, []).append(
                {"name": key.name, "status": record.status.value, "severity": severity}
            )
    return result


def _compute_pipeline_ledger(context: AssetExecutionContext) -> MaterializeResult:
    roster = list(context.instance.get_dynamic_partitions(stem_partitions.name))

    materialized_by_stage = {
        asset_name: context.instance.get_materialized_partitions(AssetKey(asset_name)) for asset_name, _label in STAGES
    }
    checks_by_stage = _fetch_checks_by_stage(context, roster)

    ledger = build_ledger(roster, materialized_by_stage, checks_by_stage)
    summary = ledger["summary"]

    out_path = Path(DATA_ROOT) / "_meta" / "v6b_pipeline_ledger.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": ledger["rows"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    context.log.info(
        "v6b_pipeline_ledger: %d files (%d passed, %d warn, %d fail, %d not started)",
        summary["files"],
        summary["fully_passed"],
        summary["with_warnings"],
        summary["with_failures"],
        summary["not_started"],
    )

    return MaterializeResult(
        metadata={
            "ledger": MetadataValue.md(ledger["markdown"]),
            "files": summary["files"],
            "fully_passed": summary["fully_passed"],
            "with_warnings": summary["with_warnings"],
            "with_failures": summary["with_failures"],
            "not_started": summary["not_started"],
            "report_path": str(out_path),
        }
    )


v6b_pipeline_ledger = asset(
    name="v6b_pipeline_ledger",
    group_name="v6b_ops",
    description=(
        "Report-only: one table of every file (syllabus stem) × pipeline stage, each cell "
        "rolling up materialization + asset-check status (✓ pass / ⚠ warn / ✗ fail / · missing). "
        "Reads the Dagster instance + event log; no deps, never mutates."
    ),
)(_compute_pipeline_ledger)
